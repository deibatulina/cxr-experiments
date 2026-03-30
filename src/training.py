from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .evaluation import evaluate_split, load_model_and_processor, load_saved_generation
from .utils import append_jsonl, clear_memory, load_json, save_json, to_rgb


def freeze_model_for_training(model: Any) -> None:
    """Замораживает базовую модель и оставляет обучаемыми только нужные блоки BLIP-2."""

    for parameter in model.parameters():
        parameter.requires_grad = False

    if hasattr(model, "language_projection"):
        for parameter in model.language_projection.parameters():
            parameter.requires_grad = True

    if hasattr(model, "query_tokens"):
        model.query_tokens.requires_grad_(True)

    if hasattr(model, "qformer"):
        for parameter in model.qformer.parameters():
            parameter.requires_grad = True


def preprocess_split(
    split: Any,
    processor: Any,
    image_field: str,
    target_field: str,
    prompt: str,
    max_label_length: int,
) -> Any:
    """Преобразует сплит датасета в токенизированные входы модели и labels."""

    def preprocess_one(sample):
        image = to_rgb(sample[image_field])
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        labels = processor.tokenizer(
            str(sample[target_field]),
            padding="max_length",
            truncation=True,
            max_length=max_label_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        labels = torch.where(
            labels == processor.tokenizer.pad_token_id,
            torch.tensor(-100, dtype=labels.dtype),
            labels,
        )
        inputs["labels"] = labels
        return inputs

    return split.map(preprocess_one, remove_columns=split.column_names)


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Объединяет токенизированные объекты в mini-batch для PyTorch."""

    return {
        "pixel_values": torch.stack([torch.tensor(item["pixel_values"]) for item in batch]),
        "input_ids": torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        "attention_mask": torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
        "labels": torch.tensor([item["labels"] for item in batch], dtype=torch.long),
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    """Переносит все тензоры батча на выбранное устройство."""

    result: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        result[key] = value.to(device)
    return result


def plot_training_loss_curve(
    output_path: Path,
    dataset_name: str,
    train_losses: list[float],
    val_losses: list[float],
) -> None:
    """Строит и сохраняет график train/validation loss по эпохам."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="Train loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="s", linestyle="--", label="Validation loss")
    plt.title(f"{dataset_name}: loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.show()
    plt.close()


def plot_learning_curve(
    output_path: Path,
    dataset_name: str,
    learning_metrics: list[dict[str, float | int]],
) -> None:
    """Строит и сохраняет динамику лексических метрик на validation по эпохам."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(
        [item["epoch"] for item in learning_metrics],
        [item["ROUGE-L"] for item in learning_metrics],
        marker="o",
        label="ROUGE-L",
    )
    plt.plot(
        [item["epoch"] for item in learning_metrics],
        [item["BLEU-4"] for item in learning_metrics],
        marker="s",
        label="BLEU-4",
    )
    plt.title(f"{dataset_name}: learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.show()
    plt.close()


def load_saved_training_result(run_dir: Path) -> dict[str, Any] | None:
    """Загружает сохранённые результаты fine-tuning, если запуск уже был завершён."""

    generation_bundle = load_saved_generation(run_dir, "fine_tuned_test")
    if generation_bundle is None:
        return None

    metrics, predictions, references = generation_bundle
    history = load_json(run_dir / "fine_tuned_training_history.json", default={})
    best_summary = load_json(run_dir / "best_epoch_summary.json", default={})
    return {
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
        "train_loss": history.get("train_loss", []),
        "val_loss": history.get("val_loss", []),
        "learning_metrics": history.get("learning_metrics", []),
        "best_epoch": history.get("best_epoch", best_summary.get("best_epoch")),
        "best_val_loss": history.get("best_val_loss", best_summary.get("best_val_loss")),
        "tensorboard_dir": history.get("tensorboard_dir", str(run_dir / "tensorboard")),
        "log_file": history.get("log_file", str(run_dir / "fine_tuned_training_log.jsonl")),
        "history_status": history.get("status", "completed"),
    }


def run_fine_tuning(
    state: dict[str, Any],
    experiment: dict[str, Any],
    splits: Any,
    run_dir: Path,
) -> dict[str, Any]:
    """Запускает fine-tuning, выбирает лучшую эпоху и оценивает модель на test."""

    config = state["config"]
    platform = state["platform"]
    dataset_spec = state["dataset_specs"][experiment["dataset_name"]]
    image_field = dataset_spec["image_field"]
    target_field = dataset_spec["target_field"]
    num_epochs = config["num_epochs_by_experiment"].get(experiment["name"], 2)

    run_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = run_dir / "tensorboard"
    log_file = run_dir / "fine_tuned_training_log.jsonl"
    history_file = run_dir / "fine_tuned_training_history.json"
    best_model_dir = run_dir / "best_model"
    last_model_dir = run_dir / "last_model"
    validation_dir = run_dir / "validation_by_epoch"
    plot_dir = state["plots_root_dir"] / experiment["artifact_dir_name"]
    validation_plot_dir = plot_dir / "validation_by_epoch"

    log_file.write_text("", encoding="utf-8")
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    print(f"\nStarting fine-tuning for: {experiment['title']}")
    print(f"TensorBoard log dir: {tensorboard_dir}")
    print(f"Training log file: {log_file}")
    print(f"Training dtype: {config['train_dtype']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Max grad norm: {config['max_grad_norm']}")
    print(f"Num epochs: {num_epochs}")

    processor, model = load_model_and_processor(
        config["model_name"],
        platform,
        model_dtype=config["train_dtype"],
    )
    freeze_model_for_training(model)

    train_data = preprocess_split(
        splits["train"],
        processor,
        image_field,
        target_field,
        config["prompt"],
        config["max_label_length"],
    )
    val_data = preprocess_split(
        splits["validation"],
        processor,
        image_field,
        target_field,
        config["prompt"],
        config["max_label_length"],
    )

    train_loader = DataLoader(
        train_data,
        batch_size=platform["train_batch_size"],
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=platform["eval_batch_size"],
        shuffle=False,
        collate_fn=collate_batch,
    )

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    train_losses = []
    val_losses = []
    learning_metrics = []
    global_step = 0
    best_val_loss = float("inf")
    best_epoch = None

    def save_training_state(status: str) -> None:
        """Сохраняет текущее состояние истории обучения после ключевых этапов."""

        save_json(
            history_file,
            {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "learning_metrics": learning_metrics,
                "best_epoch": best_epoch,
                "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
                "completed_epochs": len(train_losses),
                "requested_epochs": num_epochs,
                "status": status,
                "tensorboard_dir": str(tensorboard_dir),
                "log_file": str(log_file),
            },
        )

    try:
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0.0
            optimizer.zero_grad()

            progress = tqdm(train_loader, desc=f"{experiment['name']} train epoch {epoch + 1}", leave=False)
            for step, batch in enumerate(progress):
                batch = move_batch_to_device(batch, platform["device"])
                outputs = model(**batch)
                raw_loss = outputs.loss
                if torch.isnan(raw_loss) or torch.isinf(raw_loss):
                    raise ValueError(
                        f"Loss became invalid at epoch {epoch + 1}, step {step + 1}: {raw_loss}"
                    )

                loss = raw_loss / platform["grad_accum_steps"]
                loss.backward()
                epoch_progress = epoch + (step + 1) / max(len(train_loader), 1)

                if (step + 1) % platform["grad_accum_steps"] == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        config["max_grad_norm"],
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
                    writer.add_scalar("train/step_loss", float(raw_loss.item()), global_step)
                    writer.add_scalar("train/epoch_progress", float(epoch_progress), global_step)
                    append_jsonl(
                        log_file,
                        {
                            "type": "step",
                            "experiment": experiment["name"],
                            "global_step": global_step,
                            "epoch_progress": float(epoch_progress),
                            "loss": float(raw_loss.item()),
                        },
                    )

                total_train_loss += raw_loss.item()
                progress.set_postfix(loss=float(raw_loss.item()), epoch_progress=f"{epoch_progress:.3f}")

            train_loss = total_train_loss / max(len(train_loader), 1)
            train_losses.append(train_loss)

            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"{experiment['name']} val epoch {epoch + 1}", leave=False):
                    batch = move_batch_to_device(batch, platform["device"])
                    outputs = model(**batch)
                    total_val_loss += outputs.loss.item()

            val_loss = total_val_loss / max(len(val_loader), 1)
            val_losses.append(val_loss)

            val_metrics, _, _ = evaluate_split(
                model,
                processor,
                splits["validation"],
                image_field,
                target_field,
                platform,
                validation_dir,
                f"epoch_{epoch + 1}_validation",
                config["prompt"],
                plot_output_path=validation_plot_dir / f"epoch_{epoch + 1}_validation_lengths.png",
            )
            learning_metrics.append(
                {
                    "epoch": epoch + 1,
                    "BLEU-4": val_metrics["BLEU-4"],
                    "ROUGE-L": val_metrics["ROUGE-L"],
                    "METEOR": val_metrics["METEOR"],
                }
            )

            writer.add_scalar("epoch/train_loss", train_loss, epoch + 1)
            writer.add_scalar("epoch/val_loss", val_loss, epoch + 1)
            writer.add_scalar("epoch/val_BLEU-4", val_metrics["BLEU-4"], epoch + 1)
            writer.add_scalar("epoch/val_ROUGE-L", val_metrics["ROUGE-L"], epoch + 1)
            writer.add_scalar("epoch/val_METEOR", val_metrics["METEOR"], epoch + 1)

            append_jsonl(
                log_file,
                {
                    "type": "epoch",
                    "experiment": experiment["name"],
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "BLEU-4": float(val_metrics["BLEU-4"]),
                    "ROUGE-L": float(val_metrics["ROUGE-L"]),
                    "METEOR": float(val_metrics["METEOR"]),
                },
            )

            print(f"\nEpoch {epoch + 1}/{num_epochs} summary for {experiment['title']}")
            print(f"train_loss: {train_loss:.4f}")
            print(f"val_loss:   {val_loss:.4f}")
            print(f"BLEU-4:    {val_metrics['BLEU-4']:.4f}")
            print(f"ROUGE-L:   {val_metrics['ROUGE-L']:.4f}")
            print(f"METEOR:    {val_metrics['METEOR']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                model.save_pretrained(best_model_dir)
                processor.save_pretrained(best_model_dir)
                save_json(
                    run_dir / "best_epoch_summary.json",
                    {
                        "best_epoch": best_epoch,
                        "best_val_loss": float(best_val_loss),
                        "experiment": experiment["name"],
                    },
                )
                print(f"Best model updated at epoch {best_epoch}")

            save_training_state("running")

        model.save_pretrained(last_model_dir)
        processor.save_pretrained(last_model_dir)
        save_training_state("finished_training")

        plot_training_loss_curve(plot_dir / "fine_tuned_loss_curve.png", experiment["title"], train_losses, val_losses)
        plot_learning_curve(plot_dir / "fine_tuned_learning_curve.png", experiment["title"], learning_metrics)

        print(f"\nLoading best model from epoch {best_epoch} for final test evaluation")
        del model
        clear_memory(platform["device"])

        processor, model = load_model_and_processor(
            best_model_dir,
            platform,
            model_dtype=config["train_dtype"],
        )
        test_metrics, test_predictions, test_references = evaluate_split(
            model,
            processor,
            splits["test"],
            image_field,
            target_field,
            platform,
            run_dir,
            "fine_tuned_test",
            config["prompt"],
            plot_output_path=plot_dir / "fine_tuned_test_lengths.png",
        )

        save_training_state("completed")
        writer.close()
        del model
        clear_memory(platform["device"])

        return {
            "metrics": test_metrics,
            "predictions": test_predictions,
            "references": test_references,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "learning_metrics": learning_metrics,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "tensorboard_dir": str(tensorboard_dir),
            "log_file": str(log_file),
            "history_status": "completed",
        }
    except KeyboardInterrupt:
        print("\nTraining was interrupted manually. Saving the current state.")
        model.save_pretrained(last_model_dir)
        processor.save_pretrained(last_model_dir)
        save_training_state("interrupted")
        writer.close()
        del model
        clear_memory(platform["device"])
        return {
            "metrics": None,
            "predictions": None,
            "references": None,
            "train_loss": train_losses,
            "val_loss": val_losses,
            "learning_metrics": learning_metrics,
            "best_epoch": best_epoch,
            "best_val_loss": None if best_val_loss == float("inf") else float(best_val_loss),
            "tensorboard_dir": str(tensorboard_dir),
            "log_file": str(log_file),
            "history_status": "interrupted",
        }
