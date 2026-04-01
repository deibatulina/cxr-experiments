from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

from .utils import clear_memory, load_json, print_torchinfo_summary, save_json, to_rgb


@lru_cache(maxsize=1)
def load_metric_objects() -> dict[str, Any]:
    """Кэширует объекты метрик, чтобы не загружать их повторно."""

    return {
        "bleu": evaluate.load("bleu"),
        "rouge": evaluate.load("rouge"),
        "meteor": evaluate.load("meteor"),
    }


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Вычисляет лексические метрики и базовые статистики длины текстов."""

    metrics = load_metric_objects()
    bleu_score = metrics["bleu"].compute(predictions=predictions, references=references)["bleu"]
    rouge_scores = metrics["rouge"].compute(predictions=predictions, references=references)
    meteor_score = metrics["meteor"].compute(predictions=predictions, references=references)["meteor"]
    return {
        "BLEU-4": float(bleu_score),
        "ROUGE-1": float(rouge_scores["rouge1"]),
        "ROUGE-L": float(rouge_scores["rougeL"]),
        "METEOR": float(meteor_score),
        "avg_prediction_length": float(np.mean([len(text.split()) for text in predictions] or [0.0])),
        "avg_reference_length": float(np.mean([len(text.split()) for text in references] or [0.0])),
    }


def load_model_and_processor(
    model_source: str | Path,
    platform: dict[str, Any],
    model_dtype: torch.dtype | None = None,
) -> tuple[Blip2Processor, Blip2ForConditionalGeneration]:
    """Загружает процессор и модель BLIP-2 с настройкой под выбранное устройство."""

    processor = Blip2Processor.from_pretrained(model_source)
    if model_dtype is None:
        model_dtype = platform["dtype"]

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_source,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.language_model.config.use_cache = False
    if platform["device"] == "mps":
        model.gradient_checkpointing_enable()
    model = model.to(platform["device"])
    print_torchinfo_summary(model, title=f"Сводка модели через torchinfo: {model_source}")
    return processor, model


def generate_prediction(
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    image: Any,
    platform: dict[str, Any],
    prompt: str,
) -> str:
    """Генерирует одно текстовое заключение для одного изображения."""

    inputs = processor(images=image, text=prompt, return_tensors="pt").to(platform["device"])
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=min(platform["max_new_tokens"], 24),
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_beams=1,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            early_stopping=True,
        )
    return processor.decode(output[0], skip_special_tokens=True).strip()


def plot_text_length_distribution(
    predictions: list[str],
    references: list[str],
    title: str,
    output_path: Path,
) -> None:
    """Строит и сохраняет распределения длин предсказаний и референсов."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist([len(text.split()) for text in predictions], bins=20, alpha=0.6, label="Predictions")
    plt.hist([len(text.split()) for text in references], bins=20, alpha=0.6, label="References")
    plt.title(title)
    plt.xlabel("Words")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.show()
    plt.close()


def evaluate_split(
    model: Blip2ForConditionalGeneration,
    processor: Blip2Processor,
    split: Any,
    image_field: str,
    target_field: str,
    platform: dict[str, Any],
    output_dir: Path,
    prefix: str,
    prompt: str,
    plot_output_path: Path | None = None,
) -> tuple[dict[str, float], list[str], list[str]]:
    """Запускает генерацию на сплите, сохраняет артефакты и возвращает метрики и тексты."""

    model.eval()
    predictions = []
    references = []
    records = []
    predictions_path = output_dir / f"{prefix}_predictions.json"

    for index, sample in enumerate(tqdm(split, desc=prefix, leave=False)):
        image = to_rgb(sample[image_field])
        reference = str(sample[target_field]).strip()
        prediction = generate_prediction(model, processor, image, platform, prompt)
        predictions.append(prediction)
        references.append(reference)
        records.append(
            {
                "index": index,
                "prediction": prediction,
                "reference": reference,
            }
        )

        if (index + 1) % 100 == 0:
            save_json(predictions_path, records)

    save_json(predictions_path, records)
    metrics = compute_metrics(predictions, references)
    save_json(output_dir / f"{prefix}_metrics.json", metrics)
    if plot_output_path is not None:
        plot_text_length_distribution(
            predictions,
            references,
            f"{prefix}: text length distribution",
            plot_output_path,
        )
    return metrics, predictions, references


def load_saved_generation(
    output_dir: Path,
    prefix: str,
    fallback_dirs: list[Path] | None = None,
) -> tuple[dict[str, Any], list[str], list[str]] | None:
    """Загружает ранее сохранённые предсказания и метрики для запуска генерации."""

    candidate_dirs = [output_dir]
    if fallback_dirs:
        candidate_dirs.extend(fallback_dirs)

    for candidate_dir in candidate_dirs:
        predictions_path = candidate_dir / f"{prefix}_predictions.json"
        metrics_path = candidate_dir / f"{prefix}_metrics.json"
        if not predictions_path.exists() or not metrics_path.exists():
            continue

        records = load_json(predictions_path, default=[])
        metrics = load_json(metrics_path, default={})
        predictions = [row["prediction"] for row in records]
        references = [row["reference"] for row in records]
        return metrics, predictions, references

    return None


def evaluate_saved_best_model(
    state: dict[str, Any],
    experiment: dict[str, Any],
    run_dir: Path,
    get_experiment_splits: Any,
) -> dict[str, Any]:
    """Оценивает ранее сохранённую лучшую fine-tuned модель на test-сплите."""

    best_model_dir = run_dir / "best_model"
    if not best_model_dir.exists():
        raise FileNotFoundError(f"Saved model was not found in {best_model_dir}")

    splits = get_experiment_splits(state, experiment)
    dataset_spec = state["dataset_specs"][experiment["dataset_name"]]
    processor, model = load_model_and_processor(
        best_model_dir,
        state["platform"],
        model_dtype=state["config"]["train_dtype"],
    )
    try:
        metrics, predictions, references = evaluate_split(
            model,
            processor,
            splits["test"],
            dataset_spec["image_field"],
            dataset_spec["target_field"],
            state["platform"],
            run_dir,
            "fine_tuned_test",
            state["config"]["prompt"],
            plot_output_path=state["plots_root_dir"] / experiment["artifact_dir_name"] / "fine_tuned_test_lengths.png",
        )
    finally:
        del model
        clear_memory(state["platform"]["device"])

    return {
        "metrics": metrics,
        "predictions": predictions,
        "references": references,
    }
