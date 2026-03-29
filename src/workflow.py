import os
import random
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import set_seed

from .config import DEFAULT_RUN_MODES, build_dataset_specs, build_experiments, create_runtime_config
from .datasets import build_balanced_iu_splits, is_no_acute_style, load_dataset_splits
from .evaluation import evaluate_saved_best_model, evaluate_split, load_model_and_processor, load_saved_generation
from .training import load_saved_training_result, run_fine_tuning
from .utils import clear_memory, limit_split, load_json, prepare_run_dir_for_rerun, sanitize_model_dir_name


os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib").resolve()))


DEFAULT_PATHOLOGY_TERMS = {
    "cardiomegaly": ["cardiomegaly", "cardiac enlargement", "enlarged heart"],
    "effusion": ["effusion", "effusions", "pleural effusion", "pleural effusions"],
    "edema": ["edema", "oedema", "pulmonary edema"],
    "atelectasis": ["atelectasis", "atelectatic"],
    "pneumonia": ["pneumonia"],
    "opacity": ["opacity", "opacities", "infiltrate", "infiltrates"],
    "congestion": ["congestion", "vascular congestion", "pulmonary vascular congestion"],
    "pneumothorax": ["pneumothorax"],
    "fracture": ["fracture", "fractures"],
    "emphysema": ["emphysema"],
}


def resolve_platform():
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "dtype": torch.float16,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "grad_accum_steps": 2,
            "max_new_tokens": 96,
            "num_beams": 3,
        }

    if torch.backends.mps.is_available():
        torch.set_float32_matmul_precision("high")
        return {
            "device": "mps",
            "dtype": torch.float16,
            "train_batch_size": 1,
            "eval_batch_size": 1,
            "grad_accum_steps": 8,
            "max_new_tokens": 80,
            "num_beams": 2,
        }

    return {
        "device": "cpu",
        "dtype": torch.float32,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "grad_accum_steps": 1,
        "max_new_tokens": 64,
        "num_beams": 2,
    }


def initialize_runtime(config=None):
    if config is None:
        config = create_runtime_config()

    os.environ["MPLCONFIGDIR"] = str((Path(config["results_root_dir"]) / ".matplotlib").resolve())
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    load_dotenv()
    hf_token = os.getenv("HF_ACCESS_TOKEN")
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=False)
        except Exception as error:
            print(f"Skipping Hugging Face login: {error}")

    set_seed(config["seed"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])

    artifact_root_dir = Path(config["results_root_dir"]) / sanitize_model_dir_name(config["model_name"])
    artifact_root_dir.mkdir(parents=True, exist_ok=True)
    plots_root_dir = artifact_root_dir / "plots"
    plots_root_dir.mkdir(parents=True, exist_ok=True)

    return {
        "config": config,
        "platform": resolve_platform(),
        "artifact_root_dir": artifact_root_dir,
        "plots_root_dir": plots_root_dir,
        "dataset_specs": build_dataset_specs(),
        "experiments": build_experiments(build_balanced_iu_splits),
        "dataset_cache": {},
        "experiment_split_cache": {},
    }


def get_base_dataset_splits(state, dataset_name):
    if dataset_name not in state["dataset_cache"]:
        dataset_spec = state["dataset_specs"][dataset_name]
        state["dataset_cache"][dataset_name] = load_dataset_splits(dataset_spec, state["config"])
    return state["dataset_cache"][dataset_name]


def get_experiment_splits(state, experiment):
    if experiment["name"] in state["experiment_split_cache"]:
        return state["experiment_split_cache"][experiment["name"]]

    base_splits = get_base_dataset_splits(state, experiment["dataset_name"])
    split_builder = experiment.get("split_builder")
    if split_builder is None:
        state["experiment_split_cache"][experiment["name"]] = base_splits
    else:
        dataset_spec = state["dataset_specs"][experiment["dataset_name"]]
        state["experiment_split_cache"][experiment["name"]] = split_builder(
            base_splits,
            dataset_spec,
            state["config"]["seed"],
        )
    return state["experiment_split_cache"][experiment["name"]]


def get_artifact_dir(state, experiment):
    return state["artifact_root_dir"] / experiment["artifact_dir_name"]


def get_zero_shot_dir(state, experiment):
    return get_artifact_dir(state, experiment) / "zero_shot"


def get_fine_tuned_dir(state, experiment):
    return get_artifact_dir(state, experiment) / "fine_tuned"


def get_plot_dir(state, experiment=None):
    if experiment is None:
        return state["plots_root_dir"]
    return state["plots_root_dir"] / experiment["artifact_dir_name"]


def build_result_record(experiment, run_mode, source, output_dir, payload):
    result = {
        "name": experiment["name"],
        "kind": experiment["kind"],
        "dataset_name": experiment["dataset_name"],
        "artifact_dir_name": experiment["artifact_dir_name"],
        "title": experiment["title"],
        "run_mode": run_mode,
        "source": source,
        "output_dir": str(output_dir),
    }
    result.update(payload)
    return result


def run_zero_shot_experiment(state, experiment, run_mode="auto"):
    output_dir = get_zero_shot_dir(state, experiment)
    dataset_dir = get_artifact_dir(state, experiment)
    plot_dir = get_plot_dir(state, experiment)

    if run_mode in {"auto", "reuse"}:
        saved = load_saved_generation(output_dir, "zero_shot", fallback_dirs=[dataset_dir])
        if saved is not None:
            metrics, predictions, references = saved
            return build_result_record(
                experiment,
                run_mode,
                "saved_artifacts",
                output_dir,
                {
                    "metrics": metrics,
                    "predictions": predictions,
                    "references": references,
                },
            )
        if run_mode == "reuse":
            raise FileNotFoundError(f"Saved zero-shot artifacts were not found in {output_dir}")

    splits = get_experiment_splits(state, experiment)
    dataset_spec = state["dataset_specs"][experiment["dataset_name"]]
    split = limit_split(splits[experiment["split_name"]], state["config"]["data_limits"]["zero_shot"])
    output_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_model_and_processor(state["config"]["model_name"], state["platform"])
    try:
        metrics, predictions, references = evaluate_split(
            model,
            processor,
            split,
            dataset_spec["image_field"],
            dataset_spec["target_field"],
            state["platform"],
            output_dir,
            "zero_shot",
            state["config"]["prompt"],
            plot_output_path=plot_dir / "zero_shot_lengths.png",
        )
    finally:
        del model
        clear_memory(state["platform"]["device"])

    return build_result_record(
        experiment,
        run_mode,
        "fresh_run",
        output_dir,
        {
            "metrics": metrics,
            "predictions": predictions,
            "references": references,
        },
    )


def run_fine_tuned_experiment(state, experiment, run_mode="auto"):
    run_dir = get_fine_tuned_dir(state, experiment)

    if run_mode in {"auto", "reuse"}:
        saved = load_saved_training_result(run_dir)
        if saved is not None:
            return build_result_record(experiment, run_mode, "saved_artifacts", run_dir, saved)

        if (run_dir / "best_model").exists():
            evaluated = evaluate_saved_best_model(state, experiment, run_dir, get_experiment_splits)
            saved = load_saved_training_result(run_dir) or {}
            saved.update(evaluated)
            return build_result_record(experiment, run_mode, "saved_model_evaluation", run_dir, saved)

        if run_mode == "reuse":
            raise FileNotFoundError(f"Saved fine-tuning artifacts or best model were not found in {run_dir}")

    if run_mode == "rerun":
        run_dir.mkdir(parents=True, exist_ok=True)
        prepare_run_dir_for_rerun(run_dir)

    splits = get_experiment_splits(state, experiment)
    payload = run_fine_tuning(state, experiment, splits, run_dir)
    return build_result_record(experiment, run_mode, "fresh_run", run_dir, payload)


def run_experiment(state, experiment_name, run_mode="auto"):
    experiment = state["experiments"][experiment_name]
    if experiment["kind"] == "zero_shot":
        return run_zero_shot_experiment(state, experiment, run_mode)
    return run_fine_tuned_experiment(state, experiment, run_mode)


def run_experiments(state, experiment_names=None, run_modes=None):
    if experiment_names is None:
        experiment_names = list(state["experiments"].keys())
    if run_modes is None:
        run_modes = DEFAULT_RUN_MODES

    results = {}
    for experiment_name in experiment_names:
        run_mode = run_modes.get(experiment_name, "auto")
        print(f"\nRunning {experiment_name} [{run_mode}]")
        print("=" * 60)
        results[experiment_name] = run_experiment(state, experiment_name, run_mode)
    return results


def filter_results_by_kind(results, kind):
    return {name: result for name, result in results.items() if result["kind"] == kind}


def print_result_sources(results):
    print("Experiment sources")
    print("-" * 30)
    for experiment_name, result in results.items():
        print(f"{experiment_name:<24} {result['source']}")


def collect_metric_comparison(results):
    return {name: result["metrics"] for name, result in results.items() if result.get("metrics") is not None}


def text_contains_any(text, keywords):
    text = str(text).lower()
    return any(keyword in text for keyword in keywords)


def get_present_pathology_terms(text, pathology_terms=None):
    if pathology_terms is None:
        pathology_terms = DEFAULT_PATHOLOGY_TERMS

    present_terms = []
    for term_name, keywords in pathology_terms.items():
        if text_contains_any(text, keywords):
            present_terms.append(term_name)
    return present_terms


def analyze_prediction_behavior(result, pathology_terms=None):
    if pathology_terms is None:
        pathology_terms = DEFAULT_PATHOLOGY_TERMS

    predictions = result.get("predictions") or []
    references = result.get("references") or []
    total = len(predictions)
    counter = Counter(predictions)

    if total == 0:
        return {
            "samples": 0,
            "unique_predictions": 0,
            "unique_prediction_ratio": 0.0,
            "top5_share": 0.0,
            "no_acute_predictions": 0,
            "no_acute_rate": 0.0,
            "abnormal_reference_count": 0,
            "no_acute_on_abnormal_count": 0,
            "no_acute_on_abnormal_rate": 0.0,
            "any_pathology_reference_count": 0,
            "any_pathology_prediction_count": 0,
            "any_pathology_recall": 0.0,
            "top_predictions": [],
            "term_support": {},
            "term_recall": {},
        }

    no_acute_predictions = sum(is_no_acute_style(prediction) for prediction in predictions)
    abnormal_reference_indexes = [
        index for index, reference in enumerate(references) if not is_no_acute_style(reference)
    ]
    no_acute_on_abnormal_count = sum(
        is_no_acute_style(predictions[index]) for index in abnormal_reference_indexes
    )

    any_pathology_reference_count = 0
    any_pathology_prediction_count = 0
    any_pathology_recall_hits = 0
    term_support = {term_name: 0 for term_name in pathology_terms}
    term_hits = {term_name: 0 for term_name in pathology_terms}

    for prediction, reference in zip(predictions, references):
        prediction_terms = get_present_pathology_terms(prediction, pathology_terms)
        reference_terms = get_present_pathology_terms(reference, pathology_terms)

        if prediction_terms:
            any_pathology_prediction_count += 1

        if reference_terms:
            any_pathology_reference_count += 1
            if prediction_terms:
                any_pathology_recall_hits += 1

        for term_name in reference_terms:
            term_support[term_name] += 1
            if term_name in prediction_terms:
                term_hits[term_name] += 1

    term_recall = {}
    filtered_term_support = {}
    for term_name in pathology_terms:
        support = term_support[term_name]
        if support > 0:
            filtered_term_support[term_name] = support
            term_recall[term_name] = term_hits[term_name] / support

    return {
        "samples": total,
        "unique_predictions": len(counter),
        "unique_prediction_ratio": len(counter) / total,
        "top5_share": sum(count for _, count in counter.most_common(5)) / total,
        "no_acute_predictions": no_acute_predictions,
        "no_acute_rate": no_acute_predictions / total,
        "abnormal_reference_count": len(abnormal_reference_indexes),
        "no_acute_on_abnormal_count": no_acute_on_abnormal_count,
        "no_acute_on_abnormal_rate": (
            no_acute_on_abnormal_count / len(abnormal_reference_indexes)
            if abnormal_reference_indexes
            else 0.0
        ),
        "any_pathology_reference_count": any_pathology_reference_count,
        "any_pathology_prediction_count": any_pathology_prediction_count,
        "any_pathology_recall": (
            any_pathology_recall_hits / any_pathology_reference_count
            if any_pathology_reference_count
            else 0.0
        ),
        "top_predictions": counter.most_common(10),
        "term_support": filtered_term_support,
        "term_recall": term_recall,
    }


def analyze_results_behavior(results, pathology_terms=None):
    summary = {}
    for experiment_name, result in results.items():
        summary[experiment_name] = analyze_prediction_behavior(result, pathology_terms)
    return summary


def plot_behavior_comparison(behavior_summary, metric_name, title, output_path=None):
    labels = list(behavior_summary.keys())
    values = [behavior_summary[label][metric_name] for label in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(metric_name)
    plt.xticks(rotation=15)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.show()
    plt.close()


def plot_pathology_recall_comparison(
    behavior_summary,
    title,
    output_path=None,
    min_support=25,
    max_terms=8,
):
    if not behavior_summary:
        return

    term_support = {}
    for summary in behavior_summary.values():
        for term_name, support in summary["term_support"].items():
            term_support[term_name] = term_support.get(term_name, 0) + support

    selected_terms = [
        term_name
        for term_name, _ in sorted(term_support.items(), key=lambda item: item[1], reverse=True)
        if term_support[term_name] >= min_support
    ][:max_terms]

    if not selected_terms:
        return

    labels = list(behavior_summary.keys())
    x = np.arange(len(selected_terms))
    width = 0.8 / max(len(labels), 1)

    plt.figure(figsize=(12, 6))
    for index, label in enumerate(labels):
        values = [behavior_summary[label]["term_recall"].get(term_name, 0.0) for term_name in selected_terms]
        offset = (index - (len(labels) - 1) / 2) * width
        plt.bar(x + offset, values, width=width, label=label)

    plt.title(title)
    plt.ylabel("recall")
    plt.xticks(x, selected_terms, rotation=15)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.show()
    plt.close()


def plot_metric_comparison(comparison, metric_name, title, output_path=None):
    labels = list(comparison.keys())
    values = [comparison[label][metric_name] for label in labels]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel(metric_name)
    plt.xticks(rotation=15)
    plt.tight_layout()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=160)
    plt.show()
    plt.close()


def analyze_validation_predictions(run_dir):
    validation_dir = run_dir / "validation_by_epoch"
    validation_files = sorted(validation_dir.glob("epoch_*_validation_predictions.json"))
    result = {"validation_files": [str(path) for path in validation_files], "epochs": []}

    for file_path in validation_files:
        predictions = load_json(file_path, default=[])
        texts = [row["prediction"] for row in predictions]
        result["epochs"].append(
            {
                "file": file_path.stem,
                "samples": len(texts),
                "unique_predictions": len(set(texts)),
                "no_acute_predictions": sum(is_no_acute_style(text) for text in texts),
                "top_predictions": Counter(texts).most_common(3),
            }
        )

    history = load_json(run_dir / "fine_tuned_training_history.json", default={})
    result["best_epoch"] = history.get("best_epoch")
    result["best_val_loss"] = history.get("best_val_loss")
    return result


def print_validation_analysis(title, analysis):
    print(title)
    print("-" * 40)
    if not analysis["epochs"]:
        print("No saved validation prediction files were found.")
        return

    for epoch_info in analysis["epochs"]:
        print(epoch_info["file"])
        print(f"samples: {epoch_info['samples']}")
        print(f"unique predictions: {epoch_info['unique_predictions']}")
        print(f"no-acute style predictions: {epoch_info['no_acute_predictions']}")
        print(f"top predictions: {epoch_info['top_predictions']}")
        print("-" * 40)

    print(f"best epoch: {analysis.get('best_epoch')}")
    print(f"best validation loss: {analysis.get('best_val_loss')}")


def build_comparative_analysis(results):
    grouped = {}
    for _, result in results.items():
        grouped.setdefault(result["dataset_name"], {})[result["kind"]] = result

    lines = ["Сравнительный анализ", ""]
    for dataset_name, group in grouped.items():
        zero_shot = group.get("zero_shot")
        fine_tuned = group.get("fine_tuned")
        lines.append(f"Датасет: {dataset_name}")

        if zero_shot is not None and fine_tuned is not None:
            zero_metrics = zero_shot["metrics"]
            fine_metrics = fine_tuned["metrics"]
            lines.append(
                f"BLEU-4: {zero_metrics['BLEU-4']:.4f} -> {fine_metrics['BLEU-4']:.4f} "
                f"({fine_metrics['BLEU-4'] - zero_metrics['BLEU-4']:+.4f})"
            )
            lines.append(
                f"ROUGE-L: {zero_metrics['ROUGE-L']:.4f} -> {fine_metrics['ROUGE-L']:.4f} "
                f"({fine_metrics['ROUGE-L'] - zero_metrics['ROUGE-L']:+.4f})"
            )
            lines.append(
                f"METEOR: {zero_metrics['METEOR']:.4f} -> {fine_metrics['METEOR']:.4f} "
                f"({fine_metrics['METEOR'] - zero_metrics['METEOR']:+.4f})"
            )
        elif fine_tuned is not None:
            metrics = fine_tuned["metrics"]
            lines.append(
                f"Только fine-tuning: BLEU-4={metrics['BLEU-4']:.4f}, "
                f"ROUGE-L={metrics['ROUGE-L']:.4f}, METEOR={metrics['METEOR']:.4f}"
            )
        elif zero_shot is not None:
            metrics = zero_shot["metrics"]
            lines.append(
                f"Только zero-shot: BLEU-4={metrics['BLEU-4']:.4f}, "
                f"ROUGE-L={metrics['ROUGE-L']:.4f}, METEOR={metrics['METEOR']:.4f}"
            )

        lines.append("")

    return "\n".join(lines).strip() + "\n"
