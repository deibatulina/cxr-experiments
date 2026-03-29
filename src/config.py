from pathlib import Path

import torch


DEFAULT_MODEL_NAME = "Salesforce/blip2-flan-t5-xl"
DEFAULT_PROMPT = (
    "Radiology findings. "
    "Write one or two short clinical sentences about visible findings only. "
    "Do not describe the image itself. "
    "Do not use phrases like 'chest x-ray', 'image of', or 'patient with'."
)

ZERO_SHOT_EXPERIMENT_NAMES = [
    "iu_xray_zero_shot",
    "mimic_cxr_zero_shot",
]

FINE_TUNED_EXPERIMENT_NAMES = [
    "iu_xray_fine_tuned",
    "mimic_cxr_fine_tuned",
]

DEFAULT_RUN_MODES = {
    "iu_xray_zero_shot": "auto",
    "mimic_cxr_zero_shot": "auto",
    "iu_xray_fine_tuned": "auto",
    "mimic_cxr_fine_tuned": "auto",
}


def create_data_limits(zero_shot=None, train=None, validation=None, test=None):
    return {
        "zero_shot": zero_shot,
        "train": train,
        "validation": validation,
        "test": test,
    }


def create_runtime_config(
    model_name=DEFAULT_MODEL_NAME,
    prompt=DEFAULT_PROMPT,
    seed=42,
    results_root_dir=Path("results"),
    num_epochs_by_experiment=None,
    learning_rate=1e-5,
    weight_decay=0.01,
    train_dtype=torch.float32,
    max_label_length=96,
    max_grad_norm=1.0,
    data_limits=None,
):
    if num_epochs_by_experiment is None:
        num_epochs_by_experiment = {
            "iu_xray_fine_tuned": 3,
            "mimic_cxr_fine_tuned": 2,
        }

    if data_limits is None:
        data_limits = create_data_limits()

    return {
        "model_name": model_name,
        "prompt": prompt,
        "seed": seed,
        "results_root_dir": Path(results_root_dir),
        "num_epochs_by_experiment": num_epochs_by_experiment,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "train_dtype": train_dtype,
        "max_label_length": max_label_length,
        "max_grad_norm": max_grad_norm,
        "data_limits": data_limits,
    }


def build_dataset_specs():
    return {
        "iu_xray": {
            "name": "iu_xray",
            "path": "X-iZhang/IU-Xray-RRG",
            "config": "impression_section",
            "image_field": "main_image",
            "target_field": "findings_section",
        },
        "mimic_cxr": {
            "name": "mimic_cxr",
            "path": "itsanmolgupta/mimic-cxr-dataset",
            "config": None,
            "image_field": "image",
            "target_field": "impression",
        },
    }


def build_experiments(build_balanced_iu_splits):
    return {
        "iu_xray_zero_shot": {
            "name": "iu_xray_zero_shot",
            "kind": "zero_shot",
            "dataset_name": "iu_xray",
            "artifact_dir_name": "iu_xray",
            "title": "IU X-Ray zero-shot",
            "split_name": "test",
            "split_builder": None,
        },
        "mimic_cxr_zero_shot": {
            "name": "mimic_cxr_zero_shot",
            "kind": "zero_shot",
            "dataset_name": "mimic_cxr",
            "artifact_dir_name": "mimic_cxr",
            "title": "MIMIC-CXR zero-shot",
            "split_name": "test",
            "split_builder": None,
        },
        "iu_xray_fine_tuned": {
            "name": "iu_xray_fine_tuned",
            "kind": "fine_tuned",
            "dataset_name": "iu_xray",
            "artifact_dir_name": "iu_xray",
            "title": "IU X-Ray fine-tuned",
            "split_name": "test",
            "split_builder": build_balanced_iu_splits,
        },
        "mimic_cxr_fine_tuned": {
            "name": "mimic_cxr_fine_tuned",
            "kind": "fine_tuned",
            "dataset_name": "mimic_cxr",
            "artifact_dir_name": "mimic_cxr",
            "title": "MIMIC-CXR fine-tuned",
            "split_name": "test",
            "split_builder": None,
        },
    }
