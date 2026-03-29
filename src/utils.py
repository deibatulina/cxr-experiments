import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def sanitize_model_dir_name(model_name):
    return model_name.replace("/", "__").replace("-", "_")


def limit_split(dataset, limit):
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def clear_memory(device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    if device == "mps":
        torch.mps.empty_cache()


def to_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_metrics(title, metrics):
    print(title)
    print("-" * 30)
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            print(f"{key:<24} {float(value):.4f}")
        else:
            print(f"{key:<24} {value}")


def print_platform_summary(platform):
    print("Конфигурация запуска")
    print("-" * 30)
    for key, value in platform.items():
        print(f"{key:<20} {value}")


def prepare_run_dir_for_rerun(run_dir):
    run_dir = Path(run_dir)
    for target in [
        run_dir / "best_model",
        run_dir / "last_model",
        run_dir / "tensorboard",
        run_dir / "validation_by_epoch",
    ]:
        if target.exists():
            shutil.rmtree(target)

    for pattern in ["fine_tuned_test_*", "fine_tuned_training_*", "best_epoch_summary.json"]:
        for file_path in run_dir.glob(pattern):
            if file_path.is_file():
                file_path.unlink()

