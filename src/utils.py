from __future__ import annotations

import gc
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def sanitize_model_dir_name(model_name: str) -> str:
    """Преобразует идентификатор модели в безопасное имя директории."""

    return model_name.replace("/", "__").replace("-", "_")


def limit_split(dataset: Any, limit: int | None) -> Any:
    """Ограничивает сплит указанным числом объектов."""

    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def clear_memory(device: str) -> None:
    """Освобождает память Python и кэши ускорителя после тяжёлых операций."""

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    if device == "mps":
        torch.mps.empty_cache()


def to_rgb(image: Any) -> Image.Image:
    """Приводит входное изображение к RGB-формату PIL."""

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def save_json(path: str | Path, data: Any) -> None:
    """Сохраняет JSON-файл в UTF-8 с читаемым форматированием."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def load_json(path: str | Path, default: Any = None) -> Any:
    """Загружает JSON, а при отсутствии файла возвращает значение по умолчанию."""

    path = Path(path)
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Добавляет одну JSON-запись как строку в JSONL-файл."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_metrics(title: str, metrics: dict[str, Any]) -> None:
    """Печатает метрики в простом выровненном текстовом виде."""

    print(title)
    print("-" * 30)
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating)):
            print(f"{key:<24} {float(value):.4f}")
        else:
            print(f"{key:<24} {value}")


def print_platform_summary(platform: dict[str, Any]) -> None:
    """Печатает выбранное устройство и параметры runtime."""

    print("Конфигурация запуска")
    print("-" * 30)
    for key, value in platform.items():
        print(f"{key:<20} {value}")


def print_torchinfo_summary(model: torch.nn.Module, title: str = "Сводка модели") -> None:
    """Печатает краткую сводку модели через torchinfo, если пакет доступен."""

    try:
        from torchinfo import summary
    except ImportError:
        print("Сводка torchinfo пропущена: пакет torchinfo не установлен.")
        return

    print(title)
    print("-" * 30)
    try:
        model_summary = summary(
            model,
            depth=2,
            col_names=("num_params", "trainable"),
            row_settings=("var_names",),
            verbose=0,
        )
    except Exception as error:
        print(f"Сводка torchinfo пропущена: {error}")
        return

    print(model_summary)


def prepare_run_dir_for_rerun(run_dir: str | Path) -> None:
    """Удаляет старые артефакты fine-tuning перед принудительным перезапуском."""

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
