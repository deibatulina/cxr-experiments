import random

from datasets import DatasetDict, load_dataset

from .utils import limit_split


def load_dataset_splits(dataset_spec, config):
    raw = load_dataset(dataset_spec["path"], dataset_spec["config"])

    if "train" in raw and "validation" in raw and "test" in raw:
        splits = DatasetDict(
            {
                "train": raw["train"],
                "validation": raw["validation"],
                "test": raw["test"],
            }
        )
    else:
        if "test" in raw:
            base_split = raw["test"]
        elif "train" in raw:
            base_split = raw["train"]
        else:
            base_split = raw[list(raw.keys())[0]]

        train_test = base_split.train_test_split(test_size=0.3, seed=config["seed"])
        val_test = train_test["test"].train_test_split(test_size=0.5, seed=config["seed"])
        splits = DatasetDict(
            {
                "train": train_test["train"],
                "validation": val_test["train"],
                "test": val_test["test"],
            }
        )

    limits = config["data_limits"]
    splits["train"] = limit_split(splits["train"], limits["train"])
    splits["validation"] = limit_split(splits["validation"], limits["validation"])
    splits["test"] = limit_split(splits["test"], limits["test"])
    return splits


def is_no_acute_style(text):
    text = str(text).strip().lower()
    patterns = [
        "no acute",
        "no active disease",
        "normal chest",
        "negative chest",
        "no cardiopulmonary abnormality",
        "no acute cardiopulmonary",
        "no acute disease",
    ]
    return any(pattern in text for pattern in patterns)


def build_balanced_iu_splits(base_splits, dataset_spec, seed):
    fixed_splits = {}
    target_field = dataset_spec["target_field"]

    for split_name in ["train", "validation", "test"]:
        split = base_splits[split_name]

        if split_name == "test":
            fixed_splits[split_name] = split
            continue

        normal_indices = []
        abnormal_indices = []

        for index, sample in enumerate(split):
            if is_no_acute_style(sample[target_field]):
                normal_indices.append(index)
            else:
                abnormal_indices.append(index)

        rng = random.Random(seed if split_name == "train" else seed + 1)
        rng.shuffle(normal_indices)
        rng.shuffle(abnormal_indices)

        keep_normal = min(len(normal_indices), len(abnormal_indices))
        selected_indices = sorted(abnormal_indices + normal_indices[:keep_normal])
        fixed_splits[split_name] = split.select(selected_indices)

        print(f"{split_name} original size: {len(split)}")
        print(f"{split_name} abnormal samples kept: {len(abnormal_indices)}")
        print(f"{split_name} no-acute samples kept: {keep_normal}")
        print(f"{split_name} fixed size: {len(fixed_splits[split_name])}")
        print("-" * 40)

    return DatasetDict(fixed_splits)

