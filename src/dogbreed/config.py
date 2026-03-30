"""配置文件读取与保存工具。"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "experiment": {
        "name": "dogbreed_experiment",
        "seed": 42,
    },
    "output": {
        "root_dir": "outputs",
        "metadata_dir": "metadata",
    },
    "data": {
        "data_dir": ".",
        "train_dir": "train",
        "test_dir": "test",
        "labels_file": "labels.csv",
        "sample_submission_file": "sample_submission.csv",
        "split_file": None,
        "val_ratio": 0.2,
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "image_size": 224,
    },
    "augmentation": {
        "train": {
            "random_resized_crop_scale": [0.8, 1.0],
            "random_resized_crop_ratio": [0.75, 1.3333333333],
            "horizontal_flip_prob": 0.5,
            "rotation_degrees": 10,
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.03,
            },
            "random_grayscale_prob": 0.0,
            "perspective_prob": 0.0,
            "perspective_distortion_scale": 0.2,
            "gaussian_blur_prob": 0.0,
            "use_randaugment": False,
            "randaugment_num_ops": 2,
            "randaugment_magnitude": 9,
            "use_trivial_augment": False,
            "random_erasing_prob": 0.0,
        }
    },
    "model": {
        "name": "resnet50",
        "weights": "DEFAULT",
    },
    "training": {
        "device": "auto",
        "epochs": 10,
        "batch_size": 32,
        "freeze_backbone_epochs": 0,
        "amp": True,
        "grad_clip_norm": 1.0,
        "early_stopping_patience": 5,
        "monitor": "val_loss",
        "criterion": {
            "name": "cross_entropy",
            "label_smoothing": 0.0,
        },
        "optimizer": {
            "name": "adamw",
            "lr": 3e-4,
            "weight_decay": 1e-4,
        },
        "scheduler": {
            "name": "cosine",
            "min_lr": 1e-6,
            "step_size": 5,
            "gamma": 0.1,
            "patience": 2,
            "factor": 0.5,
        },
    },
    "inference": {
        "batch_size": 64,
        "num_workers": 4,
        "probability_clip": 1e-8,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并配置字典，使用户配置只覆盖自己显式指定的字段。"""

    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置，并补齐默认字段。"""

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file) or {}

    config = _deep_merge(DEFAULT_CONFIG, user_config)
    config["_meta"] = {
        "config_path": str(config_path.resolve()),
    }
    return config


def save_config_snapshot(config: dict[str, Any], output_path: str | Path) -> None:
    """将当前生效配置保存到输出目录，方便复现实验。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_config = {
        key: value for key, value in config.items() if key != "_meta"
    }

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            serializable_config,
            file,
            allow_unicode=True,
            sort_keys=False,
        )
