"""项目通用工具函数。"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import torch


# 在无图形界面的服务器环境中，强制使用 Agg 后端生成图片。
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def seed_everything(seed: int) -> None:
    """固定所有常见随机源，尽量保证可复现。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 这里关闭 cudnn 的随机优化，换取更稳定的可复现性。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_path(base_dir: str | Path, relative_or_absolute: str | Path | None) -> Path | None:
    """将相对路径解析为绝对路径；若本身已是绝对路径则直接返回。"""

    if relative_or_absolute is None:
        return None

    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return Path(base_dir) / path


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，并返回 Path 对象。"""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_project_paths(config: dict[str, Any], project_root: str | Path) -> dict[str, Path]:
    """根据配置统一生成工程运行过程中要用到的路径。"""

    project_root = Path(project_root).resolve()
    output_root = resolve_path(project_root, config["output"]["root_dir"])
    metadata_root = resolve_path(project_root, config["output"]["metadata_dir"])
    data_root = resolve_path(project_root, config["data"]["data_dir"])
    experiment_name = config["experiment"]["name"]

    split_file = config["data"].get("split_file")
    if split_file:
        split_path = resolve_path(project_root, split_file)
    else:
        seed = config["experiment"]["seed"]
        val_ratio = int(float(config["data"]["val_ratio"]) * 100)
        split_path = metadata_root / f"train_val_split_seed{seed}_val{val_ratio}.csv"

    return {
        "project_root": project_root,
        "output_root": output_root,
        "metadata_root": metadata_root,
        "data_root": data_root,
        "train_dir": data_root / config["data"]["train_dir"],
        "test_dir": data_root / config["data"]["test_dir"],
        "labels_file": data_root / config["data"]["labels_file"],
        "sample_submission_file": data_root / config["data"]["sample_submission_file"],
        "split_file": split_path,
        "class_names_file": metadata_root / "class_names.json",
        "class_to_idx_file": metadata_root / "class_to_idx.json",
        "idx_to_class_file": metadata_root / "idx_to_class.json",
        "dataset_summary_file": metadata_root / "dataset_summary.json",
        "test_samples_file": metadata_root / "test_samples.csv",
        "checkpoint_dir": output_root / "checkpoints" / experiment_name,
        "log_dir": output_root / "logs" / experiment_name,
        "figure_dir": output_root / "figures" / experiment_name,
        "submission_dir": output_root / "submissions" / experiment_name,
    }


def to_serializable(data: Any) -> Any:
    """将常见对象转换为 JSON 可序列化格式。"""

    if isinstance(data, Path):
        return str(data)
    if isinstance(data, dict):
        return {key: to_serializable(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_serializable(item) for item in data]
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, (np.integer, np.int64)):
        return int(data)
    if isinstance(data, (np.floating, np.float64)):
        return float(data)
    return data


def save_json(data: Any, output_path: str | Path) -> None:
    """保存 JSON 文件。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            to_serializable(data),
            file,
            ensure_ascii=False,
            indent=2,
        )


def load_json(input_path: str | Path) -> Any:
    """读取 JSON 文件。"""

    with Path(input_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def setup_logger(log_file: str | Path) -> logging.Logger:
    """同时输出到终端和日志文件，方便训练时留档。"""

    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger_name = f"dogbreed.{log_file.stem}.{log_file.parent.name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def select_device(device_name: str) -> torch.device:
    """根据配置自动选择训练设备。"""

    normalized = str(device_name).strip().lower()

    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "当前配置请求使用 CUDA，但当前环境中的 PyTorch 不支持 CUDA。\n"
                f"torch.__version__ = {torch.__version__}\n"
                f"torch.version.cuda = {torch.version.cuda}\n"
                "请安装带 CUDA 的 torch / torchvision 组合，或者把配置中的 "
                "`training.device` 改为 `auto` 或 `cpu`。"
            )
        return torch.device(device_name)

    return torch.device(device_name)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """统计模型总参数量与可训练参数量。"""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable}


def save_history_csv(history: list[dict[str, Any]], output_path: str | Path) -> None:
    """将每个 epoch 的指标保存为 CSV，便于后续写报告和画图。"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_path, index=False, encoding="utf-8-sig")


def save_history_plot(history: list[dict[str, Any]], output_path: str | Path) -> None:
    """将训练/验证损失和准确率曲线保存成图片。"""

    if not history:
        return

    df = pd.DataFrame(history)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["val_acc"], label="val_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def stabilize_probabilities(probabilities: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """对概率做裁剪并重新归一化，避免提交时出现严格 0 或 1。"""

    clipped = np.clip(probabilities, eps, 1.0 - eps)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return clipped
