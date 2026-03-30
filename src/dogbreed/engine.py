"""训练、验证、推理过程中的核心循环。"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from tqdm.auto import tqdm


def build_criterion(criterion_config: dict[str, Any]) -> nn.Module:
    """根据配置构建损失函数。"""

    name = criterion_config["name"].lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            label_smoothing=float(criterion_config.get("label_smoothing", 0.0))
        )
    raise ValueError(f"暂不支持的损失函数: {criterion_config['name']}")


def build_optimizer(parameters, optimizer_config: dict[str, Any]) -> torch.optim.Optimizer:
    """根据配置构建优化器。"""

    name = optimizer_config["name"].lower()
    if name == "adamw":
        return AdamW(
            parameters,
            lr=float(optimizer_config["lr"]),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        )
    if name == "sgd":
        return SGD(
            parameters,
            lr=float(optimizer_config["lr"]),
            momentum=float(optimizer_config.get("momentum", 0.9)),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
            nesterov=bool(optimizer_config.get("nesterov", True)),
        )
    raise ValueError(f"暂不支持的优化器: {optimizer_config['name']}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict[str, Any],
    num_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | ReduceLROnPlateau | None:
    """根据配置构建学习率调度器。"""

    name = scheduler_config["name"].lower()
    if name in {"none", "null"}:
        return None
    if name == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=int(num_epochs),
            eta_min=float(scheduler_config.get("min_lr", 1e-6)),
        )
    if name == "step":
        return StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size", 5)),
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    if name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=int(scheduler_config.get("patience", 2)),
            factor=float(scheduler_config.get("factor", 0.5)),
        )
    raise ValueError(f"暂不支持的学习率调度器: {scheduler_config['name']}")


def _get_autocast_context(device: torch.device, use_amp: bool):
    """仅在 CUDA 环境下启用 AMP，其余场景返回空上下文。"""

    if device.type == "cuda" and use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def _accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """计算当前 batch 的 top-1 准确率。"""

    predictions = logits.argmax(dim=1)
    return float((predictions == targets).float().mean().item())


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None,
    use_amp: bool,
    grad_clip_norm: float | None,
    epoch_index: int,
) -> dict[str, float]:
    """执行一个完整训练 epoch。"""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"训练 Epoch {epoch_index}", leave=False)
    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _get_autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is not None and device.type == "cuda" and use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{_accuracy_from_logits(logits, targets):.4f}",
        )

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    epoch_index: int,
) -> dict[str, float]:
    """执行一个完整验证 epoch。"""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc=f"验证 Epoch {epoch_index}", leave=False)
    for images, targets in progress_bar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with _get_autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_samples += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{_accuracy_from_logits(logits, targets):.4f}",
        )

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def predict_probabilities(
    model: nn.Module,
    dataloader,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, list[str]]:
    """对测试集进行批量推理，并输出每张图像的类别概率分布。"""

    model.eval()
    probability_list: list[np.ndarray] = []
    sample_ids: list[str] = []

    progress_bar = tqdm(dataloader, desc="测试集推理", leave=False)
    for images, batch_ids in progress_bar:
        images = images.to(device, non_blocking=True)

        with _get_autocast_context(device, use_amp):
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)

        probability_list.append(probabilities.detach().cpu().numpy())
        sample_ids.extend(list(batch_ids))

    stacked_probabilities = np.concatenate(probability_list, axis=0)
    return stacked_probabilities, sample_ids
