"""训练犬种分类模型，并保存最佳权重与训练日志。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dogbreed.config import load_config, save_config_snapshot
from src.dogbreed.data import build_train_val_dataloaders
from src.dogbreed.engine import (
    build_criterion,
    build_optimizer,
    build_scheduler,
    evaluate,
    train_one_epoch,
)
from src.dogbreed.metadata import prepare_metadata
from src.dogbreed.models import build_model, freeze_backbone
from src.dogbreed.transforms import (
    build_eval_transform,
    build_train_transform,
    get_preprocess_config,
    serialize_preprocess_config,
)
from src.dogbreed.utils import (
    build_project_paths,
    count_parameters,
    ensure_dir,
    save_history_csv,
    save_history_plot,
    save_json,
    seed_everything,
    select_device,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    """解析训练脚本参数。"""

    parser = argparse.ArgumentParser(description="训练犬种识别模型。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet50_baseline.yaml",
        help="配置文件路径。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备，例如 cuda、cuda:0 或 cpu；默认读取配置。",
    )
    parser.add_argument(
        "--rebuild-metadata",
        action="store_true",
        help="是否强制重建训练/验证划分。",
    )
    return parser.parse_args()


def is_better(metric_value: float, best_value: float, monitor: str) -> bool:
    """根据监控指标判断当前结果是否优于历史最佳。"""

    if "loss" in monitor.lower():
        return metric_value < best_value
    return metric_value > best_value


def initial_best_value(monitor: str) -> float:
    """为不同监控指标设置初始最优值。"""

    if "loss" in monitor.lower():
        return math.inf
    return -math.inf


def build_checkpoint(
    model: torch.nn.Module,
    config: dict[str, Any],
    metadata: dict[str, Any],
    preprocess_config: dict[str, Any],
    epoch: int,
    best_metric: float,
) -> dict[str, Any]:
    """整理需要持久化到磁盘的 checkpoint 内容。"""

    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_name": config["model"]["name"],
        "weights_name": config["model"]["weights"],
        "class_names": metadata["class_names"],
        "preprocess_config": serialize_preprocess_config(preprocess_config),
        "best_metric": float(best_metric),
        "config": {key: value for key, value in config.items() if key != "_meta"},
    }


def main() -> None:
    """执行训练主流程。"""

    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)

    if args.device is not None:
        config["training"]["device"] = args.device

    deterministic = bool(config["training"].get("deterministic", True))
    seed_everything(
        int(config["experiment"]["seed"]),
        deterministic=deterministic,
    )

    metadata = prepare_metadata(
        config=config,
        project_root=PROJECT_ROOT,
        force_rebuild=args.rebuild_metadata,
    )
    paths = build_project_paths(config, PROJECT_ROOT)
    ensure_dir(paths["checkpoint_dir"])
    ensure_dir(paths["log_dir"])
    ensure_dir(paths["figure_dir"])

    logger = setup_logger(paths["log_dir"] / "train.log")
    save_config_snapshot(config, paths["log_dir"] / "resolved_config.yaml")

    device = select_device(config["training"]["device"])
    matmul_precision = str(config["training"].get("matmul_precision", "high")).lower()
    if hasattr(torch, "set_float32_matmul_precision"):
        if matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError(
                "training.matmul_precision 必须是 highest / high / medium 之一。"
            )
        torch.set_float32_matmul_precision(matmul_precision)

    cudnn_benchmark = bool(config["training"].get("cudnn_benchmark", False))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if cudnn_benchmark:
            torch.backends.cudnn.deterministic = False

    logger.info("实验名称: %s", config["experiment"]["name"])
    logger.info("训练设备: %s", device)
    logger.info(
        "deterministic=%s | cudnn_benchmark=%s | matmul_precision=%s",
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        matmul_precision,
    )
    logger.info("数据集摘要: %s", json.dumps(metadata["summary"], ensure_ascii=False))

    model, weights = build_model(
        model_name=config["model"]["name"],
        num_classes=len(metadata["class_names"]),
        weights_name=config["model"]["weights"],
    )
    preprocess_config = get_preprocess_config(
        weights=weights,
        image_size_override=config["data"].get("image_size"),
    )
    train_transform = build_train_transform(config["augmentation"], preprocess_config)
    eval_transform = build_eval_transform(preprocess_config)
    train_loader, val_loader = build_train_val_dataloaders(
        config=config,
        metadata=metadata,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    model.to(device)

    freeze_epochs = int(config["training"]["freeze_backbone_epochs"])
    backbone_frozen = freeze_epochs > 0
    if backbone_frozen:
        freeze_backbone(model, freeze=True)
        logger.info("前 %d 个 epoch 冻结骨干网络，仅训练分类头。", freeze_epochs)

    parameter_stats = count_parameters(model)
    logger.info(
        "模型: %s | 总参数量: %s | 当前可训练参数量: %s",
        config["model"]["name"],
        parameter_stats["total"],
        parameter_stats["trainable"],
    )

    criterion = build_criterion(config["training"]["criterion"])
    optimizer = build_optimizer(model.parameters(), config["training"]["optimizer"])
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_config=config["training"]["scheduler"],
        num_epochs=int(config["training"]["epochs"]),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config["training"]["amp"]) and device.type == "cuda"
    )

    history: list[dict[str, Any]] = []
    monitor = config["training"]["monitor"]
    best_metric = initial_best_value(monitor)
    best_epoch = 0
    early_stop_counter = 0

    best_checkpoint_path = paths["checkpoint_dir"] / "best.pth"
    last_checkpoint_path = paths["checkpoint_dir"] / "last.pth"
    history_csv_path = paths["log_dir"] / "history.csv"
    history_plot_path = paths["figure_dir"] / "training_curves.png"

    num_epochs = int(config["training"]["epochs"])
    for epoch in range(1, num_epochs + 1):
        if backbone_frozen and epoch > freeze_epochs:
            freeze_backbone(model, freeze=False)
            backbone_frozen = False
            logger.info("从第 %d 个 epoch 开始解冻全部参数进行微调。", epoch)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_amp=bool(config["training"]["amp"]),
            grad_clip_norm=float(config["training"]["grad_clip_norm"]),
            epoch_index=epoch,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=bool(config["training"]["amp"]),
            epoch_index=epoch,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(epoch_record)
        save_history_csv(history, history_csv_path)
        save_history_plot(history, history_plot_path)

        logger.info(
            "Epoch %d/%d | lr=%.8f | train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch,
            num_epochs,
            current_lr,
            train_metrics["loss"],
            train_metrics["acc"],
            val_metrics["loss"],
            val_metrics["acc"],
        )

        metric_value = epoch_record[monitor]
        if is_better(metric_value, best_metric, monitor):
            best_metric = metric_value
            best_epoch = epoch
            early_stop_counter = 0
            torch.save(
                build_checkpoint(
                    model=model,
                    config=config,
                    metadata=metadata,
                    preprocess_config=preprocess_config,
                    epoch=epoch,
                    best_metric=best_metric,
                ),
                best_checkpoint_path,
            )
            logger.info("已更新最佳模型: %s = %.6f", monitor, best_metric)
        else:
            early_stop_counter += 1

        torch.save(
            build_checkpoint(
                model=model,
                config=config,
                metadata=metadata,
                preprocess_config=preprocess_config,
                epoch=epoch,
                best_metric=best_metric,
            ),
            last_checkpoint_path,
        )

        if scheduler is not None:
            if config["training"]["scheduler"]["name"].lower() == "plateau":
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        patience = int(config["training"]["early_stopping_patience"])
        if early_stop_counter >= patience:
            logger.info("触发早停，连续 %d 个 epoch 未提升。", patience)
            break

    summary = {
        "experiment_name": config["experiment"]["name"],
        "best_epoch": best_epoch,
        "best_metric_name": monitor,
        "best_metric_value": float(best_metric),
        "checkpoint_path": str(best_checkpoint_path),
        "history_csv": str(history_csv_path),
        "training_curve": str(history_plot_path),
    }
    save_json(summary, paths["log_dir"] / "summary.json")
    logger.info("训练完成，摘要已保存到: %s", paths["log_dir"] / "summary.json")


if __name__ == "__main__":
    main()
