"""加载最佳模型，对测试集输出概率并生成 Kaggle submission.csv。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dogbreed.config import load_config
from src.dogbreed.data import build_test_dataloader
from src.dogbreed.engine import predict_probabilities
from src.dogbreed.metadata import load_metadata
from src.dogbreed.models import build_model
from src.dogbreed.transforms import build_eval_transform, get_preprocess_config
from src.dogbreed.utils import (
    build_project_paths,
    ensure_dir,
    select_device,
    setup_logger,
    stabilize_probabilities,
)


def parse_args() -> argparse.Namespace:
    """解析推理脚本参数。"""

    parser = argparse.ArgumentParser(description="生成 Kaggle 提交文件。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet50_baseline.yaml",
        help="配置文件路径。",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint 路径；若为空则默认读取当前实验的 best.pth。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出 submission.csv 的路径；若为空则保存到默认输出目录。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备，例如 cuda、cuda:0 或 cpu；默认读取配置。",
    )
    return parser.parse_args()


def main() -> None:
    """执行测试集推理与 CSV 生成。"""

    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    if args.device is not None:
        config["training"]["device"] = args.device

    metadata = load_metadata(config, PROJECT_ROOT, auto_prepare=True)
    paths = build_project_paths(config, PROJECT_ROOT)
    ensure_dir(paths["submission_dir"])
    ensure_dir(paths["log_dir"])

    logger = setup_logger(paths["log_dir"] / "predict.log")

    default_checkpoint = paths["checkpoint_dir"] / "best.pth"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint
    if not checkpoint_path.is_absolute():
        checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"找不到 checkpoint 文件: {checkpoint_path}")

    device = select_device(config["training"]["device"])
    logger.info("推理设备: %s", device)
    logger.info("加载 checkpoint: %s", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_name = checkpoint.get("model_name", config["model"]["name"])
    class_names = checkpoint.get("class_names", metadata["class_names"])
    preprocess_config = checkpoint.get("preprocess_config")

    model, weights = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        weights_name=None,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if preprocess_config is None:
        preprocess_config = get_preprocess_config(
            weights=weights,
            image_size_override=config["data"].get("image_size"),
        )

    eval_transform = build_eval_transform(preprocess_config)
    test_loader = build_test_dataloader(
        config=config,
        metadata=metadata,
        eval_transform=eval_transform,
    )
    probabilities, sample_ids = predict_probabilities(
        model=model,
        dataloader=test_loader,
        device=device,
        use_amp=bool(config["training"]["amp"]),
    )

    probabilities = stabilize_probabilities(
        probabilities,
        eps=float(config["inference"]["probability_clip"]),
    )

    submission = pd.DataFrame(probabilities, columns=class_names)
    submission.insert(0, "id", sample_ids)

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (PROJECT_ROOT / output_path).resolve()
    else:
        output_path = paths["submission_dir"] / "submission.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, float_format="%.8f", encoding="utf-8-sig")
    logger.info("submission.csv 已生成: %s", output_path)


if __name__ == "__main__":
    main()
