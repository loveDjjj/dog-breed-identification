"""预先生成类别映射、训练/验证划分和测试样本顺序。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dogbreed.config import load_config
from src.dogbreed.metadata import prepare_metadata


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="准备犬种识别项目的元数据文件。")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/resnet50_baseline.yaml",
        help="配置文件路径。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="是否强制重建训练/验证划分。",
    )
    return parser.parse_args()


def main() -> None:
    """执行元数据准备流程。"""

    args = parse_args()
    config = load_config(PROJECT_ROOT / args.config)
    metadata = prepare_metadata(config, PROJECT_ROOT, force_rebuild=args.force)

    print("元数据准备完成。")
    print(json.dumps(metadata["summary"], ensure_ascii=False, indent=2))
    print(f"训练/验证划分文件: {metadata['paths']['split_file']}")
    print(f"测试样本顺序文件: {metadata['paths']['test_samples_file']}")


if __name__ == "__main__":
    main()
