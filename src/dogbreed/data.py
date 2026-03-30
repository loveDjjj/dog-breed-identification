"""数据集定义与 DataLoader 构建逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset


# 某些图片文件如果头部略有问题，允许 PIL 尝试继续读取，增强工程鲁棒性。
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DogBreedClassificationDataset(Dataset):
    """训练/验证集数据集，返回图像张量和整数标签。"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_root: str | Path,
        transform: Any = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self.dataframe.iloc[index]
        image_path = self.data_root / row["image_relpath"]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = int(row["label_idx"])
        return image, target


class DogBreedTestDataset(Dataset):
    """测试集数据集，返回图像张量和样本 id。"""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        data_root: str | Path,
        transform: Any = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.data_root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[Any, str]:
        row = self.dataframe.iloc[index]
        image_path = self.data_root / row["image_relpath"]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, str(row["id"])


def _build_loader_kwargs(num_workers: int, pin_memory: bool, persistent_workers: bool) -> dict[str, Any]:
    """统一整理 DataLoader 的线程参数。"""

    return {
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(persistent_workers and int(num_workers) > 0),
    }


def build_train_val_dataloaders(
    config: dict[str, Any],
    metadata: dict[str, Any],
    train_transform: Any,
    eval_transform: Any,
) -> tuple[DataLoader, DataLoader]:
    """构建训练集与验证集 DataLoader。"""

    split_df = metadata["split_df"]
    train_df = split_df[split_df["split"] == "train"].reset_index(drop=True)
    val_df = split_df[split_df["split"] == "val"].reset_index(drop=True)

    data_root = metadata["paths"]["data_root"]
    data_config = config["data"]

    train_dataset = DogBreedClassificationDataset(
        dataframe=train_df,
        data_root=data_root,
        transform=train_transform,
    )
    val_dataset = DogBreedClassificationDataset(
        dataframe=val_df,
        data_root=data_root,
        transform=eval_transform,
    )

    loader_kwargs = _build_loader_kwargs(
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]),
        persistent_workers=bool(data_config["persistent_workers"]),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader


def build_test_dataloader(
    config: dict[str, Any],
    metadata: dict[str, Any],
    eval_transform: Any,
) -> DataLoader:
    """构建测试集 DataLoader。"""

    inference_config = config["inference"]
    data_config = config["data"]

    test_dataset = DogBreedTestDataset(
        dataframe=metadata["test_df"],
        data_root=metadata["paths"]["data_root"],
        transform=eval_transform,
    )

    loader_kwargs = _build_loader_kwargs(
        num_workers=int(inference_config.get("num_workers", data_config["num_workers"])),
        pin_memory=bool(data_config["pin_memory"]),
        persistent_workers=bool(data_config["persistent_workers"]),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(inference_config["batch_size"]),
        shuffle=False,
        **loader_kwargs,
    )
    return test_loader
