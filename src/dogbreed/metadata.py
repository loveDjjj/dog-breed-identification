"""数据集元信息准备与读取逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import build_project_paths, ensure_dir, load_json, save_json


def _load_class_names(labels_df: pd.DataFrame, sample_submission_path: Path) -> list[str]:
    """优先使用 sample_submission 的列顺序，确保提交列顺序与 Kaggle 完全一致。"""

    if sample_submission_path.exists():
        sample_submission = pd.read_csv(sample_submission_path)
        class_names = sample_submission.columns[1:].tolist()
    else:
        class_names = sorted(labels_df["breed"].unique().tolist())

    label_classes = sorted(labels_df["breed"].unique().tolist())
    if set(class_names) != set(label_classes):
        raise ValueError("sample_submission 中的类别集合与 labels.csv 不一致。")

    return class_names


def prepare_metadata(
    config: dict[str, Any],
    project_root: str | Path,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """准备训练/验证划分、类别映射以及测试集样本顺序。"""

    paths = build_project_paths(config, project_root)
    ensure_dir(paths["metadata_root"])

    labels_path = paths["labels_file"]
    sample_submission_path = paths["sample_submission_file"]
    split_path = paths["split_file"]
    split_path.parent.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        raise FileNotFoundError(f"找不到标签文件: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    class_names = _load_class_names(labels_df, sample_submission_path)
    class_to_idx = {class_name: index for index, class_name in enumerate(class_names)}
    idx_to_class = {index: class_name for class_name, index in class_to_idx.items()}

    labels_df["label_idx"] = labels_df["breed"].map(class_to_idx)
    labels_df["image_relpath"] = labels_df["id"].apply(lambda sample_id: f"train/{sample_id}.jpg")

    if split_path.exists() and not force_rebuild:
        split_df = pd.read_csv(split_path)
    else:
        train_df, val_df = train_test_split(
            labels_df[["id", "breed", "label_idx", "image_relpath"]],
            test_size=float(config["data"]["val_ratio"]),
            random_state=int(config["experiment"]["seed"]),
            shuffle=True,
            stratify=labels_df["breed"],
        )

        train_df = train_df.copy()
        val_df = val_df.copy()
        train_df["split"] = "train"
        val_df["split"] = "val"
        split_df = pd.concat([train_df, val_df], ignore_index=True)
        split_df.to_csv(split_path, index=False, encoding="utf-8-sig")

    if sample_submission_path.exists():
        sample_submission = pd.read_csv(sample_submission_path)
        test_ids = sample_submission["id"].tolist()
    else:
        test_ids = sorted(path.stem for path in paths["test_dir"].glob("*.jpg"))

    test_df = pd.DataFrame(
        {
            "id": test_ids,
            "image_relpath": [f"test/{sample_id}.jpg" for sample_id in test_ids],
        }
    )
    test_df.to_csv(paths["test_samples_file"], index=False, encoding="utf-8-sig")

    summary = {
        "num_train_samples": int(len(labels_df)),
        "num_classes": int(len(class_names)),
        "num_train_split": int((split_df["split"] == "train").sum()),
        "num_val_split": int((split_df["split"] == "val").sum()),
        "num_test_samples": int(len(test_df)),
        "class_names_preview": class_names[:10],
    }

    save_json(class_names, paths["class_names_file"])
    save_json(class_to_idx, paths["class_to_idx_file"])
    save_json(idx_to_class, paths["idx_to_class_file"])
    save_json(summary, paths["dataset_summary_file"])

    return {
        "paths": paths,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "split_df": split_df,
        "test_df": test_df,
        "summary": summary,
    }


def load_metadata(
    config: dict[str, Any],
    project_root: str | Path,
    auto_prepare: bool = True,
) -> dict[str, Any]:
    """读取已经存在的元数据；若缺失则按需自动生成。"""

    paths = build_project_paths(config, project_root)
    required_files = [
        paths["split_file"],
        paths["class_names_file"],
        paths["class_to_idx_file"],
        paths["idx_to_class_file"],
        paths["test_samples_file"],
        paths["dataset_summary_file"],
    ]

    if auto_prepare and not all(path.exists() for path in required_files):
        return prepare_metadata(config, project_root, force_rebuild=False)

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "以下元数据文件缺失，请先运行 scripts/prepare_metadata.py:\n"
            + "\n".join(missing)
        )

    split_df = pd.read_csv(paths["split_file"])
    test_df = pd.read_csv(paths["test_samples_file"])
    class_names = load_json(paths["class_names_file"])
    class_to_idx = load_json(paths["class_to_idx_file"])
    idx_to_class_raw = load_json(paths["idx_to_class_file"])
    idx_to_class = {int(key): value for key, value in idx_to_class_raw.items()}
    summary = load_json(paths["dataset_summary_file"])

    return {
        "paths": paths,
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "split_df": split_df,
        "test_df": test_df,
        "summary": summary,
    }
