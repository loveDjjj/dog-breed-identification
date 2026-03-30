"""图像预处理与增强策略。"""

from __future__ import annotations

from typing import Any

from torchvision import transforms
from torchvision.transforms import InterpolationMode


def _normalize_size(size_value: Any, default_value: int) -> int:
    """将 transforms 中可能出现的单值或列表尺寸统一转成整数。"""

    if size_value is None:
        return default_value
    if isinstance(size_value, (list, tuple)):
        return int(size_value[0])
    return int(size_value)


def get_preprocess_config(weights: Any, image_size_override: int | None = None) -> dict[str, Any]:
    """从 torchvision 预训练权重中提取标准预处理参数。"""

    default_mean = [0.485, 0.456, 0.406]
    default_std = [0.229, 0.224, 0.225]

    if weights is None:
        crop_size = int(image_size_override or 224)
        resize_size = int(round(crop_size / 0.875))
        return {
            "crop_size": crop_size,
            "resize_size": resize_size,
            "mean": default_mean,
            "std": default_std,
            "interpolation": InterpolationMode.BILINEAR,
        }

    preset = weights.transforms()
    crop_size = _normalize_size(getattr(preset, "crop_size", None), 224)
    resize_size = _normalize_size(
        getattr(preset, "resize_size", None),
        int(round(crop_size / 0.875)),
    )

    if image_size_override is not None:
        crop_size = int(image_size_override)
        resize_size = int(round(crop_size / 0.875))

    return {
        "crop_size": crop_size,
        "resize_size": resize_size,
        "mean": list(getattr(preset, "mean", default_mean)),
        "std": list(getattr(preset, "std", default_std)),
        "interpolation": getattr(preset, "interpolation", InterpolationMode.BILINEAR),
    }


def build_train_transform(augmentation_config: dict[str, Any], preprocess_config: dict[str, Any]) -> transforms.Compose:
    """构建训练阶段的数据增强流水线。"""

    train_config = augmentation_config["train"]
    image_size = preprocess_config["crop_size"]
    interpolation = preprocess_config["interpolation"]

    transform_steps: list[Any] = [
        transforms.RandomResizedCrop(
            size=image_size,
            scale=tuple(train_config["random_resized_crop_scale"]),
            ratio=tuple(train_config["random_resized_crop_ratio"]),
            interpolation=interpolation,
        )
    ]

    if float(train_config["horizontal_flip_prob"]) > 0:
        transform_steps.append(
            transforms.RandomHorizontalFlip(p=float(train_config["horizontal_flip_prob"]))
        )

    if float(train_config["rotation_degrees"]) > 0:
        transform_steps.append(
            transforms.RandomRotation(
                degrees=float(train_config["rotation_degrees"]),
                interpolation=interpolation,
            )
        )

    color_jitter = train_config["color_jitter"]
    if any(float(value) > 0 for value in color_jitter.values()):
        transform_steps.append(
            transforms.ColorJitter(
                brightness=float(color_jitter["brightness"]),
                contrast=float(color_jitter["contrast"]),
                saturation=float(color_jitter["saturation"]),
                hue=float(color_jitter["hue"]),
            )
        )

    if float(train_config["random_grayscale_prob"]) > 0:
        transform_steps.append(
            transforms.RandomGrayscale(p=float(train_config["random_grayscale_prob"]))
        )

    if bool(train_config["use_randaugment"]):
        transform_steps.append(
            transforms.RandAugment(
                num_ops=int(train_config["randaugment_num_ops"]),
                magnitude=int(train_config["randaugment_magnitude"]),
            )
        )

    if bool(train_config["use_trivial_augment"]):
        transform_steps.append(transforms.TrivialAugmentWide(interpolation=interpolation))

    if float(train_config["perspective_prob"]) > 0:
        transform_steps.append(
            transforms.RandomPerspective(
                distortion_scale=float(train_config["perspective_distortion_scale"]),
                p=float(train_config["perspective_prob"]),
                interpolation=interpolation,
            )
        )

    if float(train_config["gaussian_blur_prob"]) > 0:
        blur_kernel = 3 if image_size <= 224 else 5
        transform_steps.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=blur_kernel)],
                p=float(train_config["gaussian_blur_prob"]),
            )
        )

    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocess_config["mean"],
                std=preprocess_config["std"],
            ),
        ]
    )

    if float(train_config["random_erasing_prob"]) > 0:
        transform_steps.append(
            transforms.RandomErasing(
                p=float(train_config["random_erasing_prob"]),
                value="random",
            )
        )

    return transforms.Compose(transform_steps)


def build_eval_transform(preprocess_config: dict[str, Any]) -> transforms.Compose:
    """构建验证集和测试集的确定性预处理流水线。"""

    return transforms.Compose(
        [
            transforms.Resize(
                size=preprocess_config["resize_size"],
                interpolation=preprocess_config["interpolation"],
            ),
            transforms.CenterCrop(preprocess_config["crop_size"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocess_config["mean"],
                std=preprocess_config["std"],
            ),
        ]
    )
