"""基于 torchvision 的预训练模型构建与分类头替换逻辑。"""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from torchvision.models import get_model, get_model_weights, get_weight


def resolve_torchvision_weights(model_name: str, weights_name: str | None) -> Any:
    """将配置中的权重名称解析为 torchvision 的权重枚举。"""

    if weights_name is None:
        return None

    normalized = str(weights_name).strip()
    if normalized.lower() in {"", "none", "null"}:
        return None

    if "." in normalized:
        return get_weight(normalized)

    weights_enum = get_model_weights(model_name)
    if normalized.upper() == "DEFAULT":
        return weights_enum.DEFAULT
    return getattr(weights_enum, normalized)


def _replace_sequential_last_linear(module: nn.Sequential, num_classes: int) -> nn.Sequential:
    """替换 Sequential 中最后一个全连接层。"""

    layers = list(module.children())
    for index in range(len(layers) - 1, -1, -1):
        if isinstance(layers[index], nn.Linear):
            in_features = layers[index].in_features
            layers[index] = nn.Linear(in_features, num_classes)
            return nn.Sequential(*layers)
    raise NotImplementedError("未能在 Sequential 分类头中找到 Linear 层。")


def reset_classifier(model: nn.Module, num_classes: int) -> nn.Module:
    """根据不同网络结构替换最终分类层。"""

    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Linear(classifier.in_features, num_classes)
            return model
        if isinstance(classifier, nn.Sequential):
            model.classifier = _replace_sequential_last_linear(classifier, num_classes)
            return model

    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
        return model

    if hasattr(model, "head"):
        head = model.head
        if isinstance(head, nn.Linear):
            model.head = nn.Linear(head.in_features, num_classes)
            return model
        if isinstance(head, nn.Sequential):
            model.head = _replace_sequential_last_linear(head, num_classes)
            return model

    raise NotImplementedError("当前模型结构未适配分类头替换，请在 models.py 中补充分支。")


def build_model(model_name: str, num_classes: int, weights_name: str | None = "DEFAULT") -> tuple[nn.Module, Any]:
    """构建带预训练权重的分类模型，并替换为目标任务的类别数。"""

    weights = resolve_torchvision_weights(model_name, weights_name)
    model = get_model(model_name, weights=weights)
    model = reset_classifier(model, num_classes)
    return model, weights


def _get_classifier_module(model: nn.Module) -> nn.Module:
    """定位模型当前的分类头模块，用于冻结/解冻骨干网络。"""

    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    if hasattr(model, "heads"):
        return model.heads
    if hasattr(model, "head"):
        return model.head
    raise NotImplementedError("无法识别模型的分类头模块。")


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """冻结或解冻骨干网络，但始终保持分类头可训练。"""

    for parameter in model.parameters():
        parameter.requires_grad = not freeze

    classifier_module = _get_classifier_module(model)
    for parameter in classifier_module.parameters():
        parameter.requires_grad = True
