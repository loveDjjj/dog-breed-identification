# 犬种图像分类技术报告

## 1. 任务简介

本项目使用 Kaggle `Dog Breed Identification` 数据集完成 120 类犬种图像分类任务。数据集包含：

- 训练集：10222 张
- 测试集：10357 张
- 类别数：120

竞赛官方评估指标为 `multi-class log loss`，因此测试阶段必须输出每张测试图像对全部 120 个类别的概率分布。

## 2. 数据读取与标签映射

### 2.1 数据文件

- 训练标签文件：`labels.csv`
- 测试提交模板：`sample_submission.csv`
- 训练图片目录：`train/`
- 测试图片目录：`test/`

### 2.2 标签映射策略

本项目优先按照 `sample_submission.csv` 的列顺序构建类别列表，以确保：

- 训练阶段使用的类别索引与测试提交列顺序一致。
- 生成的 `submission.csv` 可以直接上传 Kaggle。

### 2.3 训练/验证划分

采用 `train_test_split + stratify` 按类别分层抽样，将训练数据划分为：

- 训练集：80%
- 验证集：20%

随机种子固定为：`42`

## 3. 数据预处理与增强

### 3.1 验证集 / 测试集预处理

验证集与测试集使用和预训练权重匹配的标准预处理：

- Resize
- CenterCrop
- ToTensor
- Normalize

### 3.2 训练集数据增强

本项目实现并支持如下增强方法：

- RandomResizedCrop
- RandomHorizontalFlip
- RandomRotation
- ColorJitter
- RandomGrayscale
- RandomPerspective
- GaussianBlur
- RandAugment
- TrivialAugmentWide
- RandomErasing

### 3.3 本次实验采用的增强配置

请在完成训练后填写下表：

| 实验名 | 主要增强策略 |
| --- | --- |
| ResNet50 基线 | 待填写 |
| EfficientNet-B0 增强版 | 待填写 |
| ViT-B/16 可选实验 | 待填写 |

## 4. 模型与迁移学习方案

### 4.1 基线模型

- 模型名称：`ResNet50`
- 预训练权重：`torchvision DEFAULT`
- 迁移学习方式：替换最终分类头为 120 类输出

### 4.2 对比模型

- 模型名称：`EfficientNet-B0`
- 预训练权重：`torchvision DEFAULT`
- 迁移学习方式：替换最终分类头为 120 类输出

### 4.3 可选扩展模型

- 模型名称：`ViT-B/16`
- 预训练权重：`torchvision DEFAULT`
- 迁移学习方式：替换最终分类头为 120 类输出

## 5. 损失函数与训练设置

### 5.1 损失函数

- 损失函数：交叉熵损失 `CrossEntropyLoss`
- 标签平滑：建议记录是否使用 `label smoothing`

### 5.2 优化器与学习率调度

请在跑完实验后填写：

| 实验名 | 优化器 | 初始学习率 | 权重衰减 | 调度器 | epoch 数 | batch size |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet50 基线 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| EfficientNet-B0 增强版 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| ViT-B/16 可选实验 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

## 6. 实验结果

### 6.1 验证集对比结果

训练完成后，将 `outputs/logs/<实验名>/history.csv` 中的最优结果整理到下表：

| 模型 | 最优 epoch | 验证集 loss | 验证集 accuracy | 备注 |
| --- | --- | --- | --- | --- |
| ResNet50 | 待填写 | 待填写 | 待填写 | 基线 |
| EfficientNet-B0 | 待填写 | 待填写 | 待填写 | 强增强 |
| ViT-B/16 | 待填写 | 待填写 | 待填写 | 可选 |

### 6.2 训练曲线

请将训练脚本生成的曲线图放入 `reports/figures/`，例如：

- `reports/figures/resnet50_curves.png`
- `reports/figures/efficientnet_b0_curves.png`

## 7. Kaggle 提交结果

说明：

- 测试集没有公开真值标签，因此报告中**不展示测试集准确率**。
- 最终结果以提交 `submission.csv` 到 Kaggle 后得到的 `score` 为准。

请将 Kaggle 提交结果截图保存为：

- `reports/figures/kaggle_score.png`

然后在下方插入图片：

![Kaggle Score 截图](figures/kaggle_score.png)

并填写结果表：

| 最终提交模型 | 提交文件路径 | Kaggle Score |
| --- | --- | --- |
| 待填写 | 待填写 | 待填写 |

## 8. 结果分析

请结合实验结果分析：

- 额外数据增强是否带来了收益
- EfficientNet 或 ViT 相比 ResNet50 是否更适合该任务
- 验证集指标与 Kaggle Score 是否一致
- 可能的改进方向有哪些

## 9. 结论

请总结本项目的最终方案、最佳模型及其表现。
