# Dog Breed Identification 作业项目

这是一个基于 PyTorch / torchvision 的犬种图像分类工程，面向 Kaggle `Dog Breed Identification` 竞赛数据格式设计，覆盖如下完整流程：

- 读取 `labels.csv` 并建立 `类别名 <-> 类别索引` 映射。
- 按固定随机种子做分层训练/验证集划分，保证可复现。
- 为训练集、验证集、测试集分别构建预处理与增强流水线。
- 基于 torchvision 预训练模型做迁移学习分类。
- 执行训练、验证、最优模型保存与训练曲线绘制。
- 对测试集批量推理，输出 120 个犬种类别的概率分布。
- 生成符合 Kaggle 官方提交格式的 `submission.csv`。
- 支持基线模型与额外模型对比实验。

## 1. 项目目录

```text
dog-breed-identification/
├─ configs/                     # 实验配置文件
├─ metadata/                    # 自动生成的划分与类别映射
├─ outputs/
│  ├─ checkpoints/              # 训练后保存 best.pth / last.pth
│  ├─ figures/                  # 训练曲线等结果图
│  ├─ logs/                     # train.log、history.csv、summary.json
│  └─ submissions/              # 生成的 submission.csv
├─ reports/
│  ├─ figures/                  # 报告图片，包括 Kaggle score 截图
│  └─ report.md                 # 技术报告模板
├─ scripts/
│  ├─ prepare_metadata.py       # 准备划分与类别映射
│  ├─ train.py                  # 训练脚本
│  └─ predict.py                # 测试集推理与提交文件生成
├─ src/dogbreed/
│  ├─ config.py                 # 配置读取
│  ├─ metadata.py               # 元数据准备
│  ├─ transforms.py             # 数据增强
│  ├─ data.py                   # Dataset / DataLoader
│  ├─ models.py                 # torchvision 模型构建
│  ├─ engine.py                 # 训练/验证/推理循环
│  └─ utils.py                  # 通用工具函数
├─ .gitignore
├─ requirements.txt
└─ README.md
```

## 2. 数据集放置方式

将 Kaggle 下载并解压后的数据按如下结构放在项目根目录：

```text
dog-breed-identification/
├─ train/
├─ test/
├─ labels.csv
└─ sample_submission.csv
```

当前工程默认就是按以上结构读取。

## 3. 环境依赖

你已经说明会在服务器上的 `conda activate myenv` 环境里运行。需要注意：

- `torch` / `torchvision` / `torchaudio` 不建议直接写进普通 `requirements.txt` 后再用默认 `pip install -r requirements.txt` 安装。
- 原因是默认 PyPI 常常会给 Windows 环境装成 CPU 版 `torch`，从而导致 `torch.cuda.is_available()` 为 `False`。
- 正确做法是先按 PyTorch 官方命令安装匹配的 GPU 版三件套，再安装其余 Python 依赖。

其余依赖建议安装：

```bash
pip install -r requirements.txt
```

如果你计划使用 GPU，训练前建议先自检一次：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

正常情况下至少应满足：

- `torch.version.cuda` 不是 `None`
- `torch.cuda.is_available()` 为 `True`

如果这两项不满足，说明当前环境装到的是 CPU 版 PyTorch，或者 `torch` / `torchvision` 版本不匹配。

### 3.1 推荐安装顺序

如果你的 NVIDIA 驱动较老，优先安装 `CUDA 11.8` 对应的 PyTorch 官方 wheel：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

如果你已经升级到较新的显卡驱动，也可以改用 PyTorch 官方页面提供的 `CUDA 12.x` 安装命令。

## 4. 已提供的实验配置

- `configs/resnet50_baseline.yaml`
  - 基线模型，使用 `ResNet50`。
- `configs/efficientnet_b0_strong_aug.yaml`
  - 对比模型，使用 `EfficientNet-B0`，并加入更强的数据增强。
- `configs/vit_b16_finetune.yaml`
  - 额外可选模型，使用 `ViT-B/16` 做全量微调。

## 5. 推荐运行顺序

### 5.1 先生成元数据

```bash
python scripts/prepare_metadata.py --config configs/resnet50_baseline.yaml
```

该步骤会自动生成：

- `metadata/train_val_split_seed42_val20.csv`
- `metadata/class_names.json`
- `metadata/class_to_idx.json`
- `metadata/idx_to_class.json`
- `metadata/test_samples.csv`
- `metadata/dataset_summary.json`

### 5.2 训练基线模型

```bash
python scripts/train.py --config configs/resnet50_baseline.yaml
```

### 5.3 训练对比模型

```bash
python scripts/train.py --config configs/efficientnet_b0_strong_aug.yaml
```

如果你还想加入 Transformer 类模型：

```bash
python scripts/train.py --config configs/vit_b16_finetune.yaml
```

### 5.4 生成测试集提交文件

以 ResNet50 为例：

```bash
python scripts/predict.py --config configs/resnet50_baseline.yaml
```

生成的提交文件默认位于：

```text
outputs/submissions/resnet50_baseline/submission.csv
```

## 6. 输出文件说明

训练完成后，每个实验都会自动保存以下结果：

- `outputs/logs/<实验名>/train.log`
  - 训练过程文本日志。
- `outputs/logs/<实验名>/history.csv`
  - 每个 epoch 的训练/验证损失与准确率。
- `outputs/logs/<实验名>/summary.json`
  - 最优 epoch、最优指标和关键输出路径。
- `outputs/figures/<实验名>/training_curves.png`
  - 损失曲线与准确率曲线。
- `outputs/checkpoints/<实验名>/best.pth`
  - 依据 `monitor` 指标保存的最佳模型。
- `outputs/checkpoints/<实验名>/last.pth`
  - 最后一个 epoch 的模型。
- `outputs/submissions/<实验名>/submission.csv`
  - 可直接上传 Kaggle 的提交文件。

## 7. 关于提交格式与评估指标

- Kaggle 官方评估指标是 `multi-class log loss`。
- 因此测试阶段必须输出每张图片对 120 个类别的概率分布，而不是单一类别标签。
- 本项目的 `predict.py` 会对模型输出做 `softmax`，并按 `sample_submission.csv` 的列顺序生成概率列。
- 报告中不需要给出测试集准确率，因为测试集没有公开真值标签。
- 最终结果以将 `submission.csv` 上传到 Kaggle 后得到的 score 为准。

## 8. 已实现的数据增强策略

基线与增强版配置中已经支持以下增强方式，可通过 YAML 控制开关与强度：

- `RandomResizedCrop`
- `RandomHorizontalFlip`
- `RandomRotation`
- `ColorJitter`
- `RandomGrayscale`
- `RandomPerspective`
- `GaussianBlur`
- `RandAugment`
- `TrivialAugmentWide`
- `RandomErasing`

其中 `efficientnet_b0_strong_aug.yaml` 已经启用了更丰富且合理的额外增强，适合作为加分项实验。

## 9. 模型说明

当前代码对以下 torchvision 分类模型结构做了分类头替换适配：

- ResNet 系列
- EfficientNet / MobileNet / ConvNeXt / MaxVit 等带 `classifier` 头的模型
- Vision Transformer
- Swin Transformer

你当前作业至少可以直接完成：

- `ResNet50` 基线实验
- `EfficientNet-B0` 对比实验

## 10. 写报告时的建议

请在服务器跑完实验后，把以下内容补进 `reports/report.md`：

- 最终使用的模型与配置文件名称
- 数据增强策略
- 损失函数与训练超参数
- 各模型验证集对比结果
- Kaggle 提交 score 截图
- 结果分析与结论

## 11. 参考官方页面

- Kaggle 竞赛数据页: https://www.kaggle.com/competitions/dog-breed-identification/data
- TorchVision 模型文档: https://docs.pytorch.org/vision/stable/models.html
