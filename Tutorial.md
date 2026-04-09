# 金融新闻情感分析 - 使用教程

> 本教程面向零基础用户，一步一步讲解如何训练模型和查看可视化结果。

---

## 一、项目结构一览

```
CDS525-SentimentAnalysis/
├── data/
│   └── all-data.csv          # 训练数据（4846条金融新闻，已标注）
├── models/
│   ├── __init__.py           # 模型导出文件（勿动）
│   ├── baseline.py           # BiLSTM + Attention 模型代码
│   └── finbert_model.py      # FinBERT 模型代码
├── scripts/
│   ├── preprocess.py         # 数据预处理脚本（勿需手动运行）
│   └── plot_results.py       # 可视化脚本
├── results/                  # 训练结果目录（运行训练后自动生成）
├── train.py                  # 训练脚本（核心）
├── requirements.txt          # 依赖列表
├── .gitignore                # Git忽略规则
├── README.md                 # 项目说明
└── Tutorial.md              # 本教程
```

---

## 二、安装依赖

### 2.1 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

### 2.2 安装依赖包

```bash
pip install -r requirements.txt
```

> 如果安装 PyTorch 时速度慢，可以指定国内镜像：
> ```bash
> pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
> ```

### 2.3 验证安装成功

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

看到类似输出即为成功：

```
PyTorch 2.x.x, CUDA: True   # GPU可用（推荐）
PyTorch 2.x.x, CUDA: False  # 仅CPU可用（可训练，速度较慢）
```

---

## 三、快速开始：训练模型

### 3.1 训练 BiLSTM + Attention 模型

**BiLSTM + CrossEntropyLoss（默认）**
```bash
python train.py --model baseline --loss_fn crossentropy --lr 0.001 --batch_size 32 --epochs 10
```

**BiLSTM + Label Smoothing（带标签平滑）**
```bash
python train.py --model baseline --loss_fn labelsmoothing --lr 0.001 --batch_size 32 --epochs 10
```

### 3.2 训练 FinBERT 模型

```bash
python train.py --model finbert --loss_fn crossentropy --lr 2e-5 --batch_size 16 --epochs 5
```

> FinBERT 首次运行会从 HuggingFace 下载预训练权重（约 400MB），请耐心等待。

### 3.3 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型类型：`baseline`（BiLSTM）或 `finbert` | `baseline` |
| `--loss_fn` | 损失函数：`crossentropy` 或 `labelsmoothing` | `crossentropy` |
| `--lr` | 学习率 | `0.001`（BiLSTM）/ `2e-5`（FinBERT） |
| `--batch_size` | 批次大小 | `32`（BiLSTM）/ `16`（FinBERT） |
| `--epochs` | 训练轮数 | `10` |
| `--data_path` | 数据文件路径 | `data/all-data.csv` |
| `--output_dir` | 结果输出目录 | `results` |

### 3.4 参数组合示例

```bash
# 实验1：BiLSTM + CrossEntropy + 高学习率
python train.py --model baseline --loss_fn crossentropy --lr 0.001 --batch_size 32 --use_class_weight

# 实验2：BiLSTM + Label Smoothing + 低学习率
python train.py --model baseline --loss_fn labelsmoothing --lr 0.001 --batch_size 32 --use_class_weight

# 实验3：FinBERT + 低学习率
python train.py --model finbert --loss_fn crossentropy --lr 2e-5 --batch_size 16 --use_class_weight


```

---

## 四、查看训练结果

### 4.1 results 目录结构

每次训练会在 `results/` 下生成一个以参数命名的文件夹，例如：

```
results/
├── baseline_loss_crossentropy_lr_0p001_bs_32/   # BiLSTM + CE 实验
│   ├── experiment_summary.json       # 实验配置与最终指标
│   ├── training_metrics.csv         # 每轮训练的详细数据
│   ├── prediction_comparison.csv    # 测试集预测对比
│   └── best_model.pt                # 模型权重（勿提交）
│
├── baseline_loss_labelsmoothing_lr_0p001_bs_32/  # BiLSTM + LS 实验
│   └── ...
│
└── finbert_loss_crossentropy_lr_0p00002_bs_16/   # FinBERT 实验
    └── ...
```

### 4.2 每个文件的作用

| 文件 | 作用 | 重要程度 |
|------|------|---------|
| `experiment_summary.json` | 模型名、损失函数、学习率、测试准确率等汇总 | ★★★ |
| `training_metrics.csv` | 每轮（epoch）的 train_loss、train_acc、val_loss、val_acc | ★★★ |
| `prediction_comparison.csv` | 测试集文本、真实标签、预测标签、是否正确 | ★★★ |
| `best_model.pt` | 模型权重文件（用于加载模型做推理） | ★（大文件，不提交） |

### 4.3 查看实验汇总

直接打开 `experiment_summary.json` 即可看到关键指标：

```json
{
  "model": "BiLSTM + Attention",
  "loss_fn": "CrossEntropyLoss",
  "lr": 0.001,
  "batch_size": 32,
  "best_val_acc": 0.5938,
  "test_acc": 0.66,
  "test_loss": 0.9245
}
```

---

## 五、可视化结果

### 5.1 生成可视化图表

所有图表由 `scripts/plot_results.py` 自动生成。

**生成所有图表（推荐）**
```bash
python scripts/plot_results.py --results_dir results --output_dir results --separate
```

这会生成：
- 6 张分图（BiLSTM loss、FinBERT loss、损失对比、BiLSTM acc、FinBERT acc、准确率对比）
- 1 张合并曲线图
- HTML 预测表格
- 混淆矩阵热力图
- 汇总 CSV

**仅生成合并曲线图（不生成分图）**
```bash
python scripts/plot_results.py --results_dir results --output_dir results
```

### 5.2 可视化文件详解

#### 训练曲线图（Training Curves）

这些图展示模型在训练过程中 loss 和 accuracy 的变化趋势，帮助判断模型是否正常收敛。

**横轴**：训练的轮次（Epoch 1, 2, 3, ...）
**纵轴**：loss 值或 accuracy 值
**线条含义**：
- 蓝色实线（Train Loss/Acc）= 训练集表现
- 红色虚线（Val Loss/Acc）= 验证集表现

**如何解读**：
- loss 下降 + acc 上升 = 正常训练
- train acc 持续上升，但 val acc 停滞或下降 = **过拟合**（模型记忆了训练数据，泛化能力差）
- train loss 上升 = 训练出错（如学习率过大）
- val loss 在最低点附近 = 模型收敛最佳点（我们用验证集挑出来的就是这个点）

#### 合并曲线图（training_curves_combined.png/pdf）

将所有实验的 loss 和 accuracy 曲线放在一张图里，方便横向对比不同模型的表现。

#### 分图说明

| 文件 | 内容 |
|------|------|
| `fig01_bilstm_loss.png` | BiLSTM 两个实验的 loss 曲线对比 |
| `fig02_finbert_loss.png` | FinBERT 的 loss 曲线 |
| `fig03_loss_comparison.png` | 所有实验 loss 曲线汇总对比 |
| `fig04_bilstm_acc.png` | BiLSTM 两个实验的 accuracy 曲线对比 |
| `fig05_finbert_acc.png` | FinBERT 的 accuracy 曲线 |
| `fig06_acc_comparison.png` | 所有实验 accuracy 曲线汇总对比 |

#### 混淆矩阵热力图（prediction_heatmap.png）

展示模型在测试集上的分类表现。行是真实标签，列是预测标签，颜色越深表示数量越多。

```
              预测
            n  p  g
真实  n    120 10  5
      p     8 100 12
      g     3  15 97
```
- n=neutral（中性）, p=positive（正面）, g=negative（负面）
- 对角线越亮、非对角线越暗越好

#### HTML 预测表格

每个实验生成一个 `.html` 文件，用浏览器打开可看到彩色表格：

| 列 | 说明 |
|---|---|
| text | 金融新闻原文（截取前200字） |
| true_label | 真实情感标签（neutral/positive/negative） |
| pred_label | 模型预测标签 |
| correct | 绿色=预测正确，红色=预测错误 |

---

## 六、运行完整实验示例

假设你想完整复现项目中的所有实验，按以下顺序执行：

### Step 1: 训练 BiLSTM + CrossEntropy
```bash
python train.py --model baseline --loss_fn crossentropy --lr 0.001 --batch_size 32 --epochs 10
```

### Step 2: 训练 BiLSTM + Label Smoothing
```bash
python train.py --model baseline --loss_fn labelsmoothing --lr 0.001 --batch_size 32 --epochs 10
```

### Step 3: 训练 FinBERT
```bash
python train.py --model finbert --loss_fn crossentropy --lr 2e-5 --batch_size 16 --epochs 5
```

### Step 4: 生成可视化
```bash
python scripts/plot_results.py --results_dir results --output_dir results --separate
```

### Step 5: 查看结果

```
results/
├── baseline_loss_crossentropy_lr_0p001_bs_32/
│   └── ...（3个核心文件）
├── baseline_loss_labelsmoothing_lr_0p001_bs_32/
│   └── ...（3个核心文件）
├── finbert_loss_crossentropy_lr_0p00002_bs_16/
│   └── ...（3个核心文件）
├── training_curves_combined.png/pdf    # 训练曲线
├── fig01~fig06.png                     # 分图
├── prediction_heatmap.png             # 混淆矩阵热力图
├── prediction_table_*.html             # HTML 预测表
└── experiment_summary_all.csv         # 所有实验汇总表
```

---

## 七、常见问题

### Q1: 训练时显示 "CUDA out of memory"
显存不足。减小 batch_size：
```bash
python train.py --model finbert --batch_size 8
```

### Q2: FinBERT 下载失败
网络问题导致 HuggingFace 下载超时。可以先手动下载：
```bash
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')"
```
或者使用代理/VPN。

### Q3: 训练结果准确率很低
- 检查数据是否正确加载（`data/all-data.csv` 是否存在）
- 尝试调低学习率（如 `0.0001`）
- BiLSTM 数据量小（4846条），准确率 60-70% 是正常范围

### Q4: results 目录变得很大
`.gitignore` 已配置自动忽略 `*.pt`、`*.png`、`*.pdf`、`*.html` 等大文件。如需清空重新训练：
```bash
rm -rf results/*
# 然后重新运行训练脚本
```

### Q5: 如何加载训练好的模型做推理
```python
import torch
from models.baseline import BiLSTMAttention

# 加载权重
model = BiLSTMAttention(vocab_size=30000, embedding_dim=128,
                         hidden_dim=128, num_classes=3, dropout=0.3)
model.load_state_dict(torch.load('results/.../best_model.pt'))
model.eval()

# 推理示例
# （需要先将文本转为词索引，具体见 train.py 中的 TextDataset 类）
```
