"""
主训练脚本
支持 BiLSTM_Attention 和 FinBERT 模型的训练与对比实验
"""

import os
import re
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from models.baseline import BiLSTMAttention, create_bilstm_model
from models.finbert_model import FinBERTClassifier, FinBERTTokenizer, create_finbert_model


# ==================== Label Smoothing Loss ====================

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    减少模型对训练数据的过度自信，提升泛化能力
    """
    def __init__(self, smoothing=0.1, num_classes=3):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, pred, target):
        """
        Args:
            pred: (batch, num_classes) logits
            target: (batch,) integer labels
        """
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)

        # 构建 label distribution: [1-smoothing, smooth, ..., smooth]
        one_hot = torch.zeros_like(pred).scatter_(
            1, target.unsqueeze(1), 1.0
        )
        smooth_labels = one_hot * confidence + (1 - one_hot) * smooth_value

        log_probs = torch.log_softmax(pred, dim=1)
        loss = (-smooth_labels * log_probs).sum(dim=1).mean()
        return loss


# ==================== 数据集类 ====================

class TextDataset(Dataset):
    """BiLSTM 文本数据集（基于词索引）"""
    def __init__(self, texts, labels, vocab, max_len=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        tokens = text.lower().split()[:self.max_len]
        indices = [
            self.vocab.get(token, self.vocab.get('<UNK>', 1))
            for token in tokens
        ]
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class FinBERTDataset(Dataset):
    """FinBERT 文本数据集"""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            torch.tensor(label, dtype=torch.long),
        )


# ==================== 数据加载 ====================

def load_data(data_path, test_ratio=0.1, val_ratio=0.1, random_state=42):
    """
    加载 CSV 数据并按 8:1:1 划分
    CSV 格式：无表头，第0列=label，第1列=text
    """
    for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv(data_path, header=None, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    df.columns = ['label', 'text']
    texts = df['text'].astype(str).values

    label_map = {'neutral': 0, 'positive': 1, 'negative': 2}
    labels = np.array([label_map[l] for l in df['label'].values])

    # 先划分测试集
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_ratio, random_state=random_state, stratify=labels
    )
    # 再划分验证集
    val_adj = val_ratio / (1 - test_ratio)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_adj, random_state=random_state, stratify=train_val_labels
    )

    print(f"[Data] train={len(train_texts)}  val={len(val_texts)}  test={len(test_texts)}")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_map


def build_vocab(texts, vocab_size=30000, min_freq=2):
    """构建词汇表"""
    from collections import Counter
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.lower().split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(vocab_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)
    print(f"[Vocab] size={len(vocab)}")
    return vocab


# ==================== 训练 / 评估函数 ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in dataloader:
        if len(batch) == 2:
            input_ids, labels = batch
            attention_mask = None
        else:
            input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                input_ids, labels = batch
                attention_mask = None
            else:
                input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(dataloader), accuracy_score(all_labels, all_preds), all_preds, all_labels


# ==================== 单次实验运行 ====================

def run_experiment(
    model_name, loss_fn_name, lr, batch_size,
    train_loader, val_loader, test_loader,
    num_classes, device,
    exp_dir,
    class_weight=None,
):
    """
    运行一次实验，返回训练指标
    """
    # --- 模型 ---
    if model_name == 'baseline':
        model = create_bilstm_model({'vocab_size': 50000, 'num_classes': num_classes})
    else:  # finbert
        model = create_finbert_model({'num_classes': num_classes, 'dropout': 0.1})
    model = model.to(device)

    # --- 损失函数 ---
    if loss_fn_name == 'labelsmoothing':
        criterion = LabelSmoothingCrossEntropy(smoothing=0.1, num_classes=num_classes)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weight)

    # --- 优化器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # --- 训练循环 ---
    epochs = 10
    metrics = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(round(train_loss, 4))
        metrics['train_acc'].append(round(train_acc, 4))
        metrics['val_loss'].append(round(val_loss, 4))
        metrics['val_acc'].append(round(val_acc, 4))

        print(f"  Epoch {epoch:2d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  |  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # --- 保存模型 ---
    os.makedirs(exp_dir, exist_ok=True)
    torch.save(best_state, os.path.join(exp_dir, 'best_model.pt'))

    # --- 测试集评估 ---
    model.load_state_dict(best_state)
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"\n  [Test] loss={test_loss:.4f}  acc={test_acc:.4f}")

    # --- 保存训练指标 CSV ---
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(exp_dir, 'training_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)

    # --- 保存训练指标 JSON ---
    summary = {
        'model': model_name,
        'loss_fn': loss_fn_name,
        'lr': lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'best_val_acc': round(best_val_acc, 4),
        'test_acc': round(test_acc, 4),
        'test_loss': round(test_loss, 4),
        'train_final_loss': round(metrics['train_loss'][-1], 4),
        'train_final_acc': round(metrics['train_acc'][-1], 4),
        'val_final_loss': round(metrics['val_loss'][-1], 4),
        'val_final_acc': round(metrics['val_acc'][-1], 4),
    }
    with open(os.path.join(exp_dir, 'experiment_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # --- 导出测试集前 100 条预测对比表 ---
    # 获取原始测试文本
    test_texts = [test_loader.dataset.texts[i] if hasattr(test_loader.dataset, 'texts') else '' for i in range(len(test_loader.dataset))]
    label_map_inv = {0: 'neutral', 1: 'positive', 2: 'negative'}

    comparison_df = pd.DataFrame({
        'text':       [str(t)[:200] for t in test_texts[:100]],
        'true_label': [label_map_inv.get(l, l) for l in test_labels[:100]],
        'pred_label': [label_map_inv.get(p, p) for p in test_preds[:100]],
    })
    comparison_df['correct'] = comparison_df['true_label'] == comparison_df['pred_label']
    comparison_df.to_csv(
        os.path.join(exp_dir, 'prediction_comparison.csv'),
        index=False, encoding='utf-8-sig'
    )

    return summary


# ==================== 主入口 ====================

def main():
    parser = argparse.ArgumentParser(description='金融新闻情感分类训练脚本')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'finbert'],
                        help='baseline=BiLSTM_Attention, finbert=FinBERT')
    parser.add_argument('--loss_fn', type=str, default='crossentropy',
                        choices=['crossentropy', 'labelsmoothing'],
                        help='损失函数类型')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数（默认 10）')
    parser.add_argument('--data_path', type=str, default='data/all-data.csv',
                        help='数据文件路径')
    parser.add_argument('--max_len', type=int, default=128,
                        help='最大序列长度')
    parser.add_argument('--vocab_size', type=int, default=30000,
                        help='词汇表大小（仅 baseline）')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='词嵌入维度（仅 baseline）')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM 隐藏层维度（仅 baseline）')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 比率')
    parser.add_argument('--finbert_model', type=str, default='yiyanghkust/finbert-tone',
                        help='FinBERT 预训练模型名称')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='冻结 BERT backbone（仅 finbert）')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='分类类别数')
    parser.add_argument('--use_class_weight', action='store_true',
                        help='启用类别权重，缓解数据集不平衡（neutral:59.4% / positive:28.1% / negative:12.5%）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='结果输出目录')
    args = parser.parse_args()

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型与损失函数的可读名称映射
    MODEL_DISPLAY = {
        'baseline':  'BiLSTM + Attention',
        'finbert':   'FinBERT',
    }
    LOSS_DISPLAY = {
        'crossentropy':    'CrossEntropyLoss',
        'labelsmoothing':  'Label Smoothing Loss',
    }
    model_display = MODEL_DISPLAY.get(args.model, args.model.title())
    loss_display  = LOSS_DISPLAY.get(args.loss_fn, args.loss_fn)

    print(f"\n{'='*60}")
    print(f"  Model:       {model_display}")
    print(f"  Loss:        {loss_display}")
    print(f"  LR:          {args.lr}")
    print(f"  Batch Size:  {args.batch_size}")
    print(f"  Device:      {device}")
    print(f"{'='*60}\n")

    # --- 加载数据 ---
    print("=== 加载数据 ===")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_map = load_data(
        args.data_path, test_ratio=0.1, val_ratio=0.1, random_state=args.seed
    )

    # --- 类别权重（按样本数倒数，少数类权重更高）---
    class_weight = None
    if args.use_class_weight:
        from collections import Counter
        counts = Counter(train_labels)
        total = len(train_labels)
        # neutral=0, positive=1, negative=2
        # 权重 = sqrt(总样本数 / 该类样本数)，避免极端值
        weights = [np.sqrt(total / counts[i]) for i in range(args.num_classes)]
        class_weight = torch.tensor(weights, dtype=torch.float32).to(device)
        print(f"[ClassWeight] {dict(enumerate(weights))}  (sqrt逆频率加权)")

    # --- 根据模型类型准备 DataLoader ---
    if args.model == 'baseline':
        vocab = build_vocab(train_texts, vocab_size=args.vocab_size)
        train_ds = TextDataset(train_texts, train_labels, vocab, max_len=args.max_len)
        val_ds   = TextDataset(val_texts,   val_labels,   vocab, max_len=args.max_len)
        test_ds  = TextDataset(test_texts,  test_labels,  vocab, max_len=args.max_len)
        train_ds.texts = train_texts
        test_ds.texts  = test_texts
        val_ds.texts   = val_texts
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

        # 更新模型配置
        from models.baseline import BiLSTMAttention
        class ConfiguredBiLSTM(BiLSTMAttention):
            def __init__(self, cfg):
                super().__init__(
                    vocab_size=cfg['vocab_size'],
                    embedding_dim=cfg['embedding_dim'],
                    hidden_dim=cfg['hidden_dim'],
                    num_classes=cfg['num_classes'],
                    dropout=cfg['dropout'],
                )
        import types
        def create_bilstm_model_cfg(cfg):
            m = BiLSTMAttention(
                vocab_size=cfg['vocab_size'],
                embedding_dim=cfg.get('embedding_dim', 128),
                hidden_dim=cfg.get('hidden_dim', 128),
                num_classes=cfg.get('num_classes', 3),
                dropout=cfg.get('dropout', 0.3),
            )
            return m
        # 直接在模块中替换（hack，但最简洁）
        import models.baseline as bl_module
        bl_module.create_bilstm_model = create_bilstm_model_cfg

    else:  # finbert
        tokenizer = FinBERTTokenizer(model_name=args.finbert_model, max_length=args.max_len)
        train_ds = FinBERTDataset(train_texts, train_labels, tokenizer)
        val_ds   = FinBERTDataset(val_texts,   val_labels,   tokenizer)
        test_ds  = FinBERTDataset(test_texts,  test_labels,  tokenizer)
        train_ds.texts = train_texts
        test_ds.texts  = test_texts
        val_ds.texts   = val_texts
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size)

        # freeze_bert 处理
        import models.finbert_model as fb_module
        orig_create = fb_module.create_finbert_model
        def create_finbert_model_cfg(cfg):
            m = orig_create({**cfg, 'freeze_base': args.freeze_bert})
            return m
        fb_module.create_finbert_model = create_finbert_model_cfg

    # --- 构建实验目录名 ---
    # 统一使用 .6f 格式，确保所有 lr 值都能正确转换，无歧义
    # 例如: 0.1->"0p1", 0.01->"0p01", 0.001->"0p001", 2e-5->"0p00002"
    lr_str = f"{args.lr:.6f}".rstrip('0').rstrip('.')
    lr_str = lr_str.replace('.', 'p')
    bs_str = str(args.batch_size)
    exp_name = f"{args.model}_loss_{args.loss_fn}_lr_{lr_str}_bs_{bs_str}"
    exp_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # --- 运行实验 ---
    print(f"\n=== 开始训练 | {exp_name} ===")
    summary = run_experiment(
        model_name=args.model,
        loss_fn_name=args.loss_fn,
        lr=args.lr,
        batch_size=args.batch_size,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=args.num_classes,
        device=device,
        exp_dir=exp_dir,
        class_weight=class_weight,
    )

    # --- 打印最终结果 ---
    print(f"\n{'='*60}")
    print(f"  实验完成: {exp_name}")
    print(f"  最佳验证准确率: {summary['best_val_acc']:.4f}")
    print(f"  测试准确率:      {summary['test_acc']:.4f}")
    print(f"{'='*60}")
    print(f"\n  training_metrics.csv  -> {exp_dir}")
    print(f"  experiment_summary.json -> {exp_dir}")
    print(f"  prediction_comparison.csv -> {exp_dir}")
    print(f"  best_model.pt -> {exp_dir}")


if __name__ == '__main__':
    main()
