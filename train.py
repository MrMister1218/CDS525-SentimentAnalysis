"""
训练脚本
支持 Bi-LSTM + Attention 和 FinBERT 模型的训练、验证和测试
"""

import argparse
import os
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
from models.baseline import BiLSTMAttention, create_model as create_bilstm_model
from models.finbert_model import FinBERTClassifier, FinBERTTokenizer, create_finbert_model


# ==================== 数据集类定义 ====================

class TextDataset(Dataset):
    """
    文本数据集类
    用于 Bi-LSTM 模型的训练（基于词索引）
    """
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

        # 文本转词索引
        tokens = text.split()[:self.max_len]
        indices = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]

        # Padding
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class FinBERTDataset(Dataset):
    """
    FinBERT 数据集类
    用于 FinBERT 模型的训练
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 使用 FinBERT 分词器编码
        encoding = self.tokenizer.encode(
            text,
            padding=False,
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask, torch.tensor(label, dtype=torch.long)


# ==================== 数据加载函数 ====================

def load_data(data_path, test_size=0.2, val_size=0.1):
    """
    加载并划分数据集

    Args:
        data_path: 数据文件路径 (CSV 格式，需包含 'text' 和 'label' 列)
        test_size: 测试集比例
        val_size: 验证集比例 (从训练集中划分)

    Returns:
        train_texts, val_texts, test_texts: 文本数据
        train_labels, val_labels, test_labels: 标签数据
        label_map: 标签映射
    """
    # 加载数据
    df = pd.read_csv(data_path)

    # 确保包含必要的列
    assert 'text' in df.columns and 'label' in df.columns, "数据必须包含 'text' 和 'label' 列"

    texts = df['text'].values
    labels = df['label'].values

    # 创建标签映射
    unique_labels = sorted(labels.unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_map.items()}

    labels = np.array([label_map[l] for l in labels])

    # 划分数据集: 训练+验证 / 测试
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    # 从训练集划分验证集
    val_ratio = val_size / (1 - test_size)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=val_ratio, random_state=42, stratify=train_val_labels
    )

    print(f"数据集划分: 训练集 {len(train_texts)}, 验证集 {len(val_texts)}, 测试集 {len(test_texts)}")
    print(f"标签映射: {label_map}")

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_map


def build_vocab(texts, vocab_size=50000):
    """
    构建词汇表

    Args:
        texts: 文本列表
        vocab_size: 词汇表大小

    Returns:
        vocab: 词汇表字典 {word: index}
    """
    from collections import Counter

    # 统计词频
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.split())

    # 构建词汇表（保留 <PAD>=0, <UNK>=1）
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(vocab_size - 2):
        vocab[word] = len(vocab)

    print(f"词汇表大小: {len(vocab)}")
    return vocab


# ==================== 训练相关函数 ====================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """
    训练一个 epoch

    Args:
        model: 模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备

    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
    """
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

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

        # 前向传播
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)

        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        # 计算损失
        loss = criterion(logits, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 记录预测
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """
    验证/测试评估

    Args:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
        all_preds: 所有预测
        all_labels: 所有真实标签
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

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

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy, all_preds, all_labels


# ==================== 主训练函数 ====================

def train(args):
    """
    主训练函数

    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 创建结果目录
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)

    # 加载数据
    print("\n=== 加载数据 ===")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels, label_map = load_data(
        args.data_path, test_size=args.test_size, val_size=args.val_size
    )

    # 根据模型类型创建数据加载器
    if args.model == 'bilstm':
        # 构建词汇表
        vocab = build_vocab(train_texts, vocab_size=args.vocab_size)

        # 创建数据集
        train_dataset = TextDataset(train_texts, train_labels, vocab, max_len=args.max_len)
        val_dataset = TextDataset(val_texts, val_labels, vocab, max_len=args.max_len)
        test_dataset = TextDataset(test_texts, test_labels, vocab, max_len=args.max_len)

        # 创建模型
        model = create_bilstm_model({
            'vocab_size': len(vocab),
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_classes': args.num_classes,
            'dropout': args.dropout
        })

    elif args.model == 'finbert':
        # 创建分词器
        tokenizer = FinBERTTokenizer(
            model_name=args.pretrained_model,
            max_length=args.max_len
        )

        # 创建数据集
        train_dataset = FinBERTDataset(train_texts, train_labels, tokenizer, max_len=args.max_len)
        val_dataset = FinBERTDataset(val_texts, val_labels, tokenizer, max_len=args.max_len)
        test_dataset = FinBERTDataset(test_texts, test_labels, tokenizer, max_len=args.max_len)

        # 创建模型
        model = create_finbert_model({
            'model_name': args.pretrained_model,
            'num_classes': args.num_classes,
            'dropout': args.dropout,
            'freeze_base': args.freeze_bert
        })

    else:
        raise ValueError(f"不支持的模型类型: {args.model}")

    model = model.to(device)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 设置损失函数
    if args.loss_fn == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif args.loss_fn == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"不支持的损失函数: {args.loss_fn}")

    # 设置优化器
    if args.model == 'finbert' and args.freeze_bert:
        # 如果冻结 BERT，只优化分类层
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练循环
    print("\n=== 开始训练 ===")
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 记录指标
        metrics['epoch'].append(epoch + 1)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['val_loss'].append(val_loss)
        metrics['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(results_dir, 'best_model.pt'))
            print(f"  -> 保存最佳模型 (Val Acc: {best_val_acc:.4f})")

    # 保存训练指标到 CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_path = os.path.join(results_dir, 'training_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n训练指标已保存到: {metrics_path}")

    # 测试集评估
    print("\n=== 测试集评估 ===")
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best_model.pt')))
    test_loss, test_acc, test_preds, test_labels_np = validate(model, test_loader, criterion, device)

    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    print("\n分类报告:")
    print(classification_report(test_labels_np, test_preds, target_names=list(label_map.keys())))

    # 导出前 100 条预测结果对比表
    print("\n=== 导出预测结果对比表 ===")
    comparison_df = pd.DataFrame({
        'text': test_texts[:100],
        'true_label': [label_map[l] for l in test_labels_np[:100]],
        'pred_label': [label_map[l] for l in test_preds[:100]]
    })
    # 添加预测是否正确
    comparison_df['correct'] = comparison_df['true_label'] == comparison_df['pred_label']

    comparison_path = os.path.join(results_dir, 'prediction_comparison.csv')
    comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"预测结果对比表已保存到: {comparison_path}")

    print("\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")


# ==================== 命令行参数 ====================

def parse_args():
    parser = argparse.ArgumentParser(description='情感分类模型训练')

    # 模型参数
    parser.add_argument('--model', type=str, default='bilstm',
                        choices=['bilstm', 'finbert'],
                        help='选择模型类型: bilstm 或 finbert')
    parser.add_argument('--data_path', type=str, default='data/train.csv',
                        help='训练数据路径 (CSV 格式，需包含 text 和 label 列)')

    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--loss_fn', type=str, default='crossentropy',
                        choices=['crossentropy', 'bce'],
                        help='损失函数类型')

    # 数据划分
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='验证集比例')

    # BiLSTM 模型参数
    parser.add_argument('--vocab_size', type=int, default=50000,
                        help='词汇表大小 (仅 BiLSTM)')
    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='词嵌入维度 (仅 BiLSTM)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='LSTM 隐藏层维度 (仅 BiLSTM)')
    parser.add_argument('--max_len', type=int, default=128,
                        help='最大序列长度')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='分类类别数')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout 比率')

    # FinBERT 模型参数
    parser.add_argument('--pretrained_model', type=str, default='ProsusAI/finbert',
                        help='预训练模型名称')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='是否冻结 BERT 基础模型参数')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
