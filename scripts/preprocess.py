"""
数据预处理脚本
负责数据加载、标签映射、数据集划分、Dataset 构建与 DataLoader 返回
支持 Bi-LSTM（简单分词）和 FinBERT（HuggingFace BertTokenizer）
"""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from collections import Counter


# ==================== 标签映射 ====================

LABEL_MAP = {
    'neutral':  0,
    'positive': 1,
    'negative': 2,
}
LABEL_INVERSE_MAP = {v: k for k, v in LABEL_MAP.items()}


# ==================== Bi-LSTM 数据集类 ====================

class BiLSTMDataset(Dataset):
    """
    Bi-LSTM 文本数据集
    使用简单空格分词 + 词汇表映射
    """

    def __init__(self, texts, labels, vocab, max_len=128):
        """
        Args:
            texts: 文本列表
            labels: 标签列表（整数）
            vocab: 词汇表字典 {word: index}
            max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def _tokenize(self, text):
        """简单分词：转小写 + 去除标点 + 空格分割"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        tokens = self._tokenize(text)[:self.max_len]
        indices = [
            self.vocab.get(token, self.vocab['<UNK>'])
            for token in tokens
        ]

        # Padding
        if len(indices) < self.max_len:
            indices += [self.vocab['<PAD>']] * (self.max_len - len(indices))

        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


# ==================== FinBERT 数据集类 ====================

class FinBERTDataset(Dataset):
    """
    FinBERT 文本数据集
    使用 HuggingFace BertTokenizer 分词
    """

    def __init__(self, texts, labels, tokenizer):
        """
        Args:
            texts: 文本列表
            labels: 标签列表（整数）
            tokenizer: FinBERTTokenizer 实例（自带 max_length）
        """
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
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return (
            encoding['input_ids'].squeeze(0),
            encoding['attention_mask'].squeeze(0),
            torch.tensor(label, dtype=torch.long),
        )


# ==================== 词汇表构建 ====================

def build_vocab(texts, vocab_size=30000, min_freq=2):
    """
    构建词汇表（基于训练集）

    Args:
        texts: 文本列表
        vocab_size: 词汇表最大容量
        min_freq: 最小词频阈值

    Returns:
        vocab: 词汇表字典 {word: index}
    """
    word_counts = Counter()
    for text in texts:
        word_counts.update(self._tokenize(text) if hasattr(self, '_tokenize') else text.lower().split())

    # 固定特殊 token: <PAD>=0, <UNK>=1
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(vocab_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)

    print(f"[Vocab] size={len(vocab)} (raw={len(word_counts)}, min_freq={min_freq})")
    return vocab


def _tokenize_text(text):
    """独立分词函数，供 build_vocab 使用"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return text.split()


def build_vocab(texts, vocab_size=30000, min_freq=2):
    """构建词汇表（基于训练集）"""
    word_counts = Counter()
    for text in texts:
        word_counts.update(_tokenize_text(str(text)))

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, count in word_counts.most_common(vocab_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)

    print(f"[Vocab] size={len(vocab)} (raw={len(word_counts)}, min_freq={min_freq})")
    return vocab


# ==================== 数据加载与划分 ====================

def load_and_split_data(data_path, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """
    加载 CSV 数据并按 8:1:1 划分训练集、验证集、测试集

    Args:
        data_path: 数据文件路径
        val_ratio: 验证集比例（默认 0.1）
        test_ratio: 测试集比例（默认 0.1）
        random_state: 随机种子

    Returns:
        train_texts, val_texts, test_texts: 文本数据
        train_labels, val_labels, test_labels: 标签数据（整数）
    """
    # 读取（自动检测编码）
    for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            df = pd.read_csv(data_path, header=None, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    assert df.shape[1] >= 2, "数据文件至少需要 2 列（标签列 + 文本列）"

    df.columns = ['label', 'text']
    df['text'] = df['text'].astype(str)

    # 标签映射为整数
    df['label_int'] = df['label'].map(LABEL_MAP)
    assert df['label_int'].notna().all(), "存在未知标签，请检查标签是否在 neutral/positive/negative 中"

    texts = df['text'].values
    labels = df['label_int'].values

    print(f"[Data] {len(texts)} samples, label distribution:")
    for name, idx in LABEL_MAP.items():
        print(f"       {name} ({idx}): {(labels == idx).sum()}")

    # 先划分测试集
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=labels,
    )

    # 再从训练集中划分验证集
    val_adj_ratio = val_ratio / (1 - test_ratio)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels,
        test_size=val_adj_ratio,
        random_state=random_state,
        stratify=train_val_labels,
    )

    print(f"\n[Split] train={len(train_texts)}  val={len(val_texts)}  test={len(test_texts)}")
    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


# ==================== 主入口函数 ====================

def get_dataloaders(
    data_path,
    bilstm_batch_size=32,
    finbert_batch_size=16,
    max_len=128,
    vocab_size=30000,
    min_freq=2,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42,
    finbert_model_name='yiyanghkust/finbert-tone',
):
    """
    一站式返回所有 DataLoader 及相关对象

    Args:
        data_path: 数据文件路径
        bilstm_batch_size: BiLSTM 批大小
        finbert_batch_size: FinBERT 批大小
        max_len: 最大序列长度
        vocab_size: 词汇表最大容量
        min_freq: 最小词频阈值
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
        finbert_model_name: FinBERT 预训练模型名称

    Returns:
        dict: {
            'train/val/test_loader_bilstm',
            'train/val/test_loader_finbert',
            'vocab', 'label_map', 'label_inverse_map', 'tokenizer',
        }
    """
    # 1. 加载与划分数据
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_split_data(
        data_path, val_ratio=val_ratio, test_ratio=test_ratio, random_state=random_state
    )

    # 2. 构建词汇表（仅用训练集）
    vocab = build_vocab(train_texts, vocab_size=vocab_size, min_freq=min_freq)

    # 3. 加载 FinBERT 分词器
    tokenizer = BertTokenizer.from_pretrained(finbert_model_name)
    print(f"[Tokenizer] loaded: {finbert_model_name}")

    # 4. BiLSTM DataLoader
    train_ds_bl = BiLSTMDataset(train_texts, train_labels, vocab, max_len=max_len)
    val_ds_bl   = BiLSTMDataset(val_texts,   val_labels,   vocab, max_len=max_len)
    test_ds_bl  = BiLSTMDataset(test_texts,  test_labels,  vocab, max_len=max_len)

    train_loader_bl = DataLoader(train_ds_bl, batch_size=bilstm_batch_size, shuffle=True)
    val_loader_bl   = DataLoader(val_ds_bl,   batch_size=bilstm_batch_size, shuffle=False)
    test_loader_bl  = DataLoader(test_ds_bl,  batch_size=bilstm_batch_size, shuffle=False)

    # 5. FinBERT DataLoader
    train_ds_fb = FinBERTDataset(train_texts, train_labels, tokenizer)
    val_ds_fb   = FinBERTDataset(val_texts,   val_labels,   tokenizer)
    test_ds_fb  = FinBERTDataset(test_texts,  test_labels,  tokenizer)

    train_loader_fb = DataLoader(train_ds_fb, batch_size=finbert_batch_size, shuffle=True)
    val_loader_fb   = DataLoader(val_ds_fb,   batch_size=finbert_batch_size, shuffle=False)
    test_loader_fb  = DataLoader(test_ds_fb,  batch_size=finbert_batch_size, shuffle=False)

    print(f"\n[Done] BiLSTM bs={bilstm_batch_size} max_len={max_len} | FinBERT bs={finbert_batch_size}")

    return {
        'train_loader_bilstm': train_loader_bl,
        'val_loader_bilstm':   val_loader_bl,
        'test_loader_bilstm':  test_loader_bl,
        'train_loader_finbert': train_loader_fb,
        'val_loader_finbert':  val_loader_fb,
        'test_loader_finbert': test_loader_fb,
        'vocab':               vocab,
        'label_map':           LABEL_MAP,
        'label_inverse_map':   LABEL_INVERSE_MAP,
        'tokenizer':           tokenizer,
    }


# ==================== 快速测试 ====================

if __name__ == '__main__':
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(project_root, 'data', 'all-data.csv')

    if not os.path.exists(data_file):
        print(f"[Error] not found: {data_file}")
        sys.exit(1)

    result = get_dataloaders(data_file)

    print("\n=== BiLSTM DataLoader ===")
    ids, labs = next(iter(result['train_loader_bilstm']))
    print(f"  input_ids: {ids.shape}  labels: {labs.shape}  vocab: {len(result['vocab'])}")

    print("\n=== FinBERT DataLoader ===")
    ids, att, labs = next(iter(result['train_loader_finbert']))
    print(f"  input_ids: {ids.shape}  attention_mask: {att.shape}  labels: {labs.shape}")

    print("\n=== All checks passed ===")
