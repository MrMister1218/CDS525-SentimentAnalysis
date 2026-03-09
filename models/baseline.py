"""
基于 Bi-LSTM + Attention 的情感分类模型
用于金融文本 sentiment analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    注意力机制模块
    用于计算上下文向量，对序列中不同位置赋予不同权重
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        # 注意力权重矩阵: 将隐藏状态映射到注意力分数
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: Bi-LSTM 的输出，shape 为 (batch_size, seq_len, hidden_dim)
        Returns:
            context_vector: 加权后的上下文向量，shape 为 (batch_size, hidden_dim)
            attention_weights: 注意力权重，shape 为 (batch_size, seq_len)
        """
        # 计算注意力分数 (batch_size, seq_len, 1)
        attention_scores = self.attention(lstm_output)

        # 对最后一个维度进行 softmax，得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)

        # 加权求和得到上下文向量
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)

        return context_vector, attention_weights


class BiLSTMAttention(nn.Module):
    """
    Bi-LSTM + Attention 模型
    用于金融文本情感分类
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.3):
        """
        初始化模型参数

        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM 隐藏层维度
            num_classes: 分类类别数 (二分类: 2, 三分类: 3)
            dropout: Dropout 比率，防止过拟合
        """
        super(BiLSTMAttention, self).__init__()

        # 词嵌入层：将词索引转换为词向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 双向 LSTM 层
        # batch_first=True 表示输入输出的第一个维度是 batch_size
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,           # 双层 LSTM
            bidirectional=True,    # 双向 LSTM
            batch_first=True,
            dropout=dropout
        )

        # 注意力机制层
        # 注意：由于是双向 LSTM，hidden_dim 需要乘以 2
        self.attention = Attention(hidden_dim * 2)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入文本，shape 为 (batch_size, seq_len)
               其中 seq_len 是序列长度

        Returns:
            logits: 分类 logits，shape 为 (batch_size, num_classes)
        """
        # 1. 词嵌入 (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # 2. Bi-LSTM 编码 (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_dim*2)
        lstm_output, _ = self.lstm(embedded)

        # 3. 注意力机制 (batch_size, seq_len, hidden_dim*2) -> (batch_size, hidden_dim*2)
        context_vector, attention_weights = self.attention(lstm_output)

        # 4. 全连接层分类 (batch_size, hidden_dim*2) -> (batch_size, num_classes)
        logits = self.fc(context_vector)

        return logits


def create_model(config):
    """
    创建 Bi-LSTM + Attention 模型的工厂函数

    Args:
        config: 包含模型配置的字典
            - vocab_size: 词汇表大小
            - embedding_dim: 词嵌入维度
            - hidden_dim: LSTM 隐藏层维度
            - num_classes: 分类类别数
            - dropout: Dropout 比率

    Returns:
        model: 初始化后的模型
    """
    model = BiLSTMAttention(
        vocab_size=config.get('vocab_size', 50000),
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dim=config.get('hidden_dim', 128),
        num_classes=config.get('num_classes', 2),
        dropout=config.get('dropout', 0.3)
    )
    return model
