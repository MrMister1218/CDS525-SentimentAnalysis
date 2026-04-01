"""
BiLSTM + Attention 模型
用于金融文本情感分类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    单层注意力机制
    将 Bi-LSTM 输出的隐藏状态序列映射为上下文向量
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        """
        Args:
            lstm_output: (batch_size, seq_len, hidden_dim)
        Returns:
            context_vector: (batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        scores = self.attention(lstm_output)            # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)              # (batch, seq_len, 1)
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_dim)
        return context, weights.squeeze(-1)


class BiLSTMAttention(nn.Module):
    """
    BiLSTM + Attention 文本分类模型

    Architecture:
        Embedding -> Bi-LSTM -> Attention -> FC -> Softmax
    """
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
        pretrained_embeddings=None,
        freeze_embeddings=False,
    ):
        """
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_dim: LSTM 单向隐藏层维度
            num_layers: LSTM 层数
            num_classes: 分类类别数
            dropout: Dropout 比率
            pretrained_embeddings: 预训练词向量矩阵，shape (vocab_size, embedding_dim)
            freeze_embeddings: 是否冻结 Embedding 层
        """
        super(BiLSTMAttention, self).__init__()

        # Embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        # 双向 LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention 层（输入是双向 hidden_dim * 2）
        self.attention = Attention(hidden_dim * 2)

        # 全连接分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播

        Args:
            input_ids: (batch_size, seq_len)  词索引
            attention_mask: (batch_size, seq_len)  忽略（仅接口兼容 FinBERT）

        Returns:
            logits: (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(input_ids)           # (batch, seq, emb_dim)

        # Bi-LSTM
        lstm_out, _ = self.lstm(embedded)               # (batch, seq, hidden_dim*2)

        # Attention
        context, weights = self.attention(lstm_out)    # (batch, hidden_dim*2)

        # 分类
        logits = self.classifier(context)               # (batch, num_classes)
        return logits


def create_bilstm_model(config):
    """
    BiLSTM_Attention 模型工厂函数

    Args:
        config: 配置字典
            vocab_size, embedding_dim, hidden_dim, num_layers,
            num_classes, dropout, pretrained_embeddings, freeze_embeddings

    Returns:
        BiLSTMAttention model
    """
    return BiLSTMAttention(
        vocab_size=config['vocab_size'],
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        num_classes=config.get('num_classes', 3),
        dropout=config.get('dropout', 0.3),
        pretrained_embeddings=config.get('pretrained_embeddings'),
        freeze_embeddings=config.get('freeze_embeddings', False),
    )
