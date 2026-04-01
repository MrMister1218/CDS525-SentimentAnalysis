"""
FinBERT 模型
封装 Hugging Face 预训练 FinBERT 用于金融文本情感分类
支持 yiyanghkust/finbert-tone（neutral/positive/negative 三分类）
"""

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, AutoConfig


class FinBERTClassifier(nn.Module):
    """
    FinBERT 情感分类器
    封装 BertForSequenceClassification，统一返回 logits
    """
    def __init__(
        self,
        model_name: str = "yiyanghkust/finbert-tone",
        num_classes: int = 3,
        dropout: float = 0.1,
        freeze_base: bool = False,
    ):
        """
        Args:
            model_name: Hugging Face 模型名称
            num_classes: 分类类别数（FinBERT-tone 固定为 3）
            dropout: Dropout 比率
            freeze_base: 是否冻结 BERT backbone
        """
        super(FinBERTClassifier, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            use_safetensors=True,
        )
        if freeze_base:
            for param in self.bert.bert.parameters():
                param.requires_grad = False
        self.num_classes = num_classes
        self.model_name = model_name

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len)
            labels: (batch_size,)  仅训练时计算损失
        Returns:
            logits: (batch_size, num_classes)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return outputs.logits


class FinBERTTokenizer:
    """
    FinBERT 分词器封装（基于 transformers.BertTokenizer）
    """
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone", max_length: int = 128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def __call__(self, text, padding=True, truncation=True, return_tensors='pt', **kwargs):
        return self.tokenizer(
            text,
            max_length=kwargs.pop('max_length', self.max_length),
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            **kwargs,
        )

    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)

    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def create_finbert_model(config: dict):
    """FinBERT 模型工厂函数"""
    return FinBERTClassifier(
        model_name=config.get('model_name', 'yiyanghkust/finbert-tone'),
        num_classes=config.get('num_classes', 3),
        dropout=config.get('dropout', 0.1),
        freeze_base=config.get('freeze_base', False),
    )


def create_finbert_tokenizer(config: dict):
    """FinBERT 分词器工厂函数"""
    return FinBERTTokenizer(
        model_name=config.get('model_name', 'yiyanghkust/finbert-tone'),
        max_length=config.get('max_length', 128),
    )
