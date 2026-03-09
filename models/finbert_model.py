"""
FinBERT 模型
使用 Hugging Face Transformers 加载预训练的 FinBERT 模型
用于金融文本情感分类
"""

import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer, AutoConfig
from typing import Optional, Union


class FinBERTClassifier(nn.Module):
    """
    FinBERT 分类器
    基于预训练的 FinBERT 模型进行金融文本情感分类
    """
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        num_classes: int = 3,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        """
        初始化 FinBERT 分类器

        Args:
            model_name: Hugging Face 模型名称或本地路径
            num_classes: 分类类别数
                - 3: positive, neutral, negative
                - 2: positive, negative
            dropout: Dropout 比率
            freeze_base: 是否冻结 BERT 基础模型参数 (仅训练分类层)
        """
        super(FinBERTClassifier, self).__init__()

        # 加载预训练 FinBERT 配置
        self.config = AutoConfig.from_pretrained(model_name)

        # 加载预训练 BERT 模型用于序列分类
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            dropout=dropout
        )

        # 如果设置冻结基础模型，则冻结 BERT 参数
        if freeze_base:
            for param in self.bert.bert.parameters():
                param.requires_grad = False

        self.num_classes = num_classes
        self.model_name = model_name

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ):
        """
        前向传播

        Args:
            input_ids: 输入 token 的索引，shape 为 (batch_size, seq_len)
            attention_mask: 注意力掩码，shape 为 (batch_size, seq_len)
                1 表示真实 token，0 表示 padding
            token_type_ids: token 类型 IDs，shape 为 (batch_size, seq_len)
                用于区分两个句子（在情感分类中通常为 None）
            labels: 真实标签，shape 为 (batch_size,)
                用于计算损失（训练时需要）

        Returns:
            如果提供 labels:
                loss: 损失值 (标量)
                logits: 分类 logits，shape 为 (batch_size, num_classes)
            否则:
                logits: 分类 logits，shape 为 (batch_size, num_classes)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        return outputs

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        获取 BERT 的注意力权重（用于可视化分析）

        Args:
            input_ids: 输入 token 的索引
            attention_mask: 注意力掩码

        Returns:
            attention_weights: 注意力权重
        """
        # 获取 BERT 最后一层的隐藏状态
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 返回最后一层的注意力权重
        return outputs.attentions


class FinBERTTokenizer:
    """
    FinBERT 分词器封装
    提供便捷的文本预处理功能
    """
    def __init__(self, model_name: str = "ProsusAI/finbert", max_length: int = 128):
        """
        初始化分词器

        Args:
            model_name: Hugging Face 模型名称
            max_length: 最大序列长度
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode(
        self,
        texts: Union[str, list[str]],
        padding: bool = True,
        truncation: bool = True,
        return_tensors: str = "pt"
    ):
        """
        对文本进行编码

        Args:
            texts: 单个文本或文本列表
            padding: 是否进行 padding
            truncation: 是否进行截断
            return_tensors: 返回的张量类型 ("pt" for PyTorch)

        Returns:
            encoding: 编码后的字典，包含 input_ids, attention_mask 等
        """
        if isinstance(texts, str):
            texts = [texts]

        encoding = self.tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=self.max_length,
            return_tensors=return_tensors
        )

        return encoding

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True):
        """
        将 token IDs 解码为文本

        Args:
            token_ids: token ID 张量
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            texts: 解码后的文本
        """
        texts = self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
        return texts


def create_finbert_model(config: dict):
    """
    创建 FinBERT 分类模型的工厂函数

    Args:
        config: 包含模型配置的字典
            - model_name: 模型名称 (默认: "ProsusAI/finbert")
            - num_classes: 分类类别数 (默认: 3)
            - dropout: Dropout 比率 (默认: 0.1)
            - freeze_base: 是否冻结基础模型 (默认: False)

    Returns:
        model: FinBERT 分类器实例
    """
    model = FinBERTClassifier(
        model_name=config.get('model_name', 'ProsusAI/finbert'),
        num_classes=config.get('num_classes', 3),
        dropout=config.get('dropout', 0.1),
        freeze_base=config.get('freeze_base', False)
    )
    return model


def create_tokenizer(config: dict):
    """
    创建 FinBERT 分词器的工厂函数

    Args:
        config: 包含配置信息的字典
            - model_name: 模型名称
            - max_length: 最大序列长度

    Returns:
        tokenizer: FinBERT 分词器实例
    """
    tokenizer = FinBERTTokenizer(
        model_name=config.get('model_name', 'ProsusAI/finbert'),
        max_length=config.get('max_length', 128)
    )
    return tokenizer
