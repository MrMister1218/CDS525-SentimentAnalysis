"""
models 包
包含 BiLSTM_Attention 和 FinBERT 情感分类模型
两个模型接口一致：forward(input_ids, attention_mask=None) -> logits
"""

from .baseline import BiLSTMAttention, create_bilstm_model
from .finbert_model import (
    FinBERTClassifier,
    FinBERTTokenizer,
    create_finbert_model,
    create_finbert_tokenizer,
)

__all__ = [
    'BiLSTMAttention',
    'create_bilstm_model',
    'FinBERTClassifier',
    'FinBERTTokenizer',
    'create_finbert_model',
    'create_finbert_tokenizer',
]
