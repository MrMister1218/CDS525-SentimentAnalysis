# CDS 525 - Financial News Sentiment Analysis

## Project Overview

This project implements a deep learning-based sentiment analysis system for financial news text. The goal is to classify financial news articles into three sentiment categories:

- **Positive**: Bullish/positive sentiment
- **Negative**: Bearish/negative sentiment
- **Neutral**: Neutral sentiment

## Dataset

The project uses `data/all-data.csv` (4846 samples from Financial Phrasebank), containing financial news headlines with sentiment labels. Labels are mapped to integers: `neutral=0, positive=1, negative=2`.

## Models

Two models are implemented with a unified `forward(input_ids, attention_mask=None) -> logits` interface:

1. **BiLSTM_Attention** (`models/baseline.py`)
   - Embedding → Bidirectional LSTM (2 layers) → Attention → FC Classifier
   - Tokenization: lowercase + punctuation removal + whitespace split

2. **FinBERT** (`models/finbert_model.py`)
   - Pretrained: `yiyanghkust/finbert-tone` (12-layer BERT fine-tuned on financial text)
   - 3-class output: Neutral / Positive / Negative
   - Tokenization: HuggingFace `BertTokenizer`

## Approach

- **Framework**: PyTorch + Hugging Face Transformers
- **Evaluation**: Accuracy, Precision, Recall, F1-Score

## Project Structure

```
.
├── data/
│   └── all-data.csv              # 4846 labeled financial news samples
├── models/
│   ├── __init__.py
│   ├── baseline.py                # BiLSTM_Attention model
│   └── finbert_model.py           # FinBERT classifier + tokenizer
├── scripts/
│   ├── preprocess.py              # Dataset, vocabulary, DataLoader builder
│   └── plot_results.py           # Training curves & HTML prediction tables
├── results/                       # Training outputs (metrics, plots, tables)
├── train.py                       # Main training script
├── requirements.txt
└── README.md
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) Preprocess data independently:
   ```bash
   python scripts/preprocess.py
   ```
   Note: `train.py` handles data loading internally; this step is only needed to inspect the dataset structure.

3. Train a model:
   ```bash
   # BiLSTM + CrossEntropyLoss
   python train.py --model baseline --loss_fn crossentropy --lr 0.001 --batch_size 32 --epochs 10

   # BiLSTM + Label Smoothing
   python train.py --model baseline --loss_fn labelsmoothing --lr 0.001 --batch_size 32

   # FinBERT
   python train.py --model finbert --loss_fn crossentropy --lr 2e-5 --batch_size 16 --epochs 5

   ```

4. Visualize results:
   ```bash
   python scripts/plot_results.py --results_dir results --output_dir results
   ```
