# utils/io.py

import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Đọc file CSV
def load_data(path):
    return pd.read_csv(path)

# Lưu DataFrame ra CSV
def save_data(df, path):
    df.to_csv(path, index=False)

# Lưu model PhoBERT và tokenizer
def save_phobert(model, tokenizer, path):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


 # Load model PhoBERT đã fine-tune
def load_phobert(path):
    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForSequenceClassification.from_pretrained(
        path
    )
    return model, tokenizer