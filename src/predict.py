import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.preprocess import preprocess_text

MODEL_PATH = "models/phobert_sentiment"

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# load 1 lần
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()


def predict_sentiment(text):

    clean_text = preprocess_text(text)

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()

    return LABEL_MAP[prediction]