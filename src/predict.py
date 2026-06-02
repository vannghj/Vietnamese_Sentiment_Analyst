from src.preprocess import preprocess_text
from utils.io import load_model


model = load_model("models/model.pkl")
vectorizer = load_model("models/tfidf.pkl")

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


def predict_sentiment(text):

    clean_text = preprocess_text(text)

    X = vectorizer.transform([clean_text])

    prediction = model.predict(X)[0]

    return LABEL_MAP[prediction]