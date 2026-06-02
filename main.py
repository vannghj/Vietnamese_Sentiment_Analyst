from src.preprocess import preprocess_data
from utils.io import load_data, save_model
from src.train import train_model

def main():
    df = load_data("data/data.csv")

    df = preprocess_data(df, text_column="content")

    print(df["label"].value_counts())
    model, vectorizer = train_model(df)

    save_model(model, "models/model.pkl")
    save_model(vectorizer, "models/tfidf.pkl")

    print("Training Complete")

if __name__ == "__main__":
    main()