
import numpy as np

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

from utils.io import (
    load_data,
    save_phobert
)

from src.preprocess import preprocess_data

MODEL_NAME = "vinai/phobert-base"


def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    return {
        "accuracy": accuracy
    }


def train():

    print("Loading dataset...")

    df = load_data("data/data.csv")

    # Preprocess
    df = preprocess_data(
        df,
        text_column="content"
    )

    # Xóa dữ liệu lỗi
    df = df.dropna(
        subset=["clean_text", "label"]
    )

    df["clean_text"] = df[
        "clean_text"
    ].astype(str)

    print(df["label"].value_counts())

    # Train/Test Split
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    print("Loading PhoBERT...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3
    )

    # HuggingFace Dataset
    train_dataset = Dataset.from_pandas(
        train_df.reset_index(drop=True)
    )

    test_dataset = Dataset.from_pandas(
        test_df.reset_index(drop=True)
    )

    def tokenize_function(examples):

        return tokenizer(
            examples["clean_text"],
            truncation=True,
            padding="max_length",
            max_length=256
        )

    print("Tokenizing...")

    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True
    )

    train_dataset = train_dataset.rename_column(
        "label",
        "labels"
    )

    test_dataset = test_dataset.rename_column(
        "label",
        "labels"
    )

    train_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )

    test_dataset.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "labels"
        ]
    )

    training_args = TrainingArguments(
        output_dir="results",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="logs",
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        logging_steps=50,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Training PhoBERT...")

    trainer.train()

    print("Evaluating...")

    predictions = trainer.predict(
        test_dataset
    )

    y_pred = np.argmax(
        predictions.predictions,
        axis=1
    )

    y_true = predictions.label_ids

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n===== Accuracy =====")
    print(f"{accuracy:.4f}")

    print("\n===== Classification Report =====")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "negative",
                "neutral",
                "positive"
            ]
        )
    )

    print("Saving model...")

    save_phobert(
        trainer.model,
        tokenizer,
        "models/phobert_sentiment"
    )

    print(
        "Model saved to models/phobert_sentiment"
    )

    print("Training completed!")