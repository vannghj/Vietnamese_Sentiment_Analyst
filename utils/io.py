import pandas as pd
import joblib


def load_data(path):
    """
    Đọc file csv
    """
    return pd.read_csv(path)


def save_data(df, path):
    """
    Lưu DataFrame ra csv
    """
    df.to_csv(path, index=False)


def save_model(model, path):
    """
    Lưu model
    """
    joblib.dump(model, path)


def load_model(path):
    """
    Load model đã train
    """
    return joblib.load(path)