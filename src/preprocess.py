import re
from underthesea import word_tokenize

#CLEAN 1 CÂU TEXT
def clean_text(text: str) -> str:
    """
    Input: raw Vietnamese text (string)
    Output: cleaned text (string)
    """

    if not isinstance(text, str):
        return ""

    # lowercase
    text = text.lower()

    # bỏ URL
    text = re.sub(r"http\S+|www\S+", " ", text)

    # bỏ ký tự đặc biệt, giữ chữ và khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text)

    # bỏ số (tuỳ dataset có thể giữ lại nếu cần)
    text = re.sub(r"\d+", " ", text)

    # xóa khoảng trắng dư
    text = re.sub(r"\s+", " ", text).strip()

    return text

# TOKENIZE TIẾNG VIỆT
def tokenize_vietnamese(text: str) -> str:
    if not text:
        return ""

    return word_tokenize(text, format="text")

# PIPELINE FULL PREPROCESS
def preprocess_text(text: str) -> str:
    text = clean_text(text)
    text = tokenize_vietnamese(text)
    return text

# Encoding label

def encode_labels(df):
    mapping = {
        "NEG": 0,
        "NEU": 1,
        "POS": 2,
    }
    df['label'] = df['label'].map(mapping)

# PREPROCESS DATAFRAME
def preprocess_data(df, text_column):
    df = df.copy()
    df["clean_text"] = df[text_column].apply(preprocess_text)
    encode_labels(df)
    return df