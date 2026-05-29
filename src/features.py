import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer


def build_features(df: pd.DataFrame):

    descriptions = df["description"].fillna("")
    codes = df["code"].fillna("")


    desc_vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_desc = desc_vectorizer.fit_transform(descriptions)

    code_vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[A-Za-z_][A-Za-z0-9_]*",
        max_features=20000
    )

    X_code = code_vectorizer.fit_transform(codes)

    X_difficulty = (
        df["difficulty"]
        .fillna(0)
        .astype(float)
        .values.reshape(-1, 1)
    )

    X = hstack([
        X_desc,
        X_code,
        X_difficulty
    ])

    return X, desc_vectorizer, code_vectorizer


def transform_features(
    df: pd.DataFrame,
    desc_vectorizer,
    code_vectorizer
):

    descriptions = df["description"].fillna("")
    codes = df["code"].fillna("")

    X_desc = desc_vectorizer.transform(descriptions)
    X_code = code_vectorizer.transform(codes)

    X_difficulty = (
        df["difficulty"]
        .fillna(0)
        .astype(float)
        .values.reshape(-1, 1)
    )

    X = hstack([
        X_desc,
        X_code,
        X_difficulty
    ])

    return X