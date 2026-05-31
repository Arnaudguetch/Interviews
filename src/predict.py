import joblib
import pandas as pd

from features import transform_features

from config import (
    MODEL_LGR_PATH,
    MODEL_SVC_PATH,
    MLB_PATH,
    DESC_VECTORIZER_PATH,
    CODE_VECTORIZER_PATH
)


def predict(
    description: str,
    code: str,
    difficulty: float
):
    """
    Predict tags using Logistic Regression and Linear SVC.
    """

    model_lgr = joblib.load(MODEL_LGR_PATH)
    model_svc = joblib.load(MODEL_SVC_PATH)

    provider = joblib.load(MLB_PATH)

    desc_vectorizer = joblib.load(
        DESC_VECTORIZER_PATH
    )

    code_vectorizer = joblib.load(
        CODE_VECTORIZER_PATH
    )

    df = pd.DataFrame([
        {
            "description": description or "",
            "code": code or "",
            "difficulty": difficulty
        }
    ])

    X = transform_features(
        df,
        desc_vectorizer,
        code_vectorizer
    )
    
    probas = model_lgr.predict_proba(X)[0]
    
    scores = {
        tag: round(score, 3)
        for tag, score in zip(
            provider.classes_,
            probas
        )
    }

    y_pred_lgr = model_lgr.predict(X)
    y_pred_svc = model_svc.predict(X)

    tags_lgr = list(
        provider.inverse_transform(y_pred_lgr)[0]
    )

    tags_svc = list(
        provider.inverse_transform(y_pred_svc)[0]
    )

    return {
        "logistic_regression": tags_lgr,
        "linear_svc": tags_svc,
        "scores": scores
    }
    
    
if __name__ == "__main__":

    result = predict(
        description="Trouve le chemin le plus court dans un graphe.",
        code="vector<int> adj[n];",
        difficulty=1200
    )

    print(result)