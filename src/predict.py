import joblib
import pandas as pd

from features import transform_features


def predict(
    description: str,
    code: str,
    difficulty: float
):
    """
    Predict tags using Logistic Regression and Linear SVC.
    """

    model_lgr = joblib.load("models/model_lgr.pkl")
    model_svc = joblib.load("models/model_svc.pkl")

    provider = joblib.load("models/provider.pkl")

    desc_vectorizer = joblib.load(
        "models/desc_vectorizer.pkl"
    )

    code_vectorizer = joblib.load(
        "models/code_vectorizer.pkl"
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
        "linear_svc": tags_svc
    }
    
if __name__ == "__main__":

    result = predict(
        description="Trouve le chemin le plus court dans un graphe.",
        code="vector<int> adj[n];",
        difficulty=3
    )

    print(result)