import joblib 

from sklearn.metrics import (
    classification_report,
    f1_score
)

from preprocessing import load_dataset
from features import transform_features 


def evaluate(): 
    
    model_lgr = joblib.load("models/model_lgr.pkl")
    model_svc = joblib.load("models/model_svc.pkl")
    
    provider = joblib.load("models/provider.pkl")
    
    desc_vectorizer = joblib.load(
        "models/desc_vectorizer.pkl"
    )
    
    code_vectorizer = joblib.load(
        "models/code_vectorizer.pkl"
    )
    
    df = load_dataset("data/test")
    
    if df.empty:
        raise ValueError("Le dataset de test est vide.")
    
    y_true = provider.transform(df["tags"])
    
    X = transform_features(
        df,
        desc_vectorizer,
        code_vectorizer
    )
    
    y_pred_lgr = model_lgr.predict(X)
    
    print("=" * 70)
    print("LOGISTIC REGRESSION")
    print("=" * 70)
    print()
    
    print(
        classification_report(
            y_true,
            y_pred_lgr,
            target_names=provider.classes_,
            zero_division=0
        )
    )
    
    print(
        f"Micro F1 Score: "
        f"{f1_score(y_true, y_pred_lgr, average='micro'):.4f}"
    )
    
    print(f"Macro F1 Score: "
          f"{f1_score(y_true, y_pred_lgr, average='macro'):.4f}"
    )
    
    
    y_pred_svc = model_svc.predict(X)
    
    print()
    print("=" * 70)
    print("LINEAR SVC")
    print("=" * 70)
    
    print(
        classification_report(
            y_true,
            y_pred_svc,
            target_names=provider.classes_,
            zero_division=0
        )
    )
    
    print(
        f"Micro F1 Score: "
        f"{f1_score(y_true, y_pred_svc, average='micro'):.4f}"
    )
    
    print(f"Macro F1 Score: "
          f"{f1_score(y_true, y_pred_svc, average='macro'):.4f}"
    )
    
if __name__ == "__main__":
    evaluate()