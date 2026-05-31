import joblib 
import pandas as pd 

from sklearn.metrics import (
    classification_report,
    f1_score
)

from features import transform_features 


def load_test_dataset(
    data_path="data/processed/test.csv") -> pd.DataFrame:
    
    df = pd.read_csv(data_path)
    df["tags"] = df["tags"].str.split(",")
    
    return df


def print_metrics(
    model_name,
    y_true,
    y_pred,
    provider
): 
    
    
    print("=" * 70)
    print(model_name.upper())
    print("=" * 70)
    print()
    
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=provider.classes_,
            zero_division=0
        )
    )
    
    micro_f1 = f1_score(
        y_true, 
        y_pred, 
        average='micro'
    )
    
    macro_f1 = f1_score(
        y_true, 
        y_pred, 
        average='macro'
    )

    print(f"Micro F1 Score: {micro_f1:.4f}")
    print(f"Macro F1 Score: {macro_f1:.4f}")
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=provider.classes_,
        zero_division=0,
        output_dict=True
    )
    
    report_df = pd.DataFrame(report).T 
    
    print()
    print("Tags les mieux prédits (triés par F1-score)")
    
    print(
        report_df.sort_values(
            by="f1-score", 
            ascending=False
        ).head(10)[["precision", "recall", "f1-score"]]
    )
    

def evaluate(): 
    
    print("Loading models...")
    
    model_lgr = joblib.load("models/model_lgr.pkl")
    model_svc = joblib.load("models/model_svc.pkl")
    
    provider = joblib.load("models/provider.pkl")
    
    desc_vectorizer = joblib.load(
        "models/desc_vectorizer.pkl"
    )
    
    code_vectorizer = joblib.load(
        "models/code_vectorizer.pkl"
    )
    
    print("Loading test dataset...")
    df = load_test_dataset()
    
    if df.empty:
        raise ValueError("Le dataset de test est vide.")
    
    y_true = provider.transform(df["tags"])
    
    X_test = transform_features(
        df,
        desc_vectorizer,
        code_vectorizer
    )
    
    print("Evaluating Logistic Regression...")
    y_pred_lgr = model_lgr.predict(X_test)
    
    print_metrics(
        "Logistic Regression",
        y_true,
        y_pred_lgr,
        provider
    )
    
    print("Evaluating Linear SVC...")
    y_pred_svc = model_svc.predict(X_test)
    
    print_metrics(
        "Linear SVC",
        y_true,
        y_pred_svc,
        provider
    )
  
    
if __name__ == "__main__":
    evaluate()