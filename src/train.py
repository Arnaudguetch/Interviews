import os 
import joblib
import pandas as pd 

from sklearn.preprocessing import MultiLabelBinarizer 

from features import build_features 
from model import build_model_lgr, build_model_svc 

RANDOM_STATE = 42 
TEST_SIZE = 0.2


def load_dataset(data_path: str) -> pd.DataFrame:
    """
    Load processed dataset from CSV file.
    """

    df = pd.read_csv(data_path)
    
    df["tags"] = df["tags"].str.split(",")

    return df


def prepare_labels(df): 
    
    provider = MultiLabelBinarizer()
    y = provider.fit_transform(df["tags"])
    
    return y, provider 


def save_artifacts(
    model_lgr,
    model_svc,
    provider,
    desc_vectorizer,
    code_vectorizer,
    output_dir="models"
):

    os.makedirs(output_dir, exist_ok=True)

    joblib.dump( model_lgr,
                os.path.join(output_dir, "model_lgr.pkl") 
    )
    joblib.dump(model_svc, 
                os.path.join(output_dir, "model_svc.pkl") 
    )

    joblib.dump(provider, 
                os.path.join(output_dir, "provider.pkl") 
    )
    
    joblib.dump(desc_vectorizer, 
                os.path.join(output_dir, "desc_vectorizer.pkl") 
    )
    joblib.dump(code_vectorizer, 
                os.path.join(output_dir, "code_vectorizer.pkl") 
    ) 
    
    print(f"Artifacts saved in {output_dir}")
    
    
def train():
    
    print("Loading dataset...")
    
    df = load_dataset("data/processed/train_dataset.csv") 
    
    y, provider = prepare_labels(df)
    
    X, desc_vectorizer, code_vectorizer = build_features(df)
    
    print("Training Logistic Regression...")
    model_lgr = build_model_lgr()
    model_lgr.fit(X, y)
    
    print("Training Linear SVC...")
    model_svc = build_model_svc()
    model_svc.fit(X, y)
    
    save_artifacts(
        model_lgr,
        model_svc,
        provider,
        desc_vectorizer,
        code_vectorizer
    )
    
    print("Training completed.")
    
if __name__ == "__main__": 
    train()