import os 
import joblib


from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MultiLabelBinarizer 

from preprocessing import load_dataset 
from features import build_features 
from model import build_model_lgr, build_model_svc 

RANDOM_STATE = 42 



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

    joblib.dump(model_lgr, f"{output_dir}/model_lgr.pkl")
    joblib.dump(model_svc, f"{output_dir}/model_svc.pkl")

    joblib.dump(provider, f"{output_dir}/provider.pkl")
    joblib.dump(desc_vectorizer, f"{output_dir}/desc_vectorizer.pkl")
    joblib.dump(code_vectorizer, f"{output_dir}/code_vectorizer.pkl") 
    
    
if __name__ == "__main__": 
    
    df = load_dataset("data/raw") 
    
    y, provider = prepare_labels(df)
    
    X, desc_vectorizer, code_vectorizer = build_features(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )
    
    
    model_lgr = build_model_lgr()
    model_lgr.fit(X_train, y_train)
    
    model_svc = build_model_svc()
    model_svc.fit(X_train, y_train)
    
    save_artifacts(
        model_lgr,
        model_svc,
        provider,
        desc_vectorizer,
        code_vectorizer
    )
    
    print("Training completed.")