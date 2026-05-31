import os
import pandas as pd

from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def create_train_test_split(
    df,
    output_dir="data/processed"
):
    

    os.makedirs(output_dir, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    train_path = os.path.join(
        output_dir,
        "train.csv"
    )

    test_path = os.path.join(
        output_dir,
        "test.csv"
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(
        f"Train set saved to: {train_path}"
    )

    print(
        f"Test set saved to: {test_path}"
    )

    return train_df, test_df

if __name__ == "__main__":

    df = pd.read_csv("data/processed/processed_dataset.csv")
    
    if df.empty:
        raise ValueError("Dataset vide ou non chargé")

    print("Dataset shape:", df.shape)

    create_train_test_split(df)