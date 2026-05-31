import os
import json
import pandas as pd

TARGET_TAGS = {
    "math",
    "graphs",
    "strings",
    "number theory",
    "trees",
    "geometry",
    "games",
    "probabilities"
}


def load_dataset(data_dir: str) -> pd.DataFrame:
    """
    Load all JSON files and build a clean DataFrame.
    """

    data = []

    for filename in os.listdir(data_dir):

        if not filename.lower().endswith(".json"):
            continue

        file_path = os.path.join(data_dir, filename)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                sample = json.load(f)

            tags = sample.get("tags", [])

            filtered_tags = [
                tag for tag in tags
                if tag in TARGET_TAGS
            ]

            if not filtered_tags:
                continue

            description = sample.get("prob_desc_description", "")
            code = sample.get("source_code", "")
            difficulty = sample.get("difficulty", 0)

            if not description and not code:
                continue

            data.append({
                "description": description,
                "code": code,
                "difficulty": difficulty,
                "tags": filtered_tags
            })

        except Exception as e:
            print(f"Erreur sur {file_path}: {e}")

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError("Dataset vide après preprocessing.")

    return df


def save_dataset(df: pd.DataFrame, output_dir: str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "processed_dataset.csv")

    df_to_save = df.copy()

    # important pour visualisation
    df_to_save["tags"] = df_to_save["tags"].apply(
        lambda x: ",".join(x)
    )

    df_to_save.to_csv(output_path, index=False)

    print(f"Dataset sauvegardé à : {output_path}")

if __name__ == "__main__":

    DATA_DIR = "code_classification_dataset"
    OUTPUT_DIR = "data/processed"

    df = load_dataset(DATA_DIR)

    print("=" * 50)
    print(f"Dataset shape: {df.shape}")
    print()

    print("First rows:")
    print(df.head())
    print()

    print("Tag distribution:")
    print(df["tags"].explode().value_counts())
    print()

    save_dataset(df, OUTPUT_DIR)