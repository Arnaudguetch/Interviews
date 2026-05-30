import argparse

from train import train 
from evaluate import evaluate 
from predict import predict 


def main():
    parser = argparse.ArgumentParser(
        description="Code Classification CLI"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True
    )

    subparsers.add_parser(
        "train", 
        help="Entraîne les modèles"
    )

    subparsers.add_parser(
        "evaluate", 
        help="Évalue les modèles sur le test set"
    )

    predict_parser = subparsers.add_parser(
        "predict",
        help="Prédit les tags pour une nouvelle tâche"
    )

    predict_parser.add_argument(
        "--description",
        required=True,
        help="Description de la tâche",
    )
    predict_parser.add_argument(
        "--code",
        required=True,
        help="Code de la tâche"
    )
    predict_parser.add_argument(
        "--difficulty",
        type=int,
        default=0,
        help="Difficulté de la tâche",
    )

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "predict":
        tags = predict(
            description=args.description,
            code=args.code,
            difficulty=args.difficulty
        )
        
        print()
        print("=== Predictions ===")

        print(
            "Logistic Regression:",
            tags["logistic_regression"]
        )

        print(
            "Linear SVC:",
            tags["linear_svc"]
        )

if __name__ == "__main__":
    main()