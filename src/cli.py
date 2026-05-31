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
        help="Évalue les modèles sur le test dataset"
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
        print("=== Training Models ===")
        train()
        
    elif args.command == "evaluate":
        print("=== Evaluating Models ===")
        evaluate()
        
    elif args.command == "predict":
        print("=== Making Predictions ===")
        result = predict(
            description=args.description,
            code=args.code,
            difficulty=args.difficulty
        )
        
        print()
        print("=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        
        print()
        print(
            "Logistic Regression:",
            result["logistic_regression"]
        )

        print()
        print(
            "Linear SVC:",
            result["linear_svc"]
        )
        
        print()
        print("Confidence Scores (Logistic Regression):")

        for tag, score in result["scores"].items():
            print(
                f"{tag:<15} {score:.3f}"
            )

if __name__ == "__main__":
    main()