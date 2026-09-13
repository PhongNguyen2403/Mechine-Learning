import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.modeling import DATASET_PATH, load_dataset, train_and_evaluate


def main() -> None:
    features, labels, feature_names = load_dataset(DATASET_PATH)
    metrics = train_and_evaluate(features, labels, feature_names)
    print("SVM training complete")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"F1 macro: {metrics['f1_macro']:.2%}")
    print(f"Best parameters: {metrics['best_params']}")
    print("Saved model to artifacts/svm_model.joblib")


if __name__ == "__main__":
    main()