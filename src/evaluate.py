import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import get_splits

MODEL_URI = "models:/WineClassifier@champion"


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    model = mlflow.sklearn.load_model(MODEL_URI)
    _, X_test, _, y_test = get_splits()

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print("Test macro F1:", round(f1_score(y_test, preds, average="macro"), 4))
    print("Test accuracy:", round(accuracy_score(y_test, preds), 4))
    print("Test log loss:", round(log_loss(y_test, probs), 4))


if __name__ == "__main__":
    main()
