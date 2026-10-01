import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import SEED, get_splits

EXPERIMENT = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"
TRACKING_URI = "sqlite:///mlflow.db"

CONFIGS = [
    ("RandomForest", RandomForestClassifier, {"n_estimators": 50, "max_depth": 3}),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 100, "max_depth": 5}),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 200, "max_depth": None}),
    ("GradientBoosting", GradientBoostingClassifier,
     {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2}),
    ("GradientBoosting", GradientBoostingClassifier,
     {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
    ("GradientBoosting", GradientBoostingClassifier,
     {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3}),
]


def run_config(family, model_cls, params, X_train, y_train):
    model = model_cls(random_state=SEED, **params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    res = cross_validate(
        model, X_train, y_train, cv=cv,
        scoring={"f1": "f1_macro", "acc": "accuracy", "ll": "neg_log_loss"},
        return_train_score=True,
    )
    metrics = {
        "train_f1_macro": res["train_f1"].mean(),
        "val_f1_macro": res["test_f1"].mean(),
        "train_accuracy": res["train_acc"].mean(),
        "val_accuracy": res["test_acc"].mean(),
        "train_log_loss": -res["train_ll"].mean(),
        "val_log_loss": -res["test_ll"].mean(),
    }

    model.fit(X_train, y_train)
    signature = infer_signature(X_train, model.predict(X_train))

    with mlflow.start_run(run_name=f"{family}-{params}") as run:
        mlflow.set_tag("model_family", family)
        mlflow.set_tag("cv", "5-fold stratified")
        mlflow.log_param("model_family", family)
        mlflow.log_param("seed", SEED)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=X_train.head(5),
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )
        return run.info.run_id, family, params, metrics


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    X_train, X_test, y_train, y_test = get_splits()

    results = []
    for family, model_cls, params in CONFIGS:
        results.append(run_config(family, model_cls, params, X_train, y_train))

    rows = [
        {"run_id": r[0][:8], "family": r[1], "params": str(r[2]), **r[3]}
        for r in results
    ]
    table = pd.DataFrame(rows).round(4)
    table.to_csv("results.csv", index=False)
    print(table.to_string(index=False))

    best = max(results, key=lambda r: r[3]["val_f1_macro"])
    print(f"\nBest run: {best[0]} ({best[1]} {best[2]}) "
          f"val_f1_macro={best[3]['val_f1_macro']:.4f}")

    mv = mlflow.register_model(f"runs:/{best[0]}/model", MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"Registered {MODEL_NAME} v{mv.version} with alias 'champion'")


if __name__ == "__main__":
    main()
