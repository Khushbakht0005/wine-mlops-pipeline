# Wine MLOps Pipeline

![CI](https://github.com/Khushbakht0005/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

MLOps pipeline for wine cultivar classification (RandomForest vs GradientBoosting)
with MLflow tracking, model registry and a GitHub Actions quality gate.

## Usage

    make install   # install dependencies
    make lint      # flake8
    make test      # pytest (includes model quality gate)
    make train     # train, log to MLflow, register champion
    python -m src.evaluate

## Quality gate

- Validation macro F1 >= 0.88
- Batch inference latency <= 30 ms
- Predictions only in {0, 1, 2}

Random seed is 42 everywhere.
