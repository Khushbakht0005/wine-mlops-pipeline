from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

SEED = 42
N_FEATURES = 13


def load_data():
    X, y = load_wine(return_X_y=True, as_frame=True)
    return X, y


def validate(X):
    if X.isnull().values.any():
        raise ValueError("Dataset contains null values")
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, got {X.shape[1]}")


def get_splits():
    X, y = load_data()
    validate(X)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
