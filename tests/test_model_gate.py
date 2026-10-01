import time

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data import SEED, get_splits

F1_THRESHOLD = 0.88
LATENCY_LIMIT_MS = 30


@pytest.fixture(scope="module")
def data():
    return get_splits()


@pytest.fixture(scope="module")
def model(data):
    X_train, _, y_train, _ = data
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=SEED, n_jobs=1
    )
    clf.fit(X_train, y_train)
    return clf


def test_f1_gate(data):
    X_train, _, y_train, _ = data
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=SEED, n_jobs=1
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1_macro")
    assert scores.mean() >= F1_THRESHOLD, f"Val macro F1 {scores.mean():.4f} < {F1_THRESHOLD}"


def test_latency_gate(model, data):
    _, X_test, _, _ = data
    model.predict(X_test)  # warm-up
    times = []
    for _ in range(5):
        start = time.perf_counter()
        model.predict(X_test)
        times.append((time.perf_counter() - start) * 1000)
    best_ms = min(times)
    assert best_ms <= LATENCY_LIMIT_MS, f"Latency {best_ms:.1f} ms > {LATENCY_LIMIT_MS} ms"


def test_output_schema(model, data):
    _, X_test, _, _ = data
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(np.unique(preds)) <= {0, 1, 2}
