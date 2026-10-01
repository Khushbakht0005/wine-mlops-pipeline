import pytest

from src.data import load_data, validate, get_splits


def test_validation_passes():
    X, _ = load_data()
    validate(X)


def test_feature_count():
    X, _ = load_data()
    assert X.shape[1] == 13


def test_no_nulls():
    X, _ = load_data()
    assert not X.isnull().values.any()


def test_split_sizes_and_stratification():
    X_train, X_test, y_train, y_test = get_splits()
    assert len(X_train) == 142
    assert len(X_test) == 36
    assert set(y_train.unique()) == {0, 1, 2}
    assert set(y_test.unique()) == {0, 1, 2}


def test_validate_catches_nulls():
    X, _ = load_data()
    X.iloc[0, 0] = None
    with pytest.raises(ValueError):
        validate(X)
