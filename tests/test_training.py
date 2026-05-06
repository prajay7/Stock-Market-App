import pandas as pd

from src.training.train import CLASSIFICATION_THRESHOLDS, _best_f1_threshold, _build_models, _select_features


def test_select_features_excludes_targets_and_ids():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3),
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "f1": [1.0, 2.0, 3.0],
            "target_up_1d": [0, 1, 1],
            "target_return_1d": [0.1, 0.2, -0.1],
            "target_close_1d": [101.0, 102.0, 103.0],
            "forward_return_1d": [0.01, -0.01, 0.0],
        }
    )
    cols = _select_features(df)
    assert cols == ["f1"]


def test_training_models_use_xgboost_instead_of_random_forest():
    classifiers = _build_models("classification")
    regressors = _build_models("regression_return")

    assert "xgboost_classifier" in classifiers
    assert "xgboost_regressor" in regressors
    assert list(classifiers) == ["xgboost_classifier"]
    assert list(regressors) == ["xgboost_regressor"]
    assert "rf_classifier" not in classifiers
    assert "rf_regressor" not in regressors


def test_classification_threshold_candidates_are_fixed():
    threshold, _ = _best_f1_threshold(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0.29, 0.39, 0.41, 0.69]).to_numpy(),
    )

    assert CLASSIFICATION_THRESHOLDS == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert threshold in CLASSIFICATION_THRESHOLDS
