from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

import src.backtesting.engine as engine


class DummyModel:
    def __init__(self) -> None:
        self.fit_calls = 0
        self.last_fit_rows = 0

    def fit(self, X, y):
        self.fit_calls += 1
        self.last_fit_rows = len(X)
        return self

    def predict(self, X):
        return pd.Series([1.0 if i % 2 == 0 else 0.0 for i in range(len(X))], index=X.index)

    def predict_proba(self, X):
        prob_up = pd.Series([0.75 if i % 2 == 0 else 0.25 for i in range(len(X))], index=X.index)
        return pd.DataFrame({0: 1.0 - prob_up, 1: prob_up}).to_numpy()


class DummyLoadedModel(DummyModel):
    pass


def test_walk_forward_backtest_retrains_periodically(tmp_path, monkeypatch):
    settings = SimpleNamespace(
        model_dir=tmp_path / "models",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        output_dir=tmp_path / "outputs",
        benchmark_symbol="SPY",
        transaction_cost_bps=10.0,
        historical_interval="1d",
        data_provider="yahoo",
    )
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    rows = []
    for date in dates:
        for symbol, base in [("AAA", 100.0), ("BBB", 120.0), ("SPY", 300.0)]:
            close = base + float(date.day % 5)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close - 1,
                    "high": close + 1,
                    "low": close - 2,
                    "close": close,
                    "volume": 1000 + date.day,
                    "f1": float(date.day),
                    "f2": float(date.day % 3),
                    "forward_return_1d": 0.01 if symbol != "SPY" else 0.005,
                    "target_up_1d": 1 if date.day % 2 == 0 else 0,
                }
            )
    frame = pd.DataFrame(rows)

    dataset_path = tmp_path / "processed" / "dataset_1d_1d.parquet"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dataset_path, index=False)

    artifact_dir = settings.model_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "hist_gb_classifier_20260420120000.joblib"
    metadata_path = artifact_dir / "hist_gb_classifier_20260420120000.metadata.json"
    model_path.write_bytes(b"dummy")
    metadata_path.write_text(
        "{"
        '"feature_list": ["f1", "f2"], '
        '"task_type": "classification", '
        '"target": "target_up_1d"'
        "}",
        encoding="utf-8",
    )

    dummy_model = DummyLoadedModel()
    retrain_model = DummyModel()
    model_map = {
        "log_reg_classifier": retrain_model,
        "xgboost_classifier": retrain_model,
        "hist_gb_classifier": retrain_model,
    }

    monkeypatch.setattr(engine, "get_settings", lambda: settings)
    monkeypatch.setattr(engine, "_latest_model_path", lambda *_args, **_kwargs: model_path)
    monkeypatch.setattr(engine.joblib, "load", lambda *_args, **_kwargs: dummy_model)
    monkeypatch.setattr(engine, "build_and_save_dataset", lambda *_args, **_kwargs: dataset_path)
    monkeypatch.setattr(engine, "_build_models", lambda _task_type: model_map)

    result = engine.run_backtest(
        symbols=["AAA", "BBB"],
        start="2024-01-01",
        end="2024-01-14",
        horizon_days=1,
        top_n=1,
        model_name="hist_gb_classifier",
        mode="walk_forward",
        retrain_every_days=2,
        min_train_rows=4,
        train_lookback_days=7,
    )

    assert result.metrics["backtest_mode"] == "walk_forward"
    assert result.metrics["walk_forward_retrain_count"] > 0
    assert "strategy_cumulative_return" in result.metrics
    assert not result.daily.empty
