from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import joblib
import pandas as pd

from app.core.config import get_settings
from src.backtesting.metrics import summary_metrics
from src.backtesting.plots import save_equity_plot
from src.training.dataset_builder import build_and_save_dataset
from src.training.train import _build_models


@dataclass
class BacktestResult:
    daily: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def _latest_model_path(model_dir: Path, model_name: str) -> Path:
    latest_path = model_dir / f"{model_name}_latest.joblib"
    if latest_path.exists():
        return latest_path
    candidates = sorted(model_dir.glob(f"{model_name}_*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No artifacts found for model {model_name}")
    return candidates[-1]


def _score_rows(model, features: pd.DataFrame, task_type: str) -> pd.Series:
    if task_type == "classification":
        if hasattr(model, "predict_proba"):
            return pd.Series(model.predict_proba(features)[:, 1], index=features.index)
        return pd.Series(model.predict(features), index=features.index)
    return pd.Series(model.predict(features), index=features.index)


def _resolve_model_for_training(task_type: str, model_name: str):
    models = _build_models(task_type if task_type in {"classification", "regression_return", "regression_close"} else "classification")
    if model_name in models:
        return models[model_name]
    if task_type == "classification" and "xgboost_classifier" in models:
        return models["xgboost_classifier"]
    if task_type != "classification" and "xgboost_regressor" in models:
        return models["xgboost_regressor"]
    if task_type == "classification" and "hist_gb_classifier" in models:
        return models["hist_gb_classifier"]
    if task_type != "classification" and "hist_gb_regressor" in models:
        return models["hist_gb_regressor"]
    first = next(iter(models.values()), None)
    if first is None:
        raise ValueError("No available models found for walk-forward training")
    return first


def run_backtest(
    symbols: list[str],
    start: str,
    end: str,
    horizon_days: int = 1,
    top_n: int = 5,
    model_name: str = "xgboost_classifier",
    mode: str = "static",
    retrain_every_days: int = 20,
    min_train_rows: int = 300,
    train_lookback_days: int | None = None,
) -> BacktestResult:
    settings = get_settings()
    model_path = _latest_model_path(settings.model_dir, model_name)
    metadata_path = model_path.with_suffix("").with_name(model_path.stem + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_cols = [str(c) for c in metadata.get("feature_list", [])]
    task_type = str(metadata.get("task_type") or "classification").strip().lower()
    target_col = str(metadata.get("target") or f"target_up_{horizon_days}d")
    loaded_model = joblib.load(model_path)

    dataset_path = build_and_save_dataset(settings.raw_data_dir, settings.processed_data_dir, symbols + [settings.benchmark_symbol], settings.historical_interval, horizon_days)
    df_all = pd.read_parquet(dataset_path).sort_values(["date", "symbol"]).reset_index(drop=True)

    for col in feature_cols:
        if col not in df_all.columns:
            df_all[col] = 0.0

    forward_col = f"forward_return_{horizon_days}d"
    if forward_col not in df_all.columns:
        raise ValueError(f"Required column '{forward_col}' not found in dataset")

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    df = df_all[(df_all["date"] >= start_ts) & (df_all["date"] <= end_ts)].copy()
    if df.empty:
        raise ValueError("No rows in selected backtest period")

    model_preds = []
    mode_norm = str(mode or "static").strip().lower()
    if mode_norm not in {"static", "walk_forward"}:
        raise ValueError("mode must be one of: static, walk_forward")

    if mode_norm == "static":
        for dt, group in df.groupby("date"):
            scored = group[["date", "symbol", forward_col]].copy().rename(columns={forward_col: "forward_return_1d"})
            X = group[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
            scored["score"] = _score_rows(loaded_model, X, task_type)
            model_preds.append(scored)
        retrain_events: list[dict] = []
    else:
        eval_dates = sorted(pd.to_datetime(df["date"]).dropna().unique().tolist())
        if not eval_dates:
            raise ValueError("No evaluation dates available for walk-forward backtest")

        active_model = None
        last_train_date = None
        retrain_events: list[dict] = []
        retrain_gap = max(1, int(retrain_every_days))
        min_rows = max(1, int(min_train_rows))
        lookback = int(train_lookback_days) if train_lookback_days is not None and int(train_lookback_days) > 0 else None

        for dt in eval_dates:
            dt_ts = pd.Timestamp(dt)
            need_retrain = (
                active_model is None
                or last_train_date is None
                or (dt_ts - last_train_date).days >= retrain_gap
            )

            if need_retrain:
                train_df = df_all[df_all["date"] < dt_ts].copy()
                if lookback is not None:
                    train_df = train_df[train_df["date"] >= (dt_ts - timedelta(days=lookback))]

                train_df = train_df.dropna(subset=[target_col])
                X_train = train_df[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
                y_train = train_df[target_col]
                if task_type == "classification":
                    y_train = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)
                else:
                    y_train = pd.to_numeric(y_train, errors="coerce")
                    valid = y_train.notna()
                    X_train = X_train.loc[valid]
                    y_train = y_train.loc[valid]

                if len(X_train) >= min_rows:
                    active_model = _resolve_model_for_training(task_type, model_name)
                    active_model.fit(X_train, y_train)
                    last_train_date = dt_ts
                    retrain_events.append({"trained_on_date": str(dt_ts.date()), "train_rows": int(len(X_train))})

            if active_model is None:
                active_model = loaded_model

            group = df[df["date"] == dt_ts].copy()
            if group.empty:
                continue
            scored = group[["date", "symbol", forward_col]].copy().rename(columns={forward_col: "forward_return_1d"})
            X_eval = group[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
            scored["score"] = _score_rows(active_model, X_eval, task_type)
            model_preds.append(scored)

    pred_df = pd.concat(model_preds, ignore_index=True, sort=False)

    trade_rows = []
    daily_rows = []

    for dt, group in pred_df.groupby("date"):
        benchmark_ret = group.loc[group["symbol"] == settings.benchmark_symbol, "forward_return_1d"]
        benchmark_ret = float(benchmark_ret.iloc[0]) if not benchmark_ret.empty else 0.0

        tradable = group[group["symbol"] != settings.benchmark_symbol].sort_values("score", ascending=False).head(top_n)
        if tradable.empty:
            port_ret = 0.0
        else:
            gross = float(tradable["forward_return_1d"].fillna(0.0).mean())
            tx_cost = settings.transaction_cost_bps / 10000.0
            port_ret = gross - tx_cost

            for _, row in tradable.iterrows():
                trade_rows.append(
                    {
                        "date": dt,
                        "symbol": row["symbol"],
                        "score": row["score"],
                        "forward_return_1d": row["forward_return_1d"],
                    }
                )

        daily_rows.append({"date": dt, "portfolio_return": port_ret, "benchmark_return": benchmark_ret})

    daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
    daily["portfolio_equity"] = (1 + daily["portfolio_return"]).cumprod()
    daily["benchmark_equity"] = (1 + daily["benchmark_return"].fillna(0.0)).cumprod()
    daily["buy_hold_return"] = daily["benchmark_return"].fillna(0.0)

    metrics = summary_metrics(daily["portfolio_return"], daily["benchmark_return"])
    metrics["win_rate"] = float((daily["portfolio_return"] > 0).mean()) if not daily.empty else 0.0
    strategy_cum = float(daily["portfolio_equity"].iloc[-1] - 1.0) if not daily.empty else 0.0
    buy_hold_cum = float(daily["benchmark_equity"].iloc[-1] - 1.0) if not daily.empty else 0.0
    metrics["strategy_cumulative_return"] = strategy_cum
    metrics["buy_and_hold_cumulative_return"] = buy_hold_cum
    metrics["strategy_vs_buy_and_hold"] = strategy_cum - buy_hold_cum
    metrics["backtest_mode"] = mode_norm
    if mode_norm == "walk_forward":
        metrics["walk_forward_retrain_events"] = retrain_events
        metrics["walk_forward_retrain_count"] = len(retrain_events)

    trades = pd.DataFrame(trade_rows)

    metrics_path = settings.output_dir / "backtest_metrics.json"
    trades_path = settings.output_dir / "backtest_trades.parquet"
    daily_path = settings.output_dir / "backtest_daily.parquet"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    trades.to_parquet(trades_path, index=False)
    daily.to_parquet(daily_path, index=False)
    save_equity_plot(daily, settings.output_dir / "backtest_equity_curve.png")

    return BacktestResult(daily=daily, trades=trades, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=False)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--model-name", type=str, default="xgboost_classifier")
    parser.add_argument("--mode", type=str, default="static", choices=["static", "walk_forward"])
    parser.add_argument("--retrain-every-days", type=int, default=20)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--train-lookback-days", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    symbols = args.symbols or settings.default_symbols
    res = run_backtest(
        symbols,
        args.start,
        args.end,
        args.horizon_days,
        args.top_n,
        args.model_name,
        mode=args.mode,
        retrain_every_days=args.retrain_every_days,
        min_train_rows=args.min_train_rows,
        train_lookback_days=(args.train_lookback_days if int(args.train_lookback_days) > 0 else None),
    )
    print(json.dumps(res.metrics, indent=2))


if __name__ == "__main__":
    main()
