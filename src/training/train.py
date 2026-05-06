from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from typing import Literal
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import TimeSeriesSplit

from app.core.config import get_settings
from src.data.db import SQLiteDataStore
from src.data.cache import symbol_price_path
from src.data.historical_loader import HistoricalLoader
from src.data.storage import read_parquet_if_exists
from src.training.dataset_builder import build_and_save_dataset
from src.training.evaluate import classification_metrics, regression_metrics
from src.training.model_registry import ModelRegistry
from src.features.targets import add_movement_target
from src.utils.symbols import normalize_symbols


TaskType = Literal["classification", "regression_return", "regression_close", "movement"]
CLASSIFICATION_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def _build_models(task_type: TaskType) -> dict[str, object]:
    if task_type in {"classification", "movement"}:
        try:
            from xgboost import XGBClassifier
        except Exception as exc:
            raise RuntimeError("XGBoost is unavailable. Install xgboost and its OpenMP runtime before training.") from exc

        models = {
            "xgboost_classifier": XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.04,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
            ),
        }
        return models

    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError("XGBoost is unavailable. Install xgboost and its OpenMP runtime before training.") from exc

    models = {
        "xgboost_regressor": XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="rmse",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        ),
    }
    return models


def _recommended_n_splits(row_count: int) -> int:
    # Favor faster training for small/medium runs while keeping enough temporal validation.
    if row_count < 300:
        return 2
    if row_count < 1500:
        return 3
    if row_count < 5000:
        return 4
    return 5


def _is_symbol_updated_today(raw_data_dir, symbol: str, interval: str) -> bool:
    settings = get_settings()
    store = SQLiteDataStore(settings.db_path)
    latest = store.latest_candle_time(symbol, interval)
    if latest:
        latest_ts = pd.to_datetime(latest, errors="coerce")
        if pd.notna(latest_ts):
            return bool(latest_ts.date() >= date.today())

    path = symbol_price_path(raw_data_dir, symbol, interval)
    existing = read_parquet_if_exists(path)
    if existing.empty:
        return False

    cols = {str(c).strip().lower(): c for c in existing.columns}
    date_col = cols.get("date") or cols.get("datetime")
    if date_col is None:
        return False

    last_ts = pd.to_datetime(existing[date_col], errors="coerce").dropna()
    if last_ts.empty:
        return False
    return bool(last_ts.max().date() >= date.today())


def _select_features(df: pd.DataFrame) -> list[str]:
    def _is_target_col(col: str) -> bool:
        return (
            col.startswith("target_up_")
            or col.startswith("target_return_")
            or col.startswith("target_close_")
            or col.startswith("forward_return_")
        )

    exclude = {"date", "symbol"}
    return [c for c in df.columns if c not in exclude and not _is_target_col(c) and pd.api.types.is_numeric_dtype(df[c])]


def _task_target_col(task_type: TaskType, horizon_days: int) -> str:
    if task_type == "classification":
        return f"target_up_{horizon_days}d"
    if task_type == "movement":
        return f"target_movement_{horizon_days}n"
    if task_type == "regression_return":
        return f"target_return_{horizon_days}d"
    return f"target_close_{horizon_days}d"


def _score_key(task_type: TaskType) -> tuple[str, bool]:
    # Returns (metric_name, higher_is_better)
    if task_type in {"classification", "movement"}:
        return "f1", True
    return "rmse", False


def _scale_pos_weight(y: pd.Series) -> float:
    counts = y.value_counts()
    positives = float(counts.get(1, 0))
    negatives = float(counts.get(0, 0))
    if positives <= 0 or negatives <= 0:
        return 1.0
    return max(0.25, min(4.0, negatives / positives))


def _best_f1_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, float]:
    if len(pd.Series(y_true).dropna().unique()) < 2:
        return 0.5, 0.0
    best_threshold = 0.5
    best_score = -1.0
    for threshold in CLASSIFICATION_THRESHOLDS:
        score = float(f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def _extract_feature_importance(model, feature_cols: list[str]) -> list[dict]:
    scores: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        raw = getattr(model, "feature_importances_", None)
        if raw is not None:
            scores = np.asarray(raw, dtype=float)
    elif hasattr(model, "coef_"):
        raw = getattr(model, "coef_", None)
        if raw is not None:
            arr = np.asarray(raw, dtype=float)
            if arr.ndim == 2:
                arr = arr[0]
            scores = np.abs(arr)

    if scores is None or len(scores) != len(feature_cols):
        return []

    frame = pd.DataFrame({"feature": feature_cols, "importance": scores})
    frame["importance"] = pd.to_numeric(frame["importance"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    return frame.to_dict(orient="records")


def _write_feature_importance_artifact(
    model_dir: Path,
    model_name: str,
    version: str,
    feature_importance: list[dict],
    keep_versioned: bool = False,
) -> Path:
    payload = {
        "model_name": model_name,
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_importance": feature_importance,
    }
    latest_path = model_dir / f"{model_name}_latest.feature_importance.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if keep_versioned:
        versioned_path = model_dir / f"{model_name}_{version}.feature_importance.json"
        versioned_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return latest_path


def train_models(
    symbols: list[str],
    horizon_days: int = 1,
    refresh_prices: bool = True,
    task_type: TaskType = "classification",
) -> dict:
    settings = get_settings()
    symbols = normalize_symbols(symbols)
    ingest_summary: dict[str, int] = {}
    fresh_symbols: list[str] = []
    if refresh_prices:
        stale_symbols = [
            symbol
            for symbol in symbols
            if not _is_symbol_updated_today(settings.raw_data_dir, symbol, settings.historical_interval)
        ]
        if stale_symbols:
            loader = HistoricalLoader(
                settings.raw_data_dir,
                alpha_vantage_api_key=settings.alpha_vantage_api_key,
                polygon_api_key=settings.polygon_api_key,
                stooq_api_key=settings.stooq_api_key,
                data_provider=settings.data_provider,
            )
            ingest_summary = loader.ingest(
                stale_symbols,
                interval=settings.historical_interval,
                lookback_days=settings.historical_lookback_days,
            )
        stale_set = set(stale_symbols)
        fresh_symbols = [symbol for symbol in symbols if symbol not in stale_set]
        for symbol in fresh_symbols:
            ingest_summary[symbol] = 0

    dataset_path = build_and_save_dataset(settings.raw_data_dir, settings.processed_data_dir, symbols, settings.historical_interval, horizon_days)
    df = pd.read_parquet(dataset_path).sort_values(["date", "symbol"]).reset_index(drop=True)

    if task_type == "movement":
        df = add_movement_target(df, horizon_days=horizon_days)

    target_col = _task_target_col(task_type, horizon_days)
    df = df.dropna(subset=[target_col]).copy()

    feature_cols = _select_features(df)
    if not feature_cols:
        raise ValueError("No usable numeric feature columns found after target preparation")
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    y = df[target_col].astype(int) if task_type == "classification" else pd.to_numeric(df[target_col], errors="coerce")
    if task_type != "classification":
        mask = y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)

    if len(X) <= 2:
        raise ValueError("Not enough rows for time-series validation (need at least 3 rows)")
    n_splits = min(_recommended_n_splits(len(X)), len(X) - 1)
    if len(X) <= n_splits:
        raise ValueError(f"Not enough rows ({len(X)}) for time-series validation with {n_splits} splits")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    registry = ModelRegistry(settings.model_dir, keep_last_versions=int(settings.model_keep_last_versions))

    results: dict[str, dict] = {}
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    score_metric, higher_is_better = _score_key(task_type)
    best_model_name = ""
    best_score: float | None = None

    for name, model in _build_models(task_type).items():
        fold_metrics = []
        fold_thresholds: list[float] = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = clone(model)
            if task_type == "classification":
                fold_model.set_params(scale_pos_weight=_scale_pos_weight(y_train))
            fold_model.fit(X_train, y_train)

            if task_type == "classification":
                y_prob = fold_model.predict_proba(X_val)[:, 1] if hasattr(fold_model, "predict_proba") else fold_model.predict(X_val)
                threshold, _ = _best_f1_threshold(y_val, y_prob)
                fold_thresholds.append(threshold)
                y_pred = (y_prob >= threshold).astype(int)
                fold_metrics.append(classification_metrics(y_val, y_pred, y_prob))
            else:
                y_pred = fold_model.predict(X_val)
                fold_metrics.append(regression_metrics(y_val, y_pred))

        mean_metrics = {k: float(np.mean([fm[k] for fm in fold_metrics if k in fm])) for k in fold_metrics[0].keys()}
        classification_threshold = float(np.mean(fold_thresholds)) if fold_thresholds else 0.5
        if task_type == "classification":
            mean_metrics["classification_threshold"] = classification_threshold

        current_score = float(mean_metrics.get(score_metric, 0.0))
        if best_score is None:
            best_score = current_score
            best_model_name = name
        else:
            if (higher_is_better and current_score > best_score) or ((not higher_is_better) and current_score < best_score):
                best_score = current_score
                best_model_name = name

        final_model = clone(model)
        if task_type == "classification":
            final_model.set_params(scale_pos_weight=_scale_pos_weight(y))
        final_model.fit(X, y)
        model_path = registry.save_model(final_model, name, version)
        feature_importance = _extract_feature_importance(final_model, feature_cols)
        feature_importance_path = _write_feature_importance_artifact(
            settings.model_dir,
            name,
            version,
            feature_importance,
            keep_versioned=registry.keep_last_versions > 0,
        )

        metadata = {
            "symbols": symbols,
            "date_range": [str(df["date"].min()), str(df["date"].max())],
            "feature_list": feature_cols,
            "feature_importance_top": feature_importance[:20],
            "metrics": mean_metrics,
            "classification_threshold": classification_threshold if task_type == "classification" else None,
            "target": target_col,
            "task_type": task_type,
            "horizon_days": int(horizon_days),
            "dataset_path": str(dataset_path),
            "pretrain_ingest_summary": ingest_summary,
            "pretrain_refresh_skipped_symbols": fresh_symbols if refresh_prices else [],
        }
        metadata_path = registry.save_metadata(name, version, metadata)
        deleted_artifacts = registry.prune_old_versions(name)
        results[name] = {
            "metrics": mean_metrics,
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
            "feature_importance_path": str(feature_importance_path),
            "feature_importance_top": feature_importance[:20],
            "pruned_artifacts": [str(path) for path in deleted_artifacts],
        }

    best_payload = {
        "version": version,
        "task_type": task_type,
        "horizon_days": int(horizon_days),
        "score_metric": score_metric,
        "best_model_name": best_model_name,
        "best_score": best_score,
    }
    if task_type == "classification" and best_model_name in results:
        best_payload["classification_threshold"] = results[best_model_name]["metrics"].get("classification_threshold")
    best_path = settings.model_dir / f"best_model_{task_type}.json"
    best_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    leaderboard_path = settings.output_dir / f"training_results_{version}.json"
    leaderboard_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if task_type == "movement":
        movement_payload = {
            "model_version": version,
            "task_type": task_type,
            "horizon_days": int(horizon_days),
            "symbols": symbols,
            "target": target_col,
            "metrics": results.get(best_model_name, {}).get("metrics", {}),
            "classification_threshold": results.get(best_model_name, {}).get("metrics", {}).get("classification_threshold"),
            "artifact_path": str(settings.movement_model_path),
            "model_type": best_model_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
        }
        settings.movement_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final_model, settings.movement_model_path)
        settings.movement_model_path.with_suffix(".metadata.json").write_text(
            json.dumps(movement_payload, indent=2),
            encoding="utf-8",
        )
        SQLiteDataStore(settings.db_path).write_training_run(movement_payload)
    return {
        "task_type": task_type,
        "target": target_col,
        "version": version,
        "best": best_payload,
        "models": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=False)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument(
        "--task-type",
        type=str,
        default="classification",
        choices=["classification", "regression_return", "regression_close"],
    )
    args = parser.parse_args()

    settings = get_settings()
    symbols = args.symbols or settings.default_symbols
    results = train_models(symbols=symbols, horizon_days=args.horizon_days, task_type=args.task_type)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
