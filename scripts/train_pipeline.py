from __future__ import annotations

import argparse
import json

from src.training.pipeline import run_model_train_pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run end-to-end model training pipeline")
    parser.add_argument("--symbols", nargs="+", required=False)
    parser.add_argument("--symbols-csv", type=str, required=False)
    parser.add_argument("--symbol-column", type=str, required=False)
    parser.add_argument("--market", type=str, default="us", choices=["us", "india"])
    parser.add_argument("--series-filter", type=str, required=False, help="CSV Series filter, e.g. EQ or EQ,SM")
    parser.add_argument("--max-symbols", type=int, required=False)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument(
        "--task-type",
        type=str,
        default="classification",
        choices=["classification", "regression_return", "regression_close", "movement"],
    )
    parser.add_argument("--ingest-first", action="store_true", help="Ingest historical candles for all symbols before training")
    parser.add_argument("--lookback-days", type=int, required=False)
    parser.add_argument("--interval", type=str, required=False)

    args = parser.parse_args()
    result = run_model_train_pipeline(
        symbols=args.symbols,
        symbols_csv=args.symbols_csv,
        symbol_column=args.symbol_column,
        market=args.market,
        series_filter=args.series_filter,
        max_symbols=args.max_symbols,
        horizon_days=args.horizon_days,
        task_type=args.task_type,
        ingest_first=bool(args.ingest_first),
        lookback_days=args.lookback_days,
        interval=args.interval,
    )
    print(json.dumps(result, indent=2))