from app.core.config import get_settings
from src.backtesting.engine import run_backtest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=False)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--model-name", default="hist_gb_classifier")
    parser.add_argument("--mode", type=str, default="static", choices=["static", "walk_forward"])
    parser.add_argument("--retrain-every-days", type=int, default=20)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--train-lookback-days", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    symbols = args.symbols or settings.default_symbols
    result = run_backtest(
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
    print(result.metrics)
