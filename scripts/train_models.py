from app.core.config import get_settings
from src.training.train import train_models
from src.utils.symbols import load_symbols_from_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        choices=["classification", "regression_return", "regression_close"],
    )
    args = parser.parse_args()

    settings = get_settings()
    if args.symbols_csv:
        symbols = load_symbols_from_csv(
            args.symbols_csv,
            symbol_column=args.symbol_column,
            market=args.market,
            series_filter=args.series_filter,
            max_symbols=args.max_symbols,
        )
    else:
        symbols = args.symbols or settings.default_symbols
    print(train_models(symbols, args.horizon_days, task_type=args.task_type))
