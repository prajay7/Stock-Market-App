from app.core.config import get_settings
from src.inference.predict import predict_for_symbols
from src.utils.symbols import load_symbols_from_csv


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=False)
    parser.add_argument("--watchlist-csv", type=str, required=False)
    parser.add_argument("--symbol-column", type=str, required=False)
    parser.add_argument("--market", type=str, default="us", choices=["us", "india"])
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--model-name", default="hist_gb_classifier")
    parser.add_argument("--include-live-quote", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    if args.watchlist_csv:
        symbols = load_symbols_from_csv(
            args.watchlist_csv,
            symbol_column=args.symbol_column,
            market=args.market,
        )
    else:
        symbols = args.symbols or settings.default_symbols

    if not symbols:
        raise ValueError("No symbols provided. Use --symbols or --watchlist-csv")

    print(
        predict_for_symbols(
            symbols,
            args.model_name,
            args.horizon_days,
            include_live_quote=bool(args.include_live_quote),
        ).to_json(orient="records", indent=2)
    )
