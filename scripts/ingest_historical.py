from app.core.config import get_settings
from src.data.historical_loader import HistoricalLoader
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
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--lookback-days", type=int, default=3650)
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
    loader = HistoricalLoader(
        settings.raw_data_dir,
        alpha_vantage_api_key=settings.alpha_vantage_api_key,
        polygon_api_key=settings.polygon_api_key,
        stooq_api_key=settings.stooq_api_key,
        data_provider=settings.data_provider,
    )
    print(loader.ingest(symbols, interval=args.interval, lookback_days=args.lookback_days))
