from app.core.config import get_settings
from src.data.news_loader import AlphaVantageNewsLoader


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=False)
    parser.add_argument("--limit-per-symbol", type=int, default=100)
    args = parser.parse_args()

    settings = get_settings()
    symbols = args.symbols or settings.default_symbols
    loader = AlphaVantageNewsLoader(
        raw_data_dir=settings.raw_data_dir,
        api_key=settings.alpha_vantage_api_key,
        database_url=settings.database_url,
        timeout_sec=settings.request_timeout_sec,
    )
    print(loader.ingest(symbols, limit_per_symbol=args.limit_per_symbol))
