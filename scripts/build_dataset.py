from app.core.config import get_settings
from src.training.dataset_builder import build_and_save_dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", required=False)
    parser.add_argument("--horizon-days", type=int, default=1)
    args = parser.parse_args()

    settings = get_settings()
    symbols = args.symbols or settings.default_symbols
    path = build_and_save_dataset(settings.raw_data_dir, settings.processed_data_dir, symbols, settings.historical_interval, args.horizon_days)
    print(path)
