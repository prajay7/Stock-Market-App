from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from dashboard.dataset_creator import create_dataset
from dashboard.dataset_processor import prepare_text_dataset
from app.core.config import get_settings
from src.data.historical_loader import HistoricalLoader


def run_full_pipeline(
    symbol_limit: Optional[int] = None,
    feed_limit: int = 3,
    items_per_feed: int = 50,
    dedupe: bool = True,
    download_history: bool = False,
    lookback_days: int = 3650,
    interval: str = "1d",
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    """Run scraping -> save raw CSV -> process -> save model-ready dataset.

    Returns a dict with keys: raw (path, rows) and processed (csv_path, parquet_path, rows).
    """
    logs: list[str] = []

    def _log(msg: str) -> None:
        message = str(msg)
        logs.append(message)
        if progress_callback is not None:
            try:
                progress_callback(message)
            except Exception:
                pass

    _log("Step 1/3: Scraping public feeds and creating raw dataset")
    raw_res = create_dataset(symbol_limit=symbol_limit, feed_limit=feed_limit, items_per_feed=items_per_feed)
    raw_path = Path(raw_res.get("path")) if raw_res and raw_res.get("path") else None
    _log(f"Raw dataset saved: {raw_res.get('path')} (rows={raw_res.get('rows')})")

    # process using the raw CSV we just created
    _log("Step 2/3: Processing raw dataset into model-ready format")
    proc_res = prepare_text_dataset(input_csv=raw_path, dedupe=dedupe)
    _log(f"Processed dataset saved: {proc_res.get('csv_path')} (rows={proc_res.get('rows')})")

    historical_summary = None
    if download_history:
        _log("Step 3/3: Downloading historical price data and storing to DB")
        # load symbols from processed dataset and ingest historical prices
        settings = get_settings()
        processed_csv = Path(proc_res.get("csv_path")) if proc_res and proc_res.get("csv_path") else None
        symbols: list[str] = []
        if processed_csv and processed_csv.exists():
            import pandas as pd

            df = pd.read_csv(processed_csv)
            if "symbol" in df.columns:
                symbols = [str(s).strip().upper() for s in df["symbol"].unique() if str(s).strip()]

        if symbols:
            _log(f"Historical symbols queued: {len(symbols)}")
            loader = HistoricalLoader(
                settings.raw_data_dir,
                alpha_vantage_api_key=settings.alpha_vantage_api_key,
                polygon_api_key=settings.polygon_api_key,
                stooq_api_key=settings.stooq_api_key,
                data_provider=settings.data_provider,
            )
            historical_summary = loader.ingest(
                symbols,
                interval=interval,
                lookback_days=int(lookback_days),
                progress_callback=_log,
            )
        else:
            _log("No symbols found for historical download")

    _log("Pipeline finished")

    return {"raw": raw_res, "processed": proc_res, "historical": historical_summary, "logs": logs}


if __name__ == "__main__":
    print(run_full_pipeline())
