from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from src.data.historical_loader import HistoricalLoader
from src.training.train import train_models
from src.utils.symbols import load_symbols_from_csv
from src.utils.unsupported_symbols import get_unsupported_symbols, mark_failure, mark_success


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols-csv", required=True)
    parser.add_argument("--symbol-column", default="Symbol")
    parser.add_argument("--market", default="india", choices=["us", "india"])
    parser.add_argument("--series-filter", default="EQ")
    parser.add_argument("--max-symbols", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--unsupported-path", type=str, default="")
    parser.add_argument("--unsupported-failure-threshold", type=int, default=2)
    parser.add_argument("--include-unsupported", action="store_true")
    parser.add_argument("--reset-unsupported", action="store_true")
    parser.add_argument("--reset-checkpoint", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    loader = HistoricalLoader(
        settings.raw_data_dir,
        alpha_vantage_api_key=settings.alpha_vantage_api_key,
        stooq_api_key=settings.stooq_api_key,
    )

    symbols = load_symbols_from_csv(
        args.symbols_csv,
        symbol_column=args.symbol_column,
        market=args.market,
        series_filter=args.series_filter,
        max_symbols=args.max_symbols,
    )

    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path else settings.output_dir / "checkpoints" / "train_from_csv_batches.json"
    unsupported_path = (
        Path(args.unsupported_path) if args.unsupported_path else settings.output_dir / "checkpoints" / "unsupported_symbols.json"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    unsupported_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
    if args.reset_unsupported and unsupported_path.exists():
        unsupported_path.unlink()

    checkpoint = {
        "processed_symbols": [],
        "success_symbols": [],
        "skipped_unsupported_symbols": [],
        "ingest_summary": {},
        "last_completed_batch": 0,
        "completed": False,
    }
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    processed = set(checkpoint.get("processed_symbols", []))
    success_symbols = list(checkpoint.get("success_symbols", []))
    skipped_unsupported = set(checkpoint.get("skipped_unsupported_symbols", []))
    ingest_summary = dict(checkpoint.get("ingest_summary", {}))
    last_completed_batch = int(checkpoint.get("last_completed_batch", 0))

    unsupported = get_unsupported_symbols(unsupported_path)
    if not args.include_unsupported:
        skipped_unsupported.update([s for s in symbols if s in unsupported and s not in processed])

    remaining = [s for s in symbols if s not in processed and (args.include_unsupported or s not in unsupported)]
    if not remaining:
        print("No remaining symbols to ingest; using checkpoint state.")

    for batch_no, batch in enumerate(chunked(remaining, args.batch_size), start=1):
        summary = loader.ingest(batch, interval=settings.historical_interval, lookback_days=args.lookback_days)
        ingest_summary.update(summary)
        ok = [s for s, rows in summary.items() if rows > 0]
        failed = [s for s, rows in summary.items() if rows <= 0]
        for symbol in ok:
            mark_success(unsupported_path, symbol)
        for symbol in failed:
            meta = mark_failure(
                unsupported_path,
                symbol,
                reason="No rows ingested in batch",
                failure_threshold=int(args.unsupported_failure_threshold),
            )
            if bool(meta.get("unsupported", False)):
                skipped_unsupported.add(symbol)

        success_symbols.extend(ok)
        processed.update(batch)
        last_completed_batch = batch_no

        checkpoint_payload = {
            "processed_symbols": sorted(processed),
            "success_symbols": sorted(set(success_symbols)),
            "skipped_unsupported_symbols": sorted(skipped_unsupported),
            "ingest_summary": ingest_summary,
            "last_completed_batch": last_completed_batch,
            "completed": False,
        }
        checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")
        print(f"batch={batch_no} size={len(batch)} success={len(ok)}")

    success_symbols = sorted(set(success_symbols))
    if not success_symbols:
        raise ValueError("No symbols ingested successfully. Try smaller batch-size or retry later due provider throttling.")

    results = train_models(success_symbols, horizon_days=args.horizon_days)
    checkpoint_payload = {
        "processed_symbols": sorted(processed),
        "success_symbols": success_symbols,
        "skipped_unsupported_symbols": sorted(skipped_unsupported),
        "ingest_summary": ingest_summary,
        "last_completed_batch": last_completed_batch,
        "completed": True,
    }
    checkpoint_path.write_text(json.dumps(checkpoint_payload, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "trained_symbols": len(success_symbols),
                "skipped_unsupported_symbols": sorted(skipped_unsupported),
                "unsupported_path": str(unsupported_path),
                "checkpoint_path": str(checkpoint_path),
                "results": results,
            },
            indent=2,
        )
    )
