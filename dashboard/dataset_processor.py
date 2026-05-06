from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import json
import sqlite3

from app.core.config import get_settings


def _safe_text(x: Optional[object]) -> str:
    if x is None:
        return ""
    return str(x)


def find_latest_raw_dataset(raw_datasets_dir: Path) -> Path | None:
    if not raw_datasets_dir.exists():
        return None
    candidates = sorted(raw_datasets_dir.glob("dataset_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def prepare_text_dataset(
    input_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    dedupe: bool = True,
) -> dict:
    settings = get_settings()
    raw_dir = Path(settings.raw_data_dir) / "datasets"
    if input_csv is None:
        input_csv = find_latest_raw_dataset(raw_dir)
    if input_csv is None or not Path(input_csv).exists():
        raise FileNotFoundError("No raw dataset CSV found to process")

    df = pd.read_csv(input_csv)
    if df.empty:
        return {"path": str(input_csv), "rows": 0}

    # fill missing fields and normalize
    df["symbol"] = df.get("symbol", "").fillna("").astype(str).str.strip().str.upper()
    df["security_name"] = df.get("security_name", "").fillna("").astype(str).str.strip()
    df["title"] = df.get("title", "").fillna("").astype(str)
    df["description"] = df.get("description", "").fillna("").astype(str)
    df["feed_label"] = df.get("feed_label", "").fillna("").astype(str)
    df["feed_url"] = df.get("feed_url", "").fillna("").astype(str)

    # combined text column used by typical text models
    df["text"] = (df["title"].fillna("") + " \n " + df["description"].fillna("")).astype(str)

    # simple text features
    df["title_word_count"] = df["title"].apply(lambda s: len(_safe_text(s).split()))
    df["description_word_count"] = df["description"].apply(lambda s: len(_safe_text(s).split()))
    df["text_word_count"] = df["text"].apply(lambda s: len(_safe_text(s).split()))

    # published and scraped times
    if "published" in df.columns:
        df["published_at"] = pd.to_datetime(df["published"], errors="coerce")
    else:
        df["published_at"] = pd.NaT
    if "scraped_at" in df.columns:
        df["scraped_at"] = pd.to_datetime(df["scraped_at"], errors="coerce")
    else:
        df["scraped_at"] = pd.to_datetime(datetime.utcnow())

    # deduplicate identical entries (by symbol + title + link)
    if dedupe:
        key_cols = [c for c in ("symbol", "title", "link") if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep="first")

    # keep only relevant columns for modeling
    keep_cols = [c for c in ("symbol", "security_name", "text", "title", "description", "link", "feed_label", "feed_url", "published_at", "scraped_at", "text_word_count", "title_word_count", "description_word_count") if c in df.columns]
    model_df = df[keep_cols].reset_index(drop=True)

    out_dir = Path(output_dir) if output_dir is not None else Path(settings.processed_data_dir) / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out_csv = out_dir / f"dataset_model_{timestamp}.csv"
    out_parquet = out_dir / f"dataset_model_{timestamp}.parquet"
    model_df.to_csv(out_csv, index=False)
    try:
        model_df.to_parquet(out_parquet, index=False)
    except Exception:
        # parquet optional; ignore if environment lacks fastparquet/pyarrow
        out_parquet = None

    # Ensure datasets table exists and record metadata
    try:
        db_path = Path(settings.db_path or Path("./data/stock_data.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                path TEXT,
                source TEXT,
                metadata_json TEXT,
                created_at TEXT
            );
            """
        )
        metadata = {
            "rows": len(model_df),
            "columns": list(model_df.columns),
            "source_raw": str(input_csv) if input_csv is not None else None,
        }
        conn.execute(
            "INSERT INTO datasets(name, path, source, metadata_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                f"dataset_model_{timestamp}",
                str(out_csv),
                "processed",
                json.dumps(metadata, default=str),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    except Exception:
        # non-fatal: if DB write fails, still return processed paths
        pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return {"csv_path": str(out_csv), "parquet_path": str(out_parquet) if out_parquet else None, "rows": len(model_df)}


if __name__ == "__main__":
    res = prepare_text_dataset()
    print(res)
