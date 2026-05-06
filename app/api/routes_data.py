from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.models.schemas import IngestHistoricalRequest, IngestNewsRequest
from app.services.data_service import data_service
from app.services.news_service import news_service
from src.data.db import SQLiteDataStore
from src.data.storage import read_parquet_if_exists

router = APIRouter(tags=["data"])


@router.post("/ingest/historical")
def ingest_historical(payload: IngestHistoricalRequest) -> dict:
    try:
        summary = data_service.ingest_historical(payload.symbols, payload.interval, payload.lookback_days)
        return {"ingested": summary}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/ingest/news")
def ingest_news(payload: IngestNewsRequest) -> dict:
    try:
        summary = news_service.ingest_news(payload.symbols, payload.limit_per_symbol)
        return {"ingested": summary}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/stocks/{symbol}/features")
def get_symbol_features(symbol: str) -> dict:
    settings = get_settings()
    latest = settings.processed_data_dir / f"dataset_{settings.historical_interval}_1d.parquet"
    if not latest.exists():
        raise HTTPException(status_code=404, detail="Dataset not found. Train or build dataset first.")

    import pandas as pd

    df = pd.read_parquet(latest)
    rows = df[df["symbol"] == symbol.upper()].tail(50)
    if rows.empty:
        raise HTTPException(status_code=404, detail="Symbol not found in dataset")
    return {"symbol": symbol.upper(), "features": rows.to_dict(orient="records")}


@router.get("/stocks/{symbol}/latest-prediction")
def get_latest_prediction(symbol: str) -> dict:
    settings = get_settings()
    store = SQLiteDataStore(settings.db_path)
    df = store.read_predictions(limit=1000)
    if df.empty:
        raise HTTPException(status_code=404, detail="Prediction not found for symbol")
    row = df[df["symbol"].astype(str).str.upper() == symbol.upper()]
    if row.empty:
        raise HTTPException(status_code=404, detail="Prediction not found for symbol")
    return row.iloc[0].to_dict()


@router.get("/source-health")
def source_health() -> dict:
    settings = get_settings()
    store = SQLiteDataStore(settings.db_path)
    return {"sources": store.read_source_health(limit=100)}


@router.get("/watchlist")
def watchlist() -> dict:
    settings = get_settings()
    csv_path = settings.raw_data_dir.parent / "watchlist.csv"
    if csv_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
            if not df.empty:
                candidates = ["symbol", "Symbol", "ticker", "Ticker"]
                col = next((c for c in candidates if c in df.columns), None)
                if col is not None:
                    symbols = [str(v).strip().upper() for v in df[col].dropna().tolist() if str(v).strip()]
                    if symbols:
                        return {"symbols": list(dict.fromkeys(symbols)), "source": str(csv_path)}
        except Exception:
            pass

    return {"symbols": settings.default_symbols, "source": "DEFAULT_SYMBOLS"}
