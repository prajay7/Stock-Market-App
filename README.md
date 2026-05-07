# Stock AI Platform

End-to-end, beginner-friendly stock ML project with FastAPI APIs, modular training/inference pipelines, backtesting, and optional news sentiment.

## Disclaimer
This project is for education and research only. It is not financial advice.

## What Works Today
- Historical OHLCV ingestion with retry/fallback (`yfinance` primary, optional Polygon/Alpha Vantage/Stooq/NSE fallback paths).
- Technical + sentiment feature pipeline with time-safe target construction.
- Model training with time-series validation (no random shuffle).
- Optional LightGBM model support (enabled automatically if `lightgbm` is installed).
- Prediction pipeline using saved model metadata + aligned feature columns.
- API endpoints for health, ingestion, training, prediction, backtest, model info.
- Per-model feature-importance artifacts and API retrieval endpoint.
- Walk-forward backtesting mode with periodic retraining windows.
- CSV/JSON/Parquet prediction outputs.
- Streamlit dashboard for monitoring and exploration.

## Project Structure
```text
app/
  api/                # FastAPI routes
  core/               # settings + logging
  services/           # orchestration services
  models/             # API schemas
  news/               # news impact modules
src/
  data/               # data loaders + storage helpers
  features/           # technical/sentiment/target engineering
  training/           # model training + evaluation + registry
  inference/          # model inference + ranking
  backtesting/        # strategy simulation + metrics
  utils/              # symbols/validation/date helpers
scripts/              # command-line entry points
dashboard/            # Streamlit app
tests/                # unit/API tests
```

## Setup
### 1) Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure environment
```bash
cp .env.example .env
```

Recommended key settings in `.env`:
- `DATA_PROVIDER=yahoo`
- `NEWS_PROVIDER=google`
- `DEFAULT_SYMBOLS=RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN`
- Optional keys: `POLYGON_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `OPENAI_API_KEY`

## Run Locally
### Start API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Start Dashboard
```bash
streamlit run dashboard/streamlit_app.py
```

## Deploy FastAPI Live (GitHub)
GitHub itself cannot host a running FastAPI backend, but this repo is ready for live deploy via Render from GitHub.

1. Push code to GitHub (includes `render.yaml`).
2. In Render, create a new Blueprint service from this repo.
3. Set required environment variables (API keys, providers, symbols).
4. Deploy and use health endpoint:
```bash
GET /health
```

## Continuous Training (GitHub Actions Cron)
This repo includes a scheduled workflow at `.github/workflows/continuous-model-training.yml`.

- Schedule: every 6 hours (UTC)
- Manual trigger: **Actions -> Continuous Model Training -> Run workflow**
- Behavior: refresh historical data and run `scripts/train_pipeline.py --ingest-first`, then upload model/output artifacts

This repo also includes fast refresh + prediction at `.github/workflows/market-refresh-predict.yml`.

- Schedule: every 10 minutes (UTC)
- Manual trigger: **Actions -> Market Refresh And Predict -> Run workflow**
- Behavior: refresh news + historical data, run prediction, and upload latest prediction artifacts

Recommended repository variables (Settings -> Secrets and variables -> Actions -> Variables):
- `DEFAULT_SYMBOLS`
- `DATA_PROVIDER`
- `NEWS_PROVIDER`
- `HISTORICAL_INTERVAL`
- `HISTORICAL_LOOKBACK_DAYS`
- `MAX_NEWS_ITEMS_PER_SYMBOL` (optional)

Optional repository secrets:
- `ALPHA_VANTAGE_API_KEY`
- `POLYGON_API_KEY`
- `OPENAI_API_KEY`

## End-to-End Commands
### Ingest historical data
```bash
python scripts/ingest_historical.py --symbols RELIANCE TCS INFY --interval 1d --lookback-days 3650
```

### Train models (classification, default)
```bash
python scripts/train_models.py --symbols RELIANCE TCS INFY --task-type classification --horizon-days 1
```

### Run model training pipeline (single command)
```bash
python scripts/train_pipeline.py --symbols RELIANCE TCS INFY --task-type classification --horizon-days 1 --ingest-first
```

You can also load symbols from CSV:
```bash
python scripts/train_pipeline.py --symbols-csv sec_list.csv --symbol-column Symbol --market india --series-filter EQ --max-symbols 100 --task-type movement
```

If `lightgbm` is available in your environment, LightGBM models are included automatically in both classification and regression training runs.

### Train models (regression on next-day return)
```bash
python scripts/train_models.py --symbols RELIANCE TCS INFY --task-type regression_return --horizon-days 1
```

### Train models (regression on next-day close)
```bash
python scripts/train_models.py --symbols RELIANCE TCS INFY --task-type regression_close --horizon-days 1
```

### Generate predictions
```bash
python scripts/generate_predictions.py --symbols RELIANCE TCS INFY --model-name hist_gb_classifier --horizon-days 1
```

### Generate predictions from watchlist CSV
```bash
python scripts/generate_predictions.py --watchlist-csv watchlist.csv --symbol-column Symbol --market india --model-name hist_gb_classifier
```

### Run backtest
```bash
python scripts/run_backtest.py --symbols RELIANCE TCS INFY --start 2023-01-01 --end 2026-01-01 --top-n 3 --model-name hist_gb_classifier
```

### Run walk-forward backtest (periodic retraining)
```bash
python scripts/run_backtest.py \
  --symbols RELIANCE TCS INFY \
  --start 2023-01-01 \
  --end 2026-01-01 \
  --model-name hist_gb_classifier \
  --mode walk_forward \
  --retrain-every-days 20 \
  --min-train-rows 300 \
  --train-lookback-days 1000
```

## Prediction Outputs
After prediction, outputs are written to:
- `data/outputs/latest_predictions.parquet`
- `data/outputs/latest_predictions.csv`
- `data/outputs/latest_predictions.json`

Each row includes:
- `symbol`
- `current_price`
- `predicted_price`
- `prob_up`
- `predicted_return`
- `confidence`
- `decision`
- timestamp fields (`price_as_of`, etc.)

## API Endpoints
- `GET /health`
- `POST /ingest/historical`
- `POST /ingest/news`
- `POST /train`
- `POST /predict`
- `GET /models`
- `GET /model-info?model_name=hist_gb_classifier`
- `GET /model-feature-importance?model_name=hist_gb_classifier&top_k=30`
- `GET /watchlist`
- `GET /backtest` (supports `mode=static|walk_forward` and retraining query params)
- `GET /stocks/{symbol}/features`
- `GET /stocks/{symbol}/latest-prediction`

API error responses use a structured envelope:
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {},
    "trace_id": "...",
    "timestamp": "..."
  }
}
```

## API Examples
### Train (classification)
```bash
curl -X POST http://127.0.0.1:8000/train \
  -H "Content-Type: application/json" \
  -d '{"symbols":["RELIANCE","TCS","INFY"],"horizon_days":1,"task_type":"classification"}'
```

### Predict
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"symbols":["RELIANCE","TCS"],"model_name":"hist_gb_classifier","horizon_days":1}'
```

## Testing
```bash
pytest -q
```

## Known Limitations
- Delisted/illiquid symbols can still return sparse or empty datasets.
- Real-time quotes depend on provider availability and key limits.
- News sentiment is optional and can be disabled without breaking ML flow.
- Backtest strategy is intentionally simple for learning purposes.

## Suggested Next Improvements
1. Add LightGBM model option behind optional dependency.
2. Add walk-forward retraining scheduler with model drift checks.
3. Add richer position sizing/risk rules in backtesting.
4. Add stricter schema validation for user-provided watchlist CSVs.
