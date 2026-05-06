# Dynamic Trending Symbols for Predictions

## Overview
The system now automatically fetches trending symbols from internet news sources instead of using hardcoded DEFAULT_SYMBOLS. This feature is integrated into the prediction pipeline and dashboard.

## New Components

### 1. **TrendingSymbolsService** (`app/services/trending_symbols_service.py`)
- Fetches trending symbols from news analysis
- Integrates with existing `news_impact_service` and metadata store
- Supports fallback to DEFAULT_SYMBOLS if no trending found
- Includes caching with TTL for performance

**Key Methods:**
```python
get_trending_symbols(fallback_to_defaults=True, limit=25) -> (symbols, source)
get_trending_symbols_cached(ttl_seconds=300) -> (symbols, source)
```

### 2. **Updated PredictionService** (`app/services/prediction_service.py`)
- New parameter: `use_trending: bool = False`
- Intelligently resolves symbols:
  1. If `use_trending=True` → fetch trending symbols
  2. Else if `symbols` provided → use provided symbols
  3. Else → fall back to DEFAULT_SYMBOLS

**Response includes:**
```json
{
  "generated_at": "...",
  "predictions": [...],
  "symbols_used": ["RELIANCE", "TCS", ...],
  "is_trending": true
}
```

### 3. **Updated API Schema** (`app/models/schemas.py`)
```python
class PredictRequest(BaseModel):
    symbols: Optional[List[str]] = None  # Now optional
    horizon_days: int = 1
    model_name: str = "xgboost_classifier"
    atr_multiplier: float = 1.0
    include_live_quote: bool = False
    use_trending: bool = False  # NEW: Request trending symbols
```

### 4. **Updated Predictions Dashboard Page** (`dashboard/views/predictions.py`)
Three symbol source options:
- ✅ **Configured defaults** → Use DEFAULT_SYMBOLS from .env
- 📰 **Trending from news** → Fetch trending symbols from internet (NEW)
- 🎯 **Custom symbols** → User-provided symbols

Visual feedback:
```
📰 Using 25 trending symbols (source: news_trending)
```

## Usage

### Via Dashboard
1. Open **Predictions** page (new page in dashboard)
2. Select model and parameters
3. Choose **"Trending from news"** as symbol source
4. Click "Generate Predictions For Selected Symbols"
5. View predictions with trending symbols highlighted

### Via API
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "use_trending": true,
    "model_name": "xgboost_classifier",
    "horizon_days": 1
  }'
```

### Via Python
```python
from app.services.prediction_service import prediction_service

# Use trending symbols
result = prediction_service.predict(
    symbols=None,
    model_name="xgboost_classifier",
    use_trending=True
)

# Access trending symbols used
symbols_used = result["symbols_used"]  # List of trending symbols
is_trending = result["is_trending"]    # True if trending was used
predictions = result["predictions"]    # Prediction data
```

## Data Flow

```
Dashboard / API Request
    ↓
    ├─ use_trending=True?
    │   ├─ YES → TrendingSymbolsService.get_trending_symbols()
    │   │        ↓
    │   │        ├─ news_impact_service.refresh() → Top opportunities
    │   │        ├─ metadata_store.read_beneficiary_opportunities()
    │   │        └─ Return deduped list
    │   │
    │   └─ NO → Use provided symbols or DEFAULT_SYMBOLS
    │
    ├─ Load movement_model or other model
    ├─ Build features with historical data from SQLiteDataStore
    ├─ Generate predictions
    ├─ Store results to SQLiteDataStore predictions table
    └─ Return predictions + symbols_used + is_trending
```

## Fallback Behavior

If trending symbols fetch **fails or returns empty**:
1. System attempts to fall back to DEFAULT_SYMBOLS
2. Logs warning: `"trending_symbols_fallback"`
3. Continues with default symbols
4. Does NOT break prediction workflow

## Performance Notes

- **Trending fetch timeout:** 30 seconds (configurable via AUTOMATION_STEP_TIMEOUT_SEC)
- **Caching:** 5 minutes TTL by default
- **Limit:** Top 25 symbols returned (configurable)
- **Deduplication:** Automatic removal of duplicates
- **Graceful degradation:** Falls back to defaults on any error

## Configuration

No new environment variables needed. Uses existing:
- `AUTOMATION_STEP_TIMEOUT_SEC` - Timeout for trending fetch
- `DEFAULT_SYMBOLS` - Fallback symbols
- `AUTOMATION_FORCE_NEWS_REFRESH` - Force refresh news analysis

## Testing

All components tested and verified:
```bash
✓ TrendingSymbolsService imports
✓ PredictRequest schema with use_trending
✓ Prediction service parameter passing
✓ API route with new parameter
✓ Dashboard predictions view renders
```

## Files Modified

1. `app/services/trending_symbols_service.py` (NEW)
2. `app/services/prediction_service.py` (MODIFIED)
3. `app/models/schemas.py` (MODIFIED - PredictRequest)
4. `app/api/routes_predict.py` (MODIFIED)
5. `dashboard/views/predictions.py` (MODIFIED)
6. `dashboard/streamlit_app.py` (MODIFIED - added Predictions page)

## Next Steps (Optional)

- Add storage inspector page to view cached trending symbols
- Add UI toggle to refresh trending symbols manually
- Add metrics to track trending symbol prediction accuracy
- Export trending symbols to CSV for analysis
