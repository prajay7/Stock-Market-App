from functools import lru_cache
import os
import json
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Stock AI Platform"
    app_version: str = "0.1.0"
    environment: str = "local"
    log_level: str = "INFO"
    api_port: int = 8000

    alpha_vantage_api_key: str = ""
    polygon_api_key: str = ""
    stooq_api_key: str = ""
    finnhub: str = Field(default="", alias="FINNHUB")
    finnhub_webhook_secret: str = ""
    news_llm_api_key: str = ""

    data_provider: str = "yahoo"
    news_provider: str = "alphavantage"
    database_url: str = "sqlite:///./stock_ai.db"
    db_path: Path = Path("./data/stock_data.db")
    log_dir: Path = Path("./logs")

    default_symbols_raw: str = Field(default="AAPL,MSFT,NVDA,TSLA,AMZN,META", alias="DEFAULT_SYMBOLS")
    benchmark_symbol: str = "SPY"

    model_dir: Path = Path("./data/models")
    movement_model_path: Path = Path("./models/movement_model.pkl")
    model_keep_last_versions: int = 0
    raw_data_dir: Path = Path("./data/raw")
    processed_data_dir: Path = Path("./data/processed")
    output_dir: Path = Path("./data/outputs")

    historical_interval: str = "1d"
    default_period: str = "10y"
    historical_lookback_days: int = 3650
    news_refresh_minutes: int = 30
    scheduler_enabled: bool = True

    transaction_cost_bps: float = 10.0
    top_n_selection: int = 5

    max_news_items_per_symbol: int = 100
    request_timeout_sec: float = 20.0
    fetch_delay_seconds: float = 1.0
    max_retries: int = 3
    force_refresh: bool = False
    market_timezone: str = "Asia/Kolkata"
    market_open_time: str = "09:15"
    market_close_time: str = "15:30"

    openai_predict_enabled: bool = False
    openai_predict_model_name: str = "gpt-4o-mini"
    openai_predict_base_url: str = "https://api.openai.com/v1"
    openai_predict_temperature: float = 0.0
    openai_predict_timeout_sec: float = 20.0
    openai_predict_api_key: str = ""

    automation_enabled: bool = False
    automation_interval_minutes: int = 30
    automation_max_symbols: int = 120
    automation_global_symbols_raw: str = "SPY,QQQ,DIA,INDA,EEM,EWJ,VGK,GLD"
    automation_prediction_model: str = "auto"
    automation_horizon_days: int = 1
    automation_include_live_quote: bool = False
    automation_top_suggestions: int = 25
    automation_force_news_refresh: bool = True
    automation_step_timeout_sec: int = 45
    automation_auto_resume_on_startup: bool = False
    automation_resume_retry_count: int = 1
    automation_resume_retry_backoff_sec: int = 10

    news_analysis_provider: str = "rule"
    news_llm_model_name: str = "gpt-4o-mini"
    news_llm_base_url: str = "https://api.openai.com/v1"
    news_rss_feeds_path: Path = Path("./data/rss_feeds.json")
    news_company_relations_path: Path = Path("./data/company_relations.json")
    news_company_ticker_map_path: Path = Path("./data/company_tickers.json")
    news_ticker_aliases_path: Path = Path("./data/ticker_aliases.json")
    news_cache_ttl_seconds: int = 900
    news_max_articles_per_refresh: int = 50
    news_max_feeds_per_refresh: int = 5
    news_score_impact_weight: float = 0.35
    news_score_relation_weight: float = 0.25
    news_score_freshness_weight: float = 0.20
    news_score_price_weight: float = 0.20
    news_price_reaction_max_abs_pct: float = 3.5
    news_price_move_early_threshold_pct: float = 1.5
    news_price_move_late_threshold_pct: float = 4.0
    news_min_signal_score: float = 0.60
    news_top_opportunities_limit: int = 10
    news_persist_signals_enabled: bool = True

    # Watchlist and alert settings
    alerts_enabled: bool = True
    alert_scan_enabled: bool = False
    alert_scan_interval_minutes: int = 60
    default_cooldown_minutes: int = 60
    enabled_notification_channels_raw: str = "in_app"  # comma-separated
    max_alerts_in_ui: int = 500
    max_recent_signals_for_alert_scan: int = 100

    # Paper trading settings
    paper_trading_enabled: bool = True
    paper_trading_price_refresh_enabled: bool = False
    paper_trading_price_refresh_interval_minutes: int = 30
    paper_trading_price_refresh_limit: int = 200
    paper_trading_default_capital: float = 1000.0
    max_paper_trades_in_ui: int = 500

    # Notification channel configs (optional)
    webhook_url: str = ""
    webhook_timeout_sec: float = 10.0
    smtp_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    alert_email_from: str = ""
    alert_email_to: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    slack_webhook_url: str = ""

    backtest_enabled: bool = True
    backtest_horizons_raw: str = "1,3,5,7"
    backtest_scheduler_enabled: bool = False
    backtest_scheduler_interval_minutes: int = 60
    benchmark_ticker: str = ""
    trading_day_fallback_mode: str = "next_available_close"
    max_backtest_rows_in_ui: int = 500

    @property
    def enabled_notification_channels(self) -> list[str]:
        raw = str(self.enabled_notification_channels_raw or "").strip()
        if not raw:
            return ["in_app"]
        channels = [ch.strip() for ch in raw.split(",") if ch.strip()]
        return channels or ["in_app"]

    @property
    def news_effective_ticker_map_path(self) -> Path:
        if self.news_company_ticker_map_path.exists():
            return self.news_company_ticker_map_path
        return self.news_ticker_aliases_path

    @property
    def backtest_horizons(self) -> list[int]:
        raw = str(self.backtest_horizons_raw or "").strip()
        if not raw:
            return [1, 3, 5, 7]
        values: list[int] = []
        try:
            if raw.startswith("["):
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    values = [int(x) for x in parsed]
            else:
                values = [int(part.strip()) for part in raw.split(",") if part.strip()]
        except Exception:
            values = [1, 3, 5, 7]
        values = sorted({v for v in values if v > 0})
        return values or [1, 3, 5, 7]

    @property
    def default_symbols(self) -> List[str]:
        return [s.strip().upper() for s in self.default_symbols_raw.split(",") if s.strip()]

    @property
    def news_llm_api_key_effective(self) -> str:
        return self.news_llm_api_key or os.getenv("NEWS_LLM_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

    @property
    def openai_predict_api_key_effective(self) -> str:
        return (
            self.openai_predict_api_key
            or os.getenv("OPENAI_PREDICT_API_KEY", "")
            or os.getenv("OPENAI_API_KEY", "")
            or self.news_llm_api_key_effective
        )

    @property
    def automation_global_symbols(self) -> list[str]:
        raw = str(self.automation_global_symbols_raw or "").strip()
        if not raw:
            return []
        return [part.strip().upper() for part in raw.split(",") if part.strip()]

    @property
    def news_rss_feeds_default(self) -> list[dict[str, str]]:
        # Free, public RSS search endpoints that cover broad market/business coverage.
        return [
            {
                "label": "Google News - Markets",
                "url": "https://news.google.com/rss/search?q=stock+market+news+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
            },
            {
                "label": "Google News - Indian Stocks",
                "url": "https://news.google.com/rss/search?q=Indian+stocks+market+news+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
            },
            {
                "label": "Google News - Business",
                "url": "https://news.google.com/rss/search?q=business+news+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
            },
        ]

    def ensure_dirs(self) -> None:
        for path in [self.model_dir, self.raw_data_dir, self.processed_data_dir, self.output_dir, self.log_dir, self.db_path.parent, self.movement_model_path.parent]:
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings
