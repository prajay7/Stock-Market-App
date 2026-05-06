from __future__ import annotations

from datetime import datetime
from typing import Iterable

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, MetaData, String, Table, UniqueConstraint, create_engine, desc, select


class MetadataStore:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, future=True)
        self.metadata = MetaData()

        self.news = Table(
            "news_items",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(20), index=True),
            Column("source", String(100)),
            Column("title", String(500)),
            Column("summary", String(2000)),
            Column("url", String(1000), unique=True),
            Column("published_at", DateTime, index=True),
            Column("sentiment_score", Float),
            Column("relevance_score", Float),
        )

        self.predictions = Table(
            "predictions",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("generated_at", DateTime, index=True),
            Column("symbol", String(20), index=True),
            Column("model_name", String(100)),
            Column("prob_up", Float),
            Column("predicted_return", Float),
            Column("confidence", Float),
            Column("payload", JSON),
        )

        self.news_impact_items = Table(
            "news_impact_items",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("generated_at", DateTime, index=True),
            Column("article_hash", String(128), unique=True, index=True),
            Column("source", String(120)),
            Column("title", String(500)),
            Column("link", String(1200)),
            Column("published_at", DateTime, index=True),
            Column("primary_company", String(200), index=True),
            Column("primary_ticker", String(32), index=True),
            Column("event_type", String(80), index=True),
            Column("sentiment_label", String(20), index=True),
            Column("impact_score", Float),
            Column("confidence_score", Float),
            Column("is_actionable", Boolean),
            Column("freshness_score", Float),
            Column("relation_strength", Float),
            Column("price_opportunity_score", Float),
            Column("overall_score", Float),
            Column("payload", JSON),
        )

        self.news_impact_signals = Table(
            "news_impact_signals",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("generated_at", DateTime, index=True),
            Column("article_hash", String(128), index=True),
            Column("headline", String(500)),
            Column("source", String(120)),
            Column("published_at", DateTime, index=True),
            Column("primary_company", String(200), index=True),
            Column("primary_ticker", String(32), index=True),
            Column("beneficiary_company", String(200), index=True),
            Column("beneficiary_ticker", String(32), index=True),
            Column("relation", String(120)),
            Column("relation_strength", Float),
            Column("sentiment_label", String(20), index=True),
            Column("event_type", String(80), index=True),
            Column("impact_score", Float),
            Column("freshness_score", Float),
            Column("price_change_pct_1d", Float),
            Column("price_reaction_ok", Boolean),
            Column("price_opportunity_score", Float),
            Column("signal_score", Float),
            Column("is_early_opportunity", Boolean, index=True),
            Column("reason", String(1000)),
            Column("payload", JSON),
        )

        self.analyzed_news_signals = Table(
            "analyzed_news_signals",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("article_hash", String(128), unique=True, index=True),
            Column("title", String(500)),
            Column("link", String(1200)),
            Column("source", String(120)),
            Column("published_at", DateTime, index=True),
            Column("primary_company", String(200), index=True),
            Column("primary_ticker", String(32), index=True),
            Column("sector", String(120), index=True),
            Column("event_type", String(80), index=True),
            Column("sentiment_label", String(20), index=True),
            Column("sentiment_score", Float),
            Column("impact_score", Float),
            Column("confidence_score", Float),
            Column("is_actionable", Boolean),
            Column("summary", String(3000)),
            Column("created_at", DateTime, index=True),
        )

        self.beneficiary_opportunities = Table(
            "beneficiary_opportunities",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("signal_id", Integer, ForeignKey("analyzed_news_signals.id"), index=True),
            Column("company", String(200), index=True),
            Column("ticker", String(32), index=True),
            Column("relation", String(120)),
            Column("relation_strength", Float),
            Column("benefit_score", Float),
            Column("freshness_score", Float),
            Column("price_change_pct", Float),
            Column("price_opportunity_score", Float),
            Column("overall_score", Float),
            Column("timing_label", String(20), index=True),
            Column("reason", String(1000)),
            Column("signal_price", Float),
            Column("signal_timestamp", DateTime, index=True),
            Column("price_source", String(120)),
            Column("created_at", DateTime, index=True),
        )

        self.signal_outcomes = Table(
            "signal_outcomes",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("opportunity_id", Integer, ForeignKey("beneficiary_opportunities.id"), index=True),
            Column("ticker", String(32), index=True),
            Column("evaluation_horizon_days", Integer, index=True),
            Column("target_date", DateTime, index=True),
            Column("entry_price", Float),
            Column("exit_price", Float),
            Column("absolute_return", Float),
            Column("percent_return", Float),
            Column("benchmark_return", Float),
            Column("alpha_return", Float),
            Column("is_positive", Boolean),
            Column("evaluation_status", String(20), index=True),
            Column("evaluated_at", DateTime, index=True),
            Column("created_at", DateTime, index=True),
            UniqueConstraint("opportunity_id", "evaluation_horizon_days", name="uq_signal_outcome_opportunity_horizon"),
        )

        self.watchlists = Table(
            "watchlists",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("name", String(200), unique=True, index=True),
            Column("description", String(1000)),
            Column("is_active", Boolean, default=True, index=True),
            Column("created_at", DateTime, index=True),
            Column("updated_at", DateTime, index=True),
        )

        self.watchlist_items = Table(
            "watchlist_items",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("watchlist_id", Integer, ForeignKey("watchlists.id"), index=True),
            Column("item_type", String(50), index=True),  # company / ticker / sector / event_type
            Column("item_value", String(200), index=True),
            Column("normalized_value", String(200), index=True),
            Column("created_at", DateTime, index=True),
            UniqueConstraint("watchlist_id", "item_type", "normalized_value", name="uq_watchlist_item"),
        )

        self.alert_rules = Table(
            "alert_rules",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("name", String(200), unique=True, index=True),
            Column("is_active", Boolean, default=True, index=True),
            Column("watchlist_id", Integer, ForeignKey("watchlists.id"), nullable=True, index=True),
            Column("sentiment_filter", String(20), nullable=True),  # positive/neutral/negative
            Column("actionable_only", Boolean, default=False),
            Column("min_confidence_score", Float, nullable=True),
            Column("min_impact_score", Float, nullable=True),
            Column("min_overall_score", Float, nullable=True),
            Column("timing_labels", String(500), nullable=True),  # serialized JSON list
            Column("event_types", String(500), nullable=True),  # serialized JSON list
            Column("sectors", String(500), nullable=True),  # serialized JSON list
            Column("max_price_move_pct", Float, nullable=True),
            Column("notification_channels", String(500), nullable=True),  # serialized JSON list
            Column("cooldown_minutes", Integer, nullable=True),
            Column("created_at", DateTime, index=True),
            Column("updated_at", DateTime, index=True),
        )

        self.generated_alerts = Table(
            "generated_alerts",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("rule_id", Integer, ForeignKey("alert_rules.id"), index=True),
            Column("signal_id", Integer, ForeignKey("analyzed_news_signals.id"), nullable=True, index=True),
            Column("opportunity_id", Integer, ForeignKey("beneficiary_opportunities.id"), nullable=True, index=True),
            Column("title", String(500)),
            Column("message", String(2000)),
            Column("severity", String(20), index=True),  # info / warning / high
            Column("status", String(20), index=True),  # new / seen / sent / failed
            Column("notification_channel", String(50), index=True),  # in_app / email / webhook / telegram / slack
            Column("dedupe_key", String(200), index=True),
            Column("created_at", DateTime, index=True),
            Column("sent_at", DateTime, nullable=True, index=True),
            Column("error_message", String(500), nullable=True),
        )

        self.paper_trades = Table(
            "paper_trades",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("symbol", String(32), index=True),
            Column("source_type", String(20), index=True),
            Column("source_id", Integer, index=True),
            Column("source_label", String(500)),
            Column("entry_date", DateTime, index=True),
            Column("exit_date", DateTime, nullable=True, index=True),
            Column("entry_price", Float),
            Column("exit_price", Float, nullable=True),
            Column("quantity", Float, nullable=True),
            Column("capital", Float, nullable=True),
            Column("status", String(20), index=True),
            Column("notes", String(3000)),
            Column("trade_reason", String(1000)),
            Column("current_price", Float, nullable=True),
            Column("current_pnl", Float, nullable=True),
            Column("current_return_pct", Float, nullable=True),
            Column("realized_pnl", Float, nullable=True),
            Column("realized_return_pct", Float, nullable=True),
            Column("holding_days", Integer, nullable=True),
            Column("created_at", DateTime, index=True),
            Column("updated_at", DateTime, index=True),
        )

        self.metadata.create_all(self.engine)
        self._ensure_sqlite_schema_compatibility()

    def _ensure_sqlite_schema_compatibility(self) -> None:
        if self.engine.dialect.name != "sqlite":
            return

        required_columns = {
            "beneficiary_opportunities": {
                "signal_price": "FLOAT",
                "signal_timestamp": "DATETIME",
                "price_source": "VARCHAR(120)",
            }
        }

        with self.engine.begin() as conn:
            for table_name, columns in required_columns.items():
                existing = {
                    row[1]
                    for row in conn.exec_driver_sql(f"PRAGMA table_info({table_name})").fetchall()
                }
                for column_name, sql_type in columns.items():
                    if column_name in existing:
                        continue
                    conn.exec_driver_sql(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sql_type}")

    def upsert_news_records(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.engine.begin() as conn:
            existing_urls = set(conn.execute(select(self.news.c.url)).scalars().all())
            inserts = [r for r in rows if r.get("url") and r.get("url") not in existing_urls]
            if inserts:
                conn.execute(self.news.insert(), inserts)

    def write_predictions(self, rows: Iterable[dict], model_name: str) -> None:
        rows = list(rows)
        if not rows:
            return
        now = datetime.utcnow()
        payloads = []
        for row in rows:
            payloads.append(
                {
                    "generated_at": now,
                    "symbol": row.get("symbol"),
                    "model_name": model_name,
                    "prob_up": row.get("prob_up"),
                    "predicted_return": row.get("predicted_return"),
                    "confidence": row.get("confidence"),
                    "payload": row,
                }
            )
        with self.engine.begin() as conn:
            conn.execute(self.predictions.insert(), payloads)

    def upsert_news_impact_items(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.engine.begin() as conn:
            existing_hashes = set(conn.execute(select(self.news_impact_items.c.article_hash)).scalars().all())
            inserts = [r for r in rows if r.get("article_hash") and r.get("article_hash") not in existing_hashes]
            if inserts:
                conn.execute(self.news_impact_items.insert(), inserts)

    def write_news_impact_signals(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.engine.begin() as conn:
            conn.execute(self.news_impact_signals.insert(), rows)

    def read_news_signal_history(self, limit: int = 200) -> list[dict]:
        limit = max(1, int(limit))
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(self.news_impact_signals)
                .order_by(desc(self.news_impact_signals.c.generated_at), desc(self.news_impact_signals.c.signal_score))
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def upsert_analyzed_news_signal(self, row: dict) -> int | None:
        article_hash = row.get("article_hash")
        if not article_hash:
            return None
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(self.analyzed_news_signals.c.id).where(self.analyzed_news_signals.c.article_hash == article_hash)
            ).scalar_one_or_none()
            if existing is not None:
                return int(existing)

            result = conn.execute(self.analyzed_news_signals.insert().values(**row))
            inserted = result.inserted_primary_key
            if inserted and inserted[0] is not None:
                return int(inserted[0])

            fallback = conn.execute(
                select(self.analyzed_news_signals.c.id).where(self.analyzed_news_signals.c.article_hash == article_hash)
            ).scalar_one_or_none()
            return int(fallback) if fallback is not None else None

    def write_beneficiary_opportunities(self, rows: Iterable[dict]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self.engine.begin() as conn:
            conn.execute(self.beneficiary_opportunities.insert(), rows)

    def upsert_signal_outcome(self, row: dict) -> None:
        opportunity_id = row.get("opportunity_id")
        horizon = row.get("evaluation_horizon_days")
        if opportunity_id is None or horizon is None:
            return
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(self.signal_outcomes.c.id)
                .where(self.signal_outcomes.c.opportunity_id == opportunity_id)
                .where(self.signal_outcomes.c.evaluation_horizon_days == horizon)
            ).scalar_one_or_none()
            if existing is None:
                conn.execute(self.signal_outcomes.insert().values(**row))
            else:
                conn.execute(
                    self.signal_outcomes.update()
                    .where(self.signal_outcomes.c.id == existing)
                    .values(**row)
                )

    def read_beneficiary_opportunities_with_signal(self, limit: int = 1000) -> list[dict]:
        limit = max(1, int(limit))
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(
                    self.beneficiary_opportunities,
                    self.analyzed_news_signals.c.article_hash,
                    self.analyzed_news_signals.c.title,
                    self.analyzed_news_signals.c.link,
                    self.analyzed_news_signals.c.source,
                    self.analyzed_news_signals.c.published_at,
                    self.analyzed_news_signals.c.primary_company,
                    self.analyzed_news_signals.c.primary_ticker,
                    self.analyzed_news_signals.c.sector,
                    self.analyzed_news_signals.c.event_type,
                    self.analyzed_news_signals.c.sentiment_label,
                    self.analyzed_news_signals.c.confidence_score,
                )
                .join(self.analyzed_news_signals, self.beneficiary_opportunities.c.signal_id == self.analyzed_news_signals.c.id)
                .order_by(desc(self.beneficiary_opportunities.c.created_at), desc(self.beneficiary_opportunities.c.overall_score))
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def read_signal_outcomes(self, limit: int = 5000) -> list[dict]:
        limit = max(1, int(limit))
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(self.signal_outcomes)
                .order_by(desc(self.signal_outcomes.c.created_at), desc(self.signal_outcomes.c.evaluated_at))
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def read_signal_outcomes_joined(self, limit: int = 5000) -> list[dict]:
        limit = max(1, int(limit))
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(
                    self.signal_outcomes,
                    self.beneficiary_opportunities.c.company,
                    self.beneficiary_opportunities.c.ticker.label("beneficiary_ticker"),
                    self.beneficiary_opportunities.c.relation,
                    self.beneficiary_opportunities.c.timing_label,
                    self.beneficiary_opportunities.c.overall_score.label("opportunity_overall_score"),
                    self.analyzed_news_signals.c.primary_company,
                    self.analyzed_news_signals.c.primary_ticker,
                    self.analyzed_news_signals.c.event_type,
                    self.analyzed_news_signals.c.sector,
                    self.analyzed_news_signals.c.sentiment_label,
                    self.analyzed_news_signals.c.confidence_score,
                    self.analyzed_news_signals.c.title,
                    self.analyzed_news_signals.c.created_at.label("signal_created_at"),
                )
                .join(self.beneficiary_opportunities, self.signal_outcomes.c.opportunity_id == self.beneficiary_opportunities.c.id)
                .join(self.analyzed_news_signals, self.beneficiary_opportunities.c.signal_id == self.analyzed_news_signals.c.id)
                .order_by(desc(self.signal_outcomes.c.created_at), desc(self.signal_outcomes.c.evaluated_at))
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def read_beneficiary_opportunity_history(self, limit: int = 200) -> list[dict]:
        limit = max(1, int(limit))
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(
                    self.beneficiary_opportunities,
                    self.analyzed_news_signals.c.article_hash,
                    self.analyzed_news_signals.c.title,
                    self.analyzed_news_signals.c.link,
                    self.analyzed_news_signals.c.source,
                    self.analyzed_news_signals.c.published_at,
                    self.analyzed_news_signals.c.primary_company,
                    self.analyzed_news_signals.c.primary_ticker,
                    self.analyzed_news_signals.c.event_type,
                    self.analyzed_news_signals.c.sentiment_label,
                    self.analyzed_news_signals.c.impact_score,
                    self.analyzed_news_signals.c.confidence_score,
                )
                .join(self.analyzed_news_signals, self.beneficiary_opportunities.c.signal_id == self.analyzed_news_signals.c.id)
                .order_by(desc(self.beneficiary_opportunities.c.created_at), desc(self.beneficiary_opportunities.c.overall_score))
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    # Watchlist CRUD
    def create_watchlist(self, name: str, description: str = "", is_active: bool = True) -> int:
        now = datetime.utcnow()
        with self.engine.begin() as conn:
            result = conn.execute(
                self.watchlists.insert().values(
                    name=name,
                    description=description,
                    is_active=is_active,
                    created_at=now,
                    updated_at=now,
                )
            )
            return int(result.inserted_primary_key[0])

    def read_watchlists(self, active_only: bool = False) -> list[dict]:
        with self.engine.begin() as conn:
            query = select(self.watchlists).order_by(desc(self.watchlists.c.updated_at))
            if active_only:
                query = query.where(self.watchlists.c.is_active == True)
            rows = conn.execute(query).mappings()
            return [dict(row) for row in rows]

    def read_watchlist(self, watchlist_id: int) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(self.watchlists).where(self.watchlists.c.id == watchlist_id)
            ).mappings().first()
            return dict(row) if row else None

    def update_watchlist(self, watchlist_id: int, name: str = None, description: str = None, is_active: bool = None) -> None:
        updates = {"updated_at": datetime.utcnow()}
        if name is not None:
            updates["name"] = name
        if description is not None:
            updates["description"] = description
        if is_active is not None:
            updates["is_active"] = is_active
        with self.engine.begin() as conn:
            conn.execute(
                self.watchlists.update().where(self.watchlists.c.id == watchlist_id).values(**updates)
            )

    def delete_watchlist(self, watchlist_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(self.watchlist_items.delete().where(self.watchlist_items.c.watchlist_id == watchlist_id))
            conn.execute(self.watchlists.delete().where(self.watchlists.c.id == watchlist_id))

    # Watchlist item CRUD
    def create_watchlist_item(self, watchlist_id: int, item_type: str, item_value: str, normalized_value: str) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                self.watchlist_items.insert().values(
                    watchlist_id=watchlist_id,
                    item_type=item_type,
                    item_value=item_value,
                    normalized_value=normalized_value.upper(),
                    created_at=datetime.utcnow(),
                )
            )
            return int(result.inserted_primary_key[0])

    def read_watchlist_items(self, watchlist_id: int) -> list[dict]:
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(self.watchlist_items)
                .where(self.watchlist_items.c.watchlist_id == watchlist_id)
                .order_by(self.watchlist_items.c.created_at)
            ).mappings()
            return [dict(row) for row in rows]

    def delete_watchlist_item(self, item_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(self.watchlist_items.delete().where(self.watchlist_items.c.id == item_id))

    # Alert rule CRUD
    def create_alert_rule(self, rule_data: dict) -> int:
        rule_data["created_at"] = datetime.utcnow()
        rule_data["updated_at"] = datetime.utcnow()
        with self.engine.begin() as conn:
            result = conn.execute(self.alert_rules.insert().values(**rule_data))
            return int(result.inserted_primary_key[0])

    def read_alert_rules(self, active_only: bool = False) -> list[dict]:
        with self.engine.begin() as conn:
            query = select(self.alert_rules).order_by(desc(self.alert_rules.c.updated_at))
            if active_only:
                query = query.where(self.alert_rules.c.is_active == True)
            rows = conn.execute(query).mappings()
            return [dict(row) for row in rows]

    def read_alert_rule(self, rule_id: int) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(self.alert_rules).where(self.alert_rules.c.id == rule_id)
            ).mappings().first()
            return dict(row) if row else None

    def update_alert_rule(self, rule_id: int, rule_data: dict) -> None:
        rule_data["updated_at"] = datetime.utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                self.alert_rules.update().where(self.alert_rules.c.id == rule_id).values(**rule_data)
            )

    def delete_alert_rule(self, rule_id: int) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.generated_alerts.delete().where(self.generated_alerts.c.rule_id == rule_id)
            )
            conn.execute(self.alert_rules.delete().where(self.alert_rules.c.id == rule_id))

    # Generated alert CRUD
    def create_alert(self, alert_data: dict) -> int:
        alert_data["created_at"] = datetime.utcnow()
        with self.engine.begin() as conn:
            result = conn.execute(self.generated_alerts.insert().values(**alert_data))
            return int(result.inserted_primary_key[0])

    def read_alerts(self, limit: int = 500, status: str = None, severity: str = None) -> list[dict]:
        limit = max(1, int(limit))
        with self.engine.begin() as conn:
            query = select(self.generated_alerts).order_by(desc(self.generated_alerts.c.created_at))
            if status:
                query = query.where(self.generated_alerts.c.status == status)
            if severity:
                query = query.where(self.generated_alerts.c.severity == severity)
            rows = conn.execute(query.limit(limit)).mappings()
            return [dict(row) for row in rows]

    def read_alert_with_context(self, alert_id: int) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(
                    self.generated_alerts,
                    self.beneficiary_opportunities.c.company,
                    self.beneficiary_opportunities.c.ticker,
                    self.beneficiary_opportunities.c.timing_label,
                    self.beneficiary_opportunities.c.overall_score.label("opportunity_overall_score"),
                    self.analyzed_news_signals.c.primary_company,
                    self.analyzed_news_signals.c.primary_ticker,
                    self.analyzed_news_signals.c.sector.label("sector"),
                    self.analyzed_news_signals.c.event_type.label("event_type"),
                    self.analyzed_news_signals.c.sentiment_label,
                    self.analyzed_news_signals.c.confidence_score.label("signal_confidence_score"),
                )
                .select_from(
                    self.generated_alerts
                    .outerjoin(self.beneficiary_opportunities, self.generated_alerts.c.opportunity_id == self.beneficiary_opportunities.c.id)
                    .outerjoin(self.analyzed_news_signals, self.generated_alerts.c.signal_id == self.analyzed_news_signals.c.id)
                )
                .where(self.generated_alerts.c.id == alert_id)
            ).mappings().first()
            return dict(row) if row else None

    def read_beneficiary_opportunity(self, opportunity_id: int) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(
                    self.beneficiary_opportunities,
                    self.analyzed_news_signals.c.article_hash,
                    self.analyzed_news_signals.c.title,
                    self.analyzed_news_signals.c.link,
                    self.analyzed_news_signals.c.source,
                    self.analyzed_news_signals.c.published_at,
                    self.analyzed_news_signals.c.primary_company,
                    self.analyzed_news_signals.c.primary_ticker,
                    self.analyzed_news_signals.c.sector.label("sector"),
                    self.analyzed_news_signals.c.event_type.label("signal_event_type"),
                    self.analyzed_news_signals.c.sentiment_label.label("signal_sentiment_label"),
                    self.analyzed_news_signals.c.confidence_score.label("signal_confidence_score"),
                )
                .join(self.analyzed_news_signals, self.beneficiary_opportunities.c.signal_id == self.analyzed_news_signals.c.id)
                .where(self.beneficiary_opportunities.c.id == opportunity_id)
            ).mappings().first()
            return dict(row) if row else None

    def create_paper_trade(self, trade_data: dict) -> int:
        now = datetime.utcnow()
        payload = dict(trade_data)
        payload.setdefault("status", "open")
        payload.setdefault("created_at", now)
        payload.setdefault("updated_at", now)
        with self.engine.begin() as conn:
            result = conn.execute(self.paper_trades.insert().values(**payload))
            return int(result.inserted_primary_key[0])

    def read_paper_trades(self, limit: int = 500, status: str = None, symbol: str = None, source_type: str = None) -> list[dict]:
        limit = max(1, int(limit))
        with self.engine.begin() as conn:
            query = select(self.paper_trades).order_by(desc(self.paper_trades.c.updated_at), desc(self.paper_trades.c.created_at))
            if status:
                query = query.where(self.paper_trades.c.status == status)
            if symbol:
                query = query.where(self.paper_trades.c.symbol == symbol)
            if source_type:
                query = query.where(self.paper_trades.c.source_type == source_type)
            rows = conn.execute(query.limit(limit)).mappings()
            return [dict(row) for row in rows]

    def read_paper_trade(self, trade_id: int) -> dict | None:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(self.paper_trades).where(self.paper_trades.c.id == trade_id)
            ).mappings().first()
            return dict(row) if row else None

    def update_paper_trade(self, trade_id: int, trade_data: dict) -> None:
        payload = dict(trade_data)
        payload["updated_at"] = datetime.utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                self.paper_trades.update().where(self.paper_trades.c.id == trade_id).values(**payload)
            )

    def close_paper_trade(self, trade_id: int, trade_data: dict) -> None:
        payload = dict(trade_data)
        payload["status"] = "closed"
        payload["updated_at"] = datetime.utcnow()
        with self.engine.begin() as conn:
            conn.execute(
                self.paper_trades.update().where(self.paper_trades.c.id == trade_id).values(**payload)
            )

    def read_open_paper_trades(self, limit: int = 500) -> list[dict]:
        return self.read_paper_trades(limit=limit, status="open")

    def read_closed_paper_trades(self, limit: int = 500) -> list[dict]:
        return self.read_paper_trades(limit=limit, status="closed")

    def read_recent_alerts(self, hours: int = 24, limit: int = 100) -> list[dict]:
        limit = max(1, int(limit))
        cutoff = datetime.utcnow() - __import__('datetime').timedelta(hours=hours)
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(self.generated_alerts)
                .where(self.generated_alerts.c.created_at >= cutoff)
                .order_by(desc(self.generated_alerts.c.created_at))
                .limit(limit)
            ).mappings()
            return [dict(row) for row in rows]

    def update_alert_status(self, alert_id: int, status: str, error_message: str = None, sent_at: datetime = None) -> None:
        updates = {"status": status}
        if error_message:
            updates["error_message"] = error_message
        if sent_at:
            updates["sent_at"] = sent_at
        with self.engine.begin() as conn:
            conn.execute(
                self.generated_alerts.update().where(self.generated_alerts.c.id == alert_id).values(**updates)
            )

    def check_alert_cooldown(self, rule_id: int, dedupe_key: str, cooldown_minutes: int) -> bool:
        if cooldown_minutes is None or cooldown_minutes <= 0:
            return False
        cutoff = datetime.utcnow() - __import__('datetime').timedelta(minutes=cooldown_minutes)
        with self.engine.begin() as conn:
            existing = conn.execute(
                select(self.generated_alerts)
                .where(self.generated_alerts.c.rule_id == rule_id)
                .where(self.generated_alerts.c.dedupe_key == dedupe_key)
                .where(self.generated_alerts.c.created_at >= cutoff)
            ).first()
            return existing is not None


# Module-level singleton instance
def _get_metadata_store():
    from app.core.config import get_settings
    settings = get_settings()
    return MetadataStore(settings.database_url)


metadata_store = _get_metadata_store()
