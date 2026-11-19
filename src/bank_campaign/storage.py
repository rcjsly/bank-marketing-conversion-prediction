"""Persist minimal prediction audit records without customer attributes."""

import os
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, insert, text
from sqlalchemy.engine import Engine

from bank_campaign.train import ROOT

metadata = MetaData()
predictions = Table(
    "predictions", metadata,
    Column("id", Integer, primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("model_name", String(100), nullable=False),
    Column("probability", Float, nullable=False),
    Column("threshold", Float, nullable=False),
    Column("predicted_conversion", Integer, nullable=False),
)


def create_storage() -> Engine:
    default_path = ROOT / "artifacts" / "predictions.sqlite3"
    url = os.environ.get("DATABASE_URL")
    if not url:
        default_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{default_path.as_posix()}"
    engine = create_engine(url, pool_pre_ping=True)
    metadata.create_all(engine)
    return engine


def check_storage(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def record_prediction(engine: Engine, model_name: str, probability: float,
                      threshold: float) -> None:
    with engine.begin() as connection:
        connection.execute(insert(predictions).values(
            created_at=datetime.now(timezone.utc),
            model_name=model_name,
            probability=probability,
            threshold=threshold,
            predicted_conversion=int(probability >= threshold),
        ))
