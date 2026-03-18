from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from src.config import settings
from src.db import validate_pgvector_schema_dimensions


def _get_column_type(engine: Engine, table: str, column: str) -> str:
    with engine.begin() as conn:
        return conn.execute(
            text(
                """
                SELECT format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = :schema
                  AND c.relname = :table
                  AND a.attname = :column
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                """
            ),
            {
                "schema": settings.DB.SCHEMA,
                "table": table,
                "column": column,
            },
        ).scalar_one()


def test_head_migration_uses_configured_vector_dimensions(
    monkeypatch,
    alembic_cfg: Config,
    alembic_engine: Engine,
) -> None:
    monkeypatch.setattr(settings.VECTOR_STORE, "DIMENSIONS", 256)

    with alembic_engine.begin() as conn:
        alembic_cfg.attributes["connection"] = conn
        command.upgrade(alembic_cfg, "head")

    assert _get_column_type(alembic_engine, "documents", "embedding") == "vector(256)"
    assert _get_column_type(alembic_engine, "message_embeddings", "embedding") == (
        "vector(256)"
    )


def test_dimension_validator_raises_when_runtime_config_drifts_from_schema(
    monkeypatch,
    alembic_cfg: Config,
    alembic_engine: Engine,
) -> None:
    monkeypatch.setattr(settings.VECTOR_STORE, "DIMENSIONS", 256)

    with alembic_engine.begin() as conn:
        alembic_cfg.attributes["connection"] = conn
        command.upgrade(alembic_cfg, "head")

    monkeypatch.setattr(settings.VECTOR_STORE, "DIMENSIONS", 384)

    with alembic_engine.begin() as conn, pytest.raises(
        RuntimeError,
        match="VECTOR_STORE_DIMENSIONS=384",
    ):
        validate_pgvector_schema_dimensions(conn)
