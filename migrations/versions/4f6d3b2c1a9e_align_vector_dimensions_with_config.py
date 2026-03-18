"""align pgvector columns with configured embedding dimensions

Revision ID: 4f6d3b2c1a9e
Revises: e4eba9cfaa6f
Create Date: 2026-03-18 18:10:00.000000

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import get_schema, index_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "4f6d3b2c1a9e"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()
DEFAULT_DIMENSIONS = 1536

VECTOR_COLUMNS = (
    ("documents", "embedding", "ix_documents_embedding_hnsw"),
    ("message_embeddings", "embedding", "ix_message_embeddings_embedding_hnsw"),
)


def _get_vector_type(table_name: str, column_name: str) -> str | None:
    conn = op.get_bind()
    return conn.execute(
        sa.text(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = :schema
              AND c.relname = :table_name
              AND a.attname = :column_name
              AND a.attnum > 0
              AND NOT a.attisdropped
            """
        ),
        {
            "schema": schema,
            "table_name": table_name,
            "column_name": column_name,
        },
    ).scalar_one_or_none()


def _count_non_null_vectors(table_name: str, column_name: str) -> int:
    conn = op.get_bind()
    return int(
        conn.execute(
            sa.text(
                f'SELECT COUNT(*) FROM "{schema}"."{table_name}" '
                + f'WHERE "{column_name}" IS NOT NULL'
            )
        ).scalar_one()
    )


def _assert_column_can_change_dimensions(
    table_name: str,
    column_name: str,
    target_dimensions: int,
    *,
    action: str,
) -> bool:
    current_type = _get_vector_type(table_name, column_name)
    target_type = f"vector({target_dimensions})"

    if current_type == target_type:
        return False

    non_null_vectors = _count_non_null_vectors(table_name, column_name)
    if non_null_vectors:
        raise RuntimeError(
            f"Cannot {action} {schema}.{table_name}.{column_name} from "
            + f"{current_type} to {target_type} while {non_null_vectors} "
            + "stored embeddings remain. Clear or re-embed the data first."
        )

    return True


def _drop_vector_indexes() -> None:
    inspector = sa.inspect(op.get_bind())
    for table_name, _column_name, index_name in VECTOR_COLUMNS:
        if index_exists(table_name, index_name, inspector):
            op.drop_index(index_name, table_name=table_name, schema=schema)


def _create_vector_indexes() -> None:
    op.create_index(
        "ix_documents_embedding_hnsw",
        "documents",
        ["embedding"],
        schema=schema,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index(
        "ix_message_embeddings_embedding_hnsw",
        "message_embeddings",
        ["embedding"],
        schema=schema,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def _alter_vector_column(
    table_name: str,
    column_name: str,
    target_dimensions: int,
) -> None:
    op.execute(
        sa.text(
            f"""
            ALTER TABLE "{schema}"."{table_name}"
            ALTER COLUMN "{column_name}" TYPE vector({target_dimensions})
            USING "{column_name}"::vector({target_dimensions})
            """
        )
    )


def _align_vector_dimensions(target_dimensions: int, *, action: str) -> None:
    changes_required = [
        (table_name, column_name)
        for table_name, column_name, _index_name in VECTOR_COLUMNS
        if _assert_column_can_change_dimensions(
            table_name,
            column_name,
            target_dimensions,
            action=action,
        )
    ]

    if not changes_required:
        return

    _drop_vector_indexes()
    for table_name, column_name in changes_required:
        _alter_vector_column(table_name, column_name, target_dimensions)
    _create_vector_indexes()


def upgrade() -> None:
    """Align pgvector columns with the configured embedding dimensionality."""
    _align_vector_dimensions(
        settings.VECTOR_STORE.DIMENSIONS,
        action="upgrade embedding dimensions for",
    )


def downgrade() -> None:
    """Restore pgvector columns to the historical 1536-dimension schema."""
    _align_vector_dimensions(
        DEFAULT_DIMENSIONS,
        action="downgrade embedding dimensions for",
    )
