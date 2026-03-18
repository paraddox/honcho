import contextvars

from sqlalchemy import MetaData, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from src.config import settings

connect_args = {"prepare_threshold": None}

# Context variable to store request context
request_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_context", default=None
)

engine_kwargs = {}

if settings.DB.POOL_CLASS == "null":
    engine_kwargs["poolclass"] = NullPool
else:
    # Only add pool-related kwargs for pooled connections
    engine_kwargs.update(  # pyright: ignore
        {
            "pool_pre_ping": settings.DB.POOL_PRE_PING,
            "pool_size": settings.DB.POOL_SIZE,
            "max_overflow": settings.DB.MAX_OVERFLOW,
            "pool_timeout": settings.DB.POOL_TIMEOUT,
            "pool_recycle": settings.DB.POOL_RECYCLE,
            "pool_use_lifo": settings.DB.POOL_USE_LIFO,
        }
    )

engine = create_async_engine(
    settings.DB.CONNECTION_URI,
    connect_args=connect_args,
    echo=settings.DB.SQL_DEBUG,
    **engine_kwargs,
)

SessionLocal = async_sessionmaker(
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    bind=engine,
)

# Define your naming convention
convention = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",  # Index - supports multi-column
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",  # Unique constraint - supports multi-column
    "ck": "ck_%(table_name)s_%(constraint_name)s",  # Check constraint
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",  # Foreign key - supports composite keys
    "pk": "pk_%(table_name)s",  # Primary key
}

table_schema = settings.DB.SCHEMA
# Note: column_0_N_name expands to include all columns in multi-column constraints
# e.g., "workspace_id_tenant_id" for a composite constraint on both columns
meta = MetaData(naming_convention=convention)
meta.schema = table_schema
Base = declarative_base(metadata=meta)

_PGVECTOR_COLUMNS = (
    ("documents", "embedding"),
    ("message_embeddings", "embedding"),
)


def _postgres_vectors_are_active() -> bool:
    return settings.VECTOR_STORE.TYPE == "pgvector" or not settings.VECTOR_STORE.MIGRATED


def validate_pgvector_schema_dimensions(connection: Connection) -> None:
    """Ensure active pgvector columns match the configured embedding size."""
    if not _postgres_vectors_are_active():
        return

    expected_type = f"vector({settings.VECTOR_STORE.DIMENSIONS})"
    mismatches: list[str] = []
    missing_columns: list[str] = []

    for table_name, column_name in _PGVECTOR_COLUMNS:
        current_type = connection.execute(
            text(
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
                "schema": table_schema,
                "table_name": table_name,
                "column_name": column_name,
            },
        ).scalar_one_or_none()

        if current_type is None:
            missing_columns.append(f"{table_name}.{column_name}")
        elif current_type != expected_type:
            mismatches.append(f"{table_name}.{column_name}={current_type}")

    if not missing_columns and not mismatches:
        return

    details: list[str] = []
    if missing_columns:
        details.append("missing columns: " + ", ".join(missing_columns))
    if mismatches:
        details.append("found " + ", ".join(mismatches))

    raise RuntimeError(
        "Configured VECTOR_STORE_DIMENSIONS="
        + f"{settings.VECTOR_STORE.DIMENSIONS} requires pgvector columns of type "
        + f"{expected_type}, but {'; '.join(details)}. "
        + "Run the matching migration/config pair before starting Honcho."
    )


async def validate_configured_vector_dimensions() -> None:
    """Validate active pgvector columns against the current runtime config."""
    if not _postgres_vectors_are_active():
        return

    async with engine.connect() as connection:
        await connection.run_sync(validate_pgvector_schema_dimensions)


async def init_db():
    """Initialize the database using Alembic migrations"""
    from alembic import command
    from alembic.config import Config

    async with engine.connect() as connection:
        # Create schema if it doesn't exist
        await connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{table_schema}"'))
        # Install pgvector extension if it doesn't exist
        await connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await connection.commit()

    # Run Alembic migrations
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    await validate_configured_vector_dimensions()
