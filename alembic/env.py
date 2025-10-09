from __future__ import annotations

import os
from logging.config import fileConfig

from personas_backend.db.models import Persona, SQLModel  # noqa: F401
from personas_backend.db.schema_config import get_default_schema
from sqlalchemy import engine_from_config, pool

from alembic import context

# Alembic Config object.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = SQLModel.metadata

# Schema Management Notes:
# - Alembic supports per-schema version tracking via version_table_schema
# - Each schema gets its own alembic_version table for independent versioning
# - The target schema is passed via config.attributes["target_schema"]
#   from db_handler
# - Falls back to default_schema (personality_trap) if not specified
# - This allows the same migration to be applied to multiple schemas
#   independently


def get_url() -> str:
    """Resolve database URL from env or fallback to local dev default."""
    return (
        os.getenv("DATABASE_URL")
        or "postgresql+psycopg2://personas:personas@localhost:5432/personas"
    )


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()

    # Get target schema from config.attributes (set by db_handler)
    # or fall back to default_schema
    schema_name = config.attributes.get("target_schema") or get_default_schema()

    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema=schema_name,
        version_table="alembic_version",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        {"sqlalchemy.url": get_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    # Get target schema from config.attributes (set by db_handler)
    # or fall back to default_schema
    schema_name = config.attributes.get("target_schema") or get_default_schema()

    with connectable.connect() as connection:
        # Ensure the schema exists
        from sqlalchemy import text

        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
        connection.commit()

        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema=schema_name,
            version_table="alembic_version",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():  # type: ignore[attr-defined]
    run_migrations_offline()
else:
    run_migrations_online()
