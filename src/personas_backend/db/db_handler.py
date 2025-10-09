"""Database helper wrapping SQLAlchemy engine creation and Alembic migrations."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd  # type: ignore
import sqlalchemy  # type: ignore
from personas_backend.utils.config import ConfigManager
from sqlalchemy import create_engine, text  # type: ignore
from sqlalchemy.exc import SQLAlchemyError  # type: ignore

from alembic import command
from alembic.config import Config


class DatabaseHandler:
    """Encapsulate database connectivity and migration utilities."""

    def __init__(
        self,
        config_manager: Optional[ConfigManager] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize database connection.

        Args:
            config_manager (ConfigManager, optional): Config manager; if None a
                new one is created.
            logger (Logger, optional): Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config_manager = config_manager or ConfigManager(logger=self.logger)
        self.connection = self.connect_to_postgres()

    def connect_to_postgres(self, port: Optional[int] = None) -> sqlalchemy.engine.base.Engine:
        """Create a connection to the PostgreSQL database.

        Args:
            port: Optional port override; defaults to configuration (pg.port)

        Returns:
            SQLAlchemy Engine bound to the target database.
        """
        pg_config = self.config_manager.pg_config
        actual_port = port if port is not None else pg_config.port

        # Establish the connection
        try:
            connection_url = (
                f"postgresql://{pg_config.user}:{pg_config.password}@"
                f"{pg_config.host}:{actual_port}/{pg_config.database}"
            )
            engine = create_engine(connection_url)
            engine.connect()
            self.logger.info("Connected to the database!")
            return engine
        except SQLAlchemyError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.error("Database connection error: %s", exc)
            raise SQLAlchemyError(f"Failed to connect to database: {exc}") from exc

    def close_connection(self) -> None:
        """
        Close the database connection and release resources.

        This method should be called when the database connection
        is no longer needed to properly clean up resources.
        """
        try:
            self.connection.dispose()
            self.logger.info("Database connection closed successfully!")
        except SQLAlchemyError as exc:
            self.logger.error("Error closing database connection: %s", exc)

    def check_table_exists(self, table_schema: str, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_schema (str): Schema name
            table_name (str): Table name

        Returns:
            bool: True if table exists, False otherwise
        """
        query = f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE  table_schema = '{table_schema}'
                AND    table_name   = '{table_name}'
            );
        """
        # Cast the pandas result to bool explicitly
        return bool(pd.read_sql_query(query, self.connection).iloc[0, 0])

    def refresh_materialized_view(self, schema: str, view_name: str) -> None:
        """
        Refresh a materialized view in the database.

        Args:
            schema (str): Schema name
            view_name (str): View name
        """
        try:
            with self.connection.connect() as connection:
                refresh_query = text(f"REFRESH MATERIALIZED VIEW {schema}.{view_name};")
                connection.execute(refresh_query)
                connection.commit()
                self.logger.info(
                    "Materialized view %s.%s refreshed successfully!",
                    schema,
                    view_name,
                )
        except SQLAlchemyError as e:
            self.logger.error(
                "Error refreshing materialized view %s.%s: %s",
                schema,
                view_name,
                e,
            )
            # Attempt to rollback if there's a transaction error
            if "PendingRollbackError" in str(e) or "invalid transaction" in str(e):
                self.rollback_transaction()
            raise

    def rollback_transaction(self) -> None:
        """
        Roll back any pending transactions in the database connection.

        This method should be called when a database operation fails
        and leaves the connection in a pending transaction state.
        """
        if hasattr(self, "connection") and self.connection is not None:
            try:
                with self.connection.connect() as connection:
                    connection.rollback()
                    self.logger.info("Transaction rolled back successfully!")
            except SQLAlchemyError as exc:
                self.logger.error("Error rolling back transaction: %s", exc)

    # ------------------------------------------------------------------
    # Schema management helpers
    # ------------------------------------------------------------------
    def schema_exists(self, schema_name: str) -> bool:
        """Return ``True`` if the given schema already exists in PostgreSQL."""

        if not schema_name:
            raise ValueError("schema_name must be a non-empty string")

        query = text(
            """
            SELECT 1
            FROM information_schema.schemata
            WHERE schema_name = :schema_name
            LIMIT 1
            """
        )

        try:
            with self.connection.connect() as connection:
                params = {"schema_name": schema_name}
                result = connection.execute(query, params)
                return result.scalar_one_or_none() is not None
        except SQLAlchemyError as exc:
            self.logger.error("Error checking schema %s existence: %s", schema_name, exc)
            raise

    def run_migrations(self, *, schema: Optional[str] = None, revision: str = "head") -> None:
        """Execute Alembic migrations for the target schema.

        The schema is determined by:
        1. Explicit schema parameter (if provided)
        2. YAML config: schema.target_schema (experimental schema)
        3. YAML config: schema.default_schema (production schema)

        Alembic migrations will read the schema directly from YAML config
        via the ConfigManager, no environment variables are used.
        """

        # Prefer explicit schema arg; else use configured target_schema if set;
        # fall back to default_schema for backwards compatibility.
        cfg_schema = self.config_manager.schema_config
        target_schema = schema or cfg_schema.target_schema or cfg_schema.default_schema

        project_root = self._resolve_project_root()
        alembic_ini = project_root / "alembic.ini"

        if not alembic_ini.exists():
            raise FileNotFoundError(f"Alembic configuration not found at {alembic_ini}")

        config = Config(str(alembic_ini))
        config.set_main_option("sqlalchemy.url", str(self.connection.url))
        config.set_main_option("script_location", str(project_root / "alembic"))

        # Pass the target schema to Alembic via config attributes
        # This will be read by the migration script
        config.attributes["target_schema"] = target_schema

        try:
            command.upgrade(config, revision)
            self.logger.info(
                "Applied Alembic migrations up to %s for schema %s",
                revision,
                target_schema,
            )
        except SQLAlchemyError:
            # Re-raise SQL errors directly so callers can handle them.
            raise
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures
            self.logger.error(
                "Alembic migration failed for schema %s: %s",
                target_schema,
                exc,
            )
            raise

    def _resolve_project_root(self) -> Path:
        """Locate the repository root containing ``pyproject.toml``."""

        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").exists():
                return parent
        # Fall back to three levels up (src/.. assumption) if discovery fails.
        return current.parents[3]
