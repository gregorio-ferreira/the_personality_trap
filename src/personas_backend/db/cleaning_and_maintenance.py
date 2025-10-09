"""Maintenance utilities for cleaning PostgreSQL tables and artifacts."""

import datetime
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from personas_backend.db.db_handler import DatabaseHandler
from sqlalchemy import text
from sqlalchemy.engine.base import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

# Type variable for database connection similar to ExperimentHandler
DBConn = Union[Connection, Engine]


class DatabaseMaintenanceHandler:
    """
    Handler for database cleaning and maintenance operations.

    This class provides functions to perform various database maintenance tasks
    such as deleting records, cleaning up old data, and optimizing database performance.
    """

    def __init__(
        self,
        db_handler: Optional[DatabaseHandler] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize database maintenance handler.

        Args:
            db_handler (DatabaseHandler, optional): Database handler instance.
                                                   If None, a new one will be created.
            logger (logging.Logger, optional): Logger instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.db_handler = db_handler or DatabaseHandler(logger=self.logger)
        self.connection = self.db_handler.connection

    def execute_delete(
        self,
        table_name: str,
        schema: str,
        where_clause: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Execute a DELETE statement on the specified table with the given WHERE clause.

        Args:
            table_name (str): Name of the table to delete from
            schema (str): Database schema name
            where_clause (str): WHERE clause for the DELETE statement (without the 'WHERE' keyword)
            parameters (Dict[str, Any], optional): Parameters to bind to the query

        Returns:
            int: Number of deleted rows
        """
        if parameters is None:
            parameters = {}

        full_table_name = f"{schema}.{table_name}"
        sql = text(f"DELETE FROM {full_table_name} WHERE {where_clause}")

        try:
            with self.connection.connect() as connection:
                result = connection.execute(sql, parameters)
                connection.commit()

                rows_deleted: int = int(result.rowcount or 0)
                self.logger.info(
                    f"Deleted {rows_deleted} rows from {full_table_name} where {where_clause}"
                )
                return rows_deleted

        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy Error executing DELETE statement: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error executing DELETE statement: {e}")
            raise

    def truncate_table(self, table_name: str, schema: str) -> None:
        """
        Truncate a table (delete all rows quickly).

        Args:
            table_name (str): Name of the table to truncate
            schema (str): Database schema name
        """
        full_table_name = f"{schema}.{table_name}"
        sql = text(f"TRUNCATE TABLE {full_table_name}")

        try:
            with self.connection.connect() as connection:
                connection.execute(sql)
                connection.commit()
                self.logger.info(f"Truncated table {full_table_name}")

        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy Error truncating table: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error truncating table: {e}")
            raise

    def delete_old_data(
        self,
        table_name: str,
        schema: str,
        date_column: str,
        days_old: int,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Delete data older than a specified number of days.

        Args:
            table_name (str): Name of the table
            schema (str): Database schema name
            date_column (str): Name of the date column to check
            days_old (int): Delete data older than this many days
            parameters (Dict[str, Any], optional): Additional parameters for the query

        Returns:
            int: Number of deleted rows
        """
        where_clause = f"{date_column} < NOW() - INTERVAL '{days_old} days'"
        return self.execute_delete(table_name, schema, where_clause, parameters)

    def delete_by_foreign_key(
        self,
        table_name: str,
        schema: str,
        foreign_key_column: str,
        foreign_key_value: Union[int, str, List],
        parameters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Delete records based on a foreign key relationship.

        Args:
            table_name (str): Name of the table
            schema (str): Database schema name
            foreign_key_column (str): Name of the foreign key column
            foreign_key_value (Union[int, str, List]): Value or list of values of the foreign key
            parameters (Dict[str, Any], optional): Additional parameters for the query

        Returns:
            int: Number of deleted rows
        """
        if parameters is None:
            parameters = {}

        if isinstance(foreign_key_value, list):
            foreign_keys_str = ", ".join([str(v) for v in foreign_key_value])
            where_clause = f"{foreign_key_column} IN ({foreign_keys_str})"
        else:
            where_clause = f"{foreign_key_column} = {foreign_key_value}"

        return self.execute_delete(table_name, schema, where_clause, parameters)

    def vacuum_table(self, table_name: str, schema: str, full: bool = False) -> None:
        """
        Run VACUUM on a table to reclaim storage and update statistics.

        Args:
            table_name (str): Name of the table
            schema (str): Database schema name
            full (bool, optional): Whether to run VACUUM FULL. Defaults to False.
        """
        full_table_name = f"{schema}.{table_name}"
        vacuum_type = "FULL" if full else ""
        vacuum_cmd = f"VACUUM {vacuum_type} {full_table_name}"

        try:
            # VACUUM cannot be executed inside a transaction, so we need to:
            # 1. Get the raw connection from the engine
            # 2. Turn off autocommit (which would start a transaction)
            # 3. Execute the vacuum command
            # 4. Turn autocommit back on

            # Get raw connection from the engine
            raw_connection = self.connection.raw_connection()

            # Save current isolation level and turn off autocommit
            original_isolation_level = raw_connection.isolation_level
            raw_connection.set_isolation_level(0)  # 0 = ISOLATION_LEVEL_AUTOCOMMIT

            # Execute the VACUUM command
            cursor = raw_connection.cursor()
            cursor.execute(vacuum_cmd)
            cursor.close()

            # Restore original isolation level
            raw_connection.set_isolation_level(original_isolation_level)

            self.logger.info(f"VACUUM {vacuum_type} completed on table {full_table_name}")

        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy Error running VACUUM: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error running VACUUM: {e}")
            raise

    def backup_database(self, backup_dir: str, backup_name: Optional[str] = None) -> str:
        """
        Create a full backup of the database and store it locally.

        Args:
            backup_dir (str): Directory where backup will be stored
            backup_name (str, optional): Custom name for the backup file.
                                        If None, a timestamp-based name will be used.

        Returns:
            str: Path to the created backup file
        """
        # Get database connection parameters
        pg_config = self.db_handler.config_manager.pg_config

        # Create backup directory if it doesn't exist
        Path(backup_dir).mkdir(parents=True, exist_ok=True)

        # Generate backup filename with timestamp if not provided
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if backup_name is None:
            backup_name = f"{pg_config.database}_backup_{timestamp}.sql"

        backup_path = os.path.join(backup_dir, backup_name)

        # Build the pg_dump command
        dump_cmd = [
            "pg_dump",
            f"-h{pg_config.host}",
            f"-p{pg_config.port if hasattr(pg_config, 'port') else '5432'}",
            f"-U{pg_config.user}",
            "-Fc",  # Custom format, compressed
            f"-f{backup_path}",
            pg_config.database,
        ]

        # Set PGPASSWORD environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = pg_config.password

        try:
            self.logger.info(f"Starting database backup to {backup_path}")
            process = subprocess.Popen(
                dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.logger.info(f"Database backup completed successfully: {backup_path}")
                return backup_path
            else:
                error_msg = stderr.decode("utf-8")
                self.logger.error(f"Database backup failed: {error_msg}")
                raise Exception(f"Database backup failed: {error_msg}")

        except Exception as e:
            self.logger.error(f"Error during database backup: {str(e)}")
            raise

    def restore_database(self, backup_path: str) -> None:
        """
        Restore a database from a backup file.

        Args:
            backup_path (str): Path to the backup file
        """
        # Get database connection parameters
        pg_config = self.db_handler.config_manager.pg_config

        # Check if backup file exists
        if not os.path.exists(backup_path):
            error_msg = f"Backup file not found: {backup_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # Build the pg_restore command
        restore_cmd = [
            "pg_restore",
            f"-h{pg_config.host}",
            f"-p{pg_config.port if hasattr(pg_config, 'port') else '5432'}",
            f"-U{pg_config.user}",
            "-d",
            pg_config.database,
            "--clean",  # Clean (drop) database objects before recreating
            backup_path,
        ]

        # Set PGPASSWORD environment variable
        env = os.environ.copy()
        env["PGPASSWORD"] = pg_config.password

        try:
            self.logger.info(f"Starting database restore from {backup_path}")
            process = subprocess.Popen(
                restore_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env
            )
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                self.logger.info("Database restore completed successfully")
            else:
                error_msg = stderr.decode("utf-8")
                self.logger.error(f"Database restore failed: {error_msg}")
                raise Exception(f"Database restore failed: {error_msg}")

        except Exception as e:
            self.logger.error(f"Error during database restore: {str(e)}")
            raise

    def analyze_table(self, table_name: str, schema: str) -> None:
        """
        Run ANALYZE on a table to update statistics.

        Args:
            table_name (str): Name of the table
            schema (str): Database schema name
        """
        full_table_name = f"{schema}.{table_name}"
        sql = text(f"ANALYZE {full_table_name}")

        try:
            with self.connection.connect() as connection:
                connection.execute(sql)
                self.logger.info(f"ANALYZE completed on table {full_table_name}")

        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy Error running ANALYZE: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error running ANALYZE: {e}")
            raise

    def list_large_tables(self, min_size_mb: float = 10.0) -> pd.DataFrame:
        """
        List tables that are larger than the specified threshold.

        Args:
            min_size_mb (float, optional): Minimum size in MB. Defaults to 10.0.

        Returns:
            pd.DataFrame: DataFrame containing table names and sizes
        """
        sql = text(
            """
            SELECT
                nspname AS schema,
                relname AS table,
                pg_size_pretty(pg_total_relation_size(C.oid)) AS total_size,
                pg_size_pretty(pg_relation_size(C.oid)) AS data_size,
                pg_size_pretty(pg_total_relation_size(C.oid) - pg_relation_size(C.oid)) AS external_size,
                pg_total_relation_size(C.oid) / 1024.0 / 1024.0 AS size_mb
            FROM pg_class C
            LEFT JOIN pg_namespace N ON (N.oid = C.relnamespace)
            WHERE nspname NOT IN ('pg_catalog', 'information_schema')
              AND C.relkind <> 'i'
              AND nspname !~ '^pg_toast'
            ORDER BY pg_total_relation_size(C.oid) DESC
        """
        )

        try:
            df = pd.read_sql(sql, self.connection)
            return df[df["size_mb"] >= min_size_mb]

        except SQLAlchemyError as e:
            self.logger.error(f"SQLAlchemy Error listing large tables: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error listing large tables: {e}")
            raise

    def optimize_tables(self, schema: Optional[str] = None, min_size_mb: float = 10.0) -> None:
        """
        Optimize large tables in the database.

        Args:
            schema (str, optional): Limit to specific schema. If None, all schemas are included.
            min_size_mb (float, optional): Minimum table size to consider for optimization.
        """
        # Get large tables
        large_tables = self.list_large_tables(min_size_mb)

        # Filter by schema if specified
        if schema is not None:
            large_tables = large_tables[large_tables["schema"] == schema]

        if large_tables.empty:
            self.logger.info(f"No tables larger than {min_size_mb}MB found for optimization")
            return

        # Optimize each large table
        for _, row in large_tables.iterrows():
            self.logger.info(
                f"Optimizing table {row['schema']}.{row['table']} (Size: {row['total_size']})"
            )

            try:
                # First analyze the table to update statistics
                self.analyze_table(row["table"], row["schema"])

                # Then vacuum the table to reclaim space
                self.vacuum_table(row["table"], row["schema"])

                self.logger.info(f"Successfully optimized {row['schema']}.{row['table']}")
            except Exception as e:
                self.logger.error(f"Error optimizing table {row['schema']}.{row['table']}: {e}")

    def setup_automated_backups(self, backup_dir: str, keep_days: int = 7) -> str:
        """
        Create a crontab entry to schedule automated backups.

        Args:
            backup_dir (str): Directory where backups will be stored
            keep_days (int, optional): Number of days to keep backups. Defaults to 7.

        Returns:
            str: The crontab command that was added
        """
        # Get the absolute path to the current Python interpreter
        python_path = sys.executable

        # Get the absolute path to the root directory of the project
        project_root = str(Path(__file__).resolve().parents[3])

        # Create the script content
        backup_script_path = os.path.join(backup_dir, "run_backup.py")
        with open(backup_script_path, "w") as f:
            f.write(
                f"""#!/usr/bin/env python3
import os
import sys
import datetime
import subprocess
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, "{project_root}")

from personas_backend.utils.logger import setup_logger
from personas_backend.db.db_handler import DatabaseHandler
from personas_backend.db.cleaning_and_maintenance import DatabaseMaintenanceHandler

# Setup
logger = setup_logger("automated_backup")
db_handler = DatabaseHandler()
db_maintenance = DatabaseMaintenanceHandler(db_handler=db_handler, logger=logger)

# Create backup
backup_dir = "{backup_dir}"
try:
    backup_path = db_maintenance.backup_database(backup_dir)
    logger.info(f"Backup created successfully at {{backup_path}}")

    # Clean up old backups (older than {keep_days} days)
    today = datetime.datetime.now()
    for backup_file in Path(backup_dir).glob("*_backup_*.sql"):
        file_date_str = backup_file.name.split("_backup_")[1].split(".")[0]
        try:
            file_date = datetime.datetime.strptime(file_date_str, "%Y%m%d_%H%M%S")
            if (today - file_date).days > {keep_days}:
                backup_file.unlink()
                logger.info(f"Deleted old backup: {{backup_file}}")
        except (ValueError, IndexError):
            pass
except Exception as e:
    logger.error(f"Backup failed: {{e}}")
finally:
    db_handler.close_connection()
"""
            )

        # Make the script executable
        os.chmod(backup_script_path, 0o755)

        # Create a crontab entry to run daily at 2am
        cron_cmd = f"0 2 * * * {python_path} {backup_script_path} >> {backup_dir}/backup.log 2>&1"

        try:
            # Get existing crontab
            process = subprocess.Popen(
                ["crontab", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            current_crontab_bytes, error = process.communicate()

            if process.returncode != 0 and "no crontab" not in error.decode("utf-8"):
                raise Exception(f"Error reading current crontab: {error.decode('utf-8')}")

            # Add new cron job if it doesn't exist already
            current_crontab: List[str] = current_crontab_bytes.decode("utf-8").strip().split("\n")
            if not current_crontab or current_crontab == [""]:
                current_crontab = []

            # Check if backup job already exists
            backup_job_exists = False
            for line in current_crontab:
                if backup_script_path in line:
                    backup_job_exists = True
                    break

            if not backup_job_exists:
                # Add the new cron job
                current_crontab.append(cron_cmd)

                # Write the updated crontab
                process = subprocess.Popen(
                    ["crontab", "-"], stdin=subprocess.PIPE, stderr=subprocess.PIPE
                )
                process.communicate(input="\n".join(current_crontab).encode())

                if process.returncode == 0:
                    self.logger.info(f"Automated backup scheduled successfully: {cron_cmd}")
                else:
                    raise Exception(f"Error scheduling backup: {error.decode('utf-8')}")
            else:
                self.logger.info("Automated backup already scheduled")

            return cron_cmd

        except Exception as e:
            self.logger.error(f"Error setting up automated backup: {e}")
            raise
