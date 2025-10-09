#!/usr/bin/env python3
"""
Schema management utilities for the personalities backend.

This script provides utilities for creating and managing database schemas
for different environments (testing, development, staging).
"""

import argparse
import sys
from pathlib import Path

# Add the src directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from personas_backend.db.db_handler import DatabaseHandler
from personas_backend.db.models import SQLModel
from personas_backend.utils.logger import setup_logger
from sqlalchemy import text


def create_schema_with_tables(schema_name: str, logger=None) -> bool:
    """
    Create a new schema and all required tables.

    Args:
        schema_name: Name of the schema to create
        logger: Optional logger instance

    Returns:
        bool: True if successful, False otherwise
    """
    if not logger:
        logger = setup_logger("schema_manager")

    try:
        # Create database handler
        db_handler = DatabaseHandler(logger=logger)
        engine = db_handler.connection

        with engine.connect() as conn:
            # Create the schema
            logger.info(f"Creating schema: {schema_name}")
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))

            # Set the schema for table creation
            # We need to temporarily override the schema configuration
            import os

            original_schema = os.environ.get("PERSONAS_TARGET_SCHEMA")
            os.environ["PERSONAS_TARGET_SCHEMA"] = schema_name

            try:
                # Create all tables in the new schema
                logger.info(f"Creating tables in schema: {schema_name}")
                SQLModel.metadata.create_all(engine)
                logger.info(f"✅ Schema {schema_name} created successfully with all tables")

            finally:
                # Restore original schema setting
                if original_schema:
                    os.environ["PERSONAS_TARGET_SCHEMA"] = original_schema
                elif "PERSONAS_TARGET_SCHEMA" in os.environ:
                    del os.environ["PERSONAS_TARGET_SCHEMA"]

            conn.commit()

        db_handler.close_connection()
        return True

    except Exception as e:
        logger.error(f"Failed to create schema {schema_name}: {e}")
        return False


def drop_schema(schema_name: str, cascade: bool = False, logger=None) -> bool:
    """
    Drop a schema and optionally all its contents.

    Args:
        schema_name: Name of the schema to drop
        cascade: Whether to cascade the drop (remove all contents)
        logger: Optional logger instance

    Returns:
        bool: True if successful, False otherwise
    """
    if not logger:
        logger = setup_logger("schema_manager")

    # Safety check - don't allow dropping production schemas
    protected_schemas = ["personality_trap", "public", "information_schema"]
    if schema_name in protected_schemas:
        logger.error(f"Cannot drop protected schema: {schema_name}")
        return False

    try:
        db_handler = DatabaseHandler(logger=logger)
        engine = db_handler.connection

        with engine.connect() as conn:
            cascade_sql = " CASCADE" if cascade else ""
            logger.info(f"Dropping schema: {schema_name}{cascade_sql}")
            conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name}{cascade_sql}"))
            conn.commit()
            logger.info(f"✅ Schema {schema_name} dropped successfully")

        db_handler.close_connection()
        return True

    except Exception as e:
        logger.error(f"Failed to drop schema {schema_name}: {e}")
        return False


def list_schemas(logger=None) -> list:
    """
    List all schemas in the database.

    Args:
        logger: Optional logger instance

    Returns:
        list: List of schema names
    """
    if not logger:
        logger = setup_logger("schema_manager")

    try:
        db_handler = DatabaseHandler(logger=logger)
        engine = db_handler.connection

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                ORDER BY schema_name
            """
                )
            )
            schemas = [row[0] for row in result]

        db_handler.close_connection()
        return schemas

    except Exception as e:
        logger.error(f"Failed to list schemas: {e}")
        return []


def main():
    """Command-line interface for schema management."""
    parser = argparse.ArgumentParser(description="Manage database schemas")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create schema command
    create_parser = subparsers.add_parser("create", help="Create a new schema with tables")
    create_parser.add_argument("schema_name", help="Name of the schema to create")

    # Drop schema command
    drop_parser = subparsers.add_parser("drop", help="Drop a schema")
    drop_parser.add_argument("schema_name", help="Name of the schema to drop")
    drop_parser.add_argument("--cascade", action="store_true", help="Cascade the drop operation")

    # List schemas command
    subparsers.add_parser("list", help="List all schemas")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    logger = setup_logger("schema_manager")

    if args.command == "create":
        success = create_schema_with_tables(args.schema_name, logger)
        sys.exit(0 if success else 1)

    elif args.command == "drop":
        success = drop_schema(args.schema_name, args.cascade, logger)
        sys.exit(0 if success else 1)

    elif args.command == "list":
        schemas = list_schemas(logger)
        print("Available schemas:")
        for schema in schemas:
            print(f"  - {schema}")


if __name__ == "__main__":
    main()
