#!/usr/bin/env python3
"""
Schema management utility for creating test/validation schemas.

This script allows creation of complete test schemas that exactly match
the production structure defined in models.py, providing proper isolation
for validation and testing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from personas_backend.utils.config import ConfigManager


def get_db_engine() -> Engine:
    """Get database engine from config."""
    config = ConfigManager()
    db_config = config.pg_config

    connection_string = (
        f"postgresql://{db_config.user}:{db_config.password}@"
        f"{db_config.host}:{db_config.port}/{db_config.database}"
    )

    return create_engine(connection_string)


def create_test_schema(schema_name: str = "test_schema") -> None:
    """Create a test schema with the same structure as personality_trap."""
    config = ConfigManager()
    db_config = config.pg_config

    engine = create_engine(
        f"postgresql://{db_config.user}:{db_config.password}@{db_config.host}:{db_config.port}/{db_config.database}"
    )

    with engine.connect() as conn:
        with conn.begin():
            # Create test schema
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))

            # Copy all tables from personality_trap to test schema
            tables_query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'personality_trap'
            AND table_type = 'BASE TABLE'
            """

            tables = conn.execute(text(tables_query)).fetchall()

            for (table_name,) in tables:
                # Create table with same structure
                create_table_sql = f"""
                CREATE TABLE {schema_name}.{table_name}
                (LIKE personality_trap.{table_name} INCLUDING ALL)
                """
                conn.execute(text(create_table_sql))
                print(f"   ✅ Created table: {schema_name}.{table_name}")

    print(f"✨ Test schema '{schema_name}' created successfully!")


def seed_test_data(engine: Engine, schema_name: str, personality_ids: list[int]) -> None:
    """
    Seed test schema with minimal data for validation.

    Args:
        engine: Database engine
        schema_name: Target test schema name
        personality_ids: List of personality IDs to copy from production
    """
    print(f"🌱 Seeding test data for personalities: {personality_ids}")

    with engine.connect() as conn:
        # Copy reference questionnaire data for selected personalities
        for personality_id in personality_ids:
            copy_ref_sql = text(
                f"""
                INSERT INTO {schema_name}.reference_questionnaires
                SELECT * FROM personality_trap.reference_questionnaires
                WHERE personality_id = :personality_id
            """
            )
            result = conn.execute(copy_ref_sql, {"personality_id": personality_id})
            print(f"   ✅ Copied {result.rowcount} questions for personality {personality_id}")

        # Copy questionnaire definitions (needed for evaluation)
        copy_questionnaire_sql = text(
            f"""
            INSERT INTO {schema_name}.questionnaire
            SELECT * FROM personality_trap.questionnaire
            WHERE questionnaire = 'epqra'
        """
        )
        result = conn.execute(copy_questionnaire_sql)
        print(f"   ✅ Copied {result.rowcount} questionnaire definitions")

        conn.commit()

    print("   🎯 Test data seeded successfully")


def cleanup_test_schema(engine: Engine, schema_name: str) -> None:
    """Drop a test schema and all its contents."""
    print(f"🧹 Cleaning up test schema: {schema_name}")

    with engine.connect() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        conn.commit()

    print(f"   ✅ Schema {schema_name} removed")


@click.group()
def cli():
    """Schema management for personality trap research."""
    pass


@cli.command()
@click.argument("schema_name")
@click.option(
    "--personality-ids",
    default="1,2,3",
    help="Comma-separated personality IDs to seed (default: 1,2,3)",
)
def create(schema_name: str, personality_ids: str):
    """Create a new test schema with seed data."""
    engine = get_db_engine()

    # Parse personality IDs
    ids = [int(x.strip()) for x in personality_ids.split(",")]

    create_test_schema(schema_name)
    seed_test_data(engine, schema_name, ids)

    print(f"\n🎉 Test schema '{schema_name}' ready!")
    print(f"   📊 Seeded with {len(ids)} personalities")
    print("   🚀 Ready for validation/testing")


@cli.command()
@click.argument("schema_name")
def drop(schema_name: str):
    """Drop a test schema."""
    if schema_name in ["exp", "population", "epqra"]:
        print(f"❌ Cannot drop production schema: {schema_name}")
        return

    engine = get_db_engine()
    cleanup_test_schema(engine, schema_name)


@cli.command()
def list_schemas():
    """List all schemas in the database."""
    engine = get_db_engine()

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

    print("📊 Available schemas:")
    for schema in schemas:
        if schema in ["exp", "population", "epqra"]:
            print(f"   🏢 {schema} (production)")
        else:
            print(f"   🧪 {schema} (test/custom)")


if __name__ == "__main__":
    cli()
