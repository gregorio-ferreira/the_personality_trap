"""Initial schema with auto-increment and server-side timestamps

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-10-08 20:00:00.000000

This is the consolidated migration for The Personality Trap dataset.
It creates the complete schema with all improvements:
- Auto-increment (SERIAL) primary keys for all tables
- Server-side CURRENT_TIMESTAMP defaults for created columns
- All necessary sequences, indexes, and constraints

Tables created:
- personas (with model_print and population_print columns)
- reference_questionnaires
- random_questionnaires
- experiments_list (with auto-increment and timestamp defaults)
- experiments_groups (with auto-increment and timestamp defaults)
- eval_questionnaires
- questionnaire
- experiment_request_metadata (with auto-increment and timestamp defaults)

Note: The materialized view 'experiments_evals' must be created manually
after this migration using create_experiments_evals_view.sql
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create complete schema with auto-increment and timestamps."""
    # Get target schema from Alembic config
    from alembic import context

    target_schema = None

    try:
        config = context.config
        if hasattr(config, "attributes") and config.attributes:
            target_schema = config.attributes.get("target_schema")
    except Exception:
        pass

    if not target_schema:
        try:
            from personas_backend.utils.config import ConfigManager

            cfg = ConfigManager()
            target_schema = cfg.pg_config.target_schema or cfg.pg_config.default_schema
        except Exception:
            pass

    if not target_schema:
        target_schema = "personality_trap"

    # Create schema
    op.execute(f"CREATE SCHEMA IF NOT EXISTS {target_schema}")

    # =========================================================================
    # TABLE: personas
    # =========================================================================
    op.create_table(
        "personas",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("ref_personality_id", sa.Integer(), nullable=True),
        sa.Column("population", sa.String(length=100), nullable=True),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("gender_original", sa.String(length=50), nullable=True),
        sa.Column("gender", sa.String(length=20), nullable=True),
        sa.Column("race_original", sa.String(length=100), nullable=True),
        sa.Column("race", sa.String(length=50), nullable=True),
        sa.Column("sexual_orientation_original", sa.String(length=100), nullable=True),
        sa.Column("sexual_orientation", sa.String(length=20), nullable=True),
        sa.Column("ethnicity", sa.String(length=100), nullable=True),
        sa.Column("religious_belief_original", sa.String(length=100), nullable=True),
        sa.Column("religious_belief", sa.String(length=50), nullable=True),
        sa.Column("occupation_original", sa.String(length=255), nullable=True),
        sa.Column("occupation", sa.String(length=100), nullable=True),
        sa.Column("political_orientation_original", sa.String(length=50), nullable=True),
        sa.Column("political_orientation", sa.String(length=20), nullable=True),
        sa.Column("location_original", sa.String(length=255), nullable=True),
        sa.Column("location", sa.String(length=100), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("word_count_description", sa.Integer(), nullable=True),
        sa.Column("repetitions", sa.Integer(), nullable=True),
        sa.Column("model_print", sa.String(length=50), nullable=True),
        sa.Column("population_print", sa.String(length=50), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_ref_personality_id",
        "personas",
        ["ref_personality_id"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_population",
        "personas",
        ["population"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_model",
        "personas",
        ["model"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_gender",
        "personas",
        ["gender"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_race",
        "personas",
        ["race"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_sexual_orientation",
        "personas",
        ["sexual_orientation"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_religious_belief",
        "personas",
        ["religious_belief"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_occupation",
        "personas",
        ["occupation"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_political_orientation",
        "personas",
        ["political_orientation"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_location",
        "personas",
        ["location"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_model_print",
        "personas",
        ["model_print"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_personas_population_print",
        "personas",
        ["population_print"],
        schema=target_schema,
    )

    # =========================================================================
    # TABLE: reference_questionnaires
    # =========================================================================
    op.create_table(
        "reference_questionnaires",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("personality_id", sa.Integer(), nullable=True),
        sa.Column("question_number", sa.Integer(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=10), nullable=True),
        sa.Column("key", sa.Boolean(), nullable=True),
        sa.Column("answer", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_reference_questionnaires_personality_id",
        "reference_questionnaires",
        ["personality_id"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_reference_questionnaires_question_number",
        "reference_questionnaires",
        ["question_number"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_reference_questionnaires_category",
        "reference_questionnaires",
        ["category"],
        schema=target_schema,
    )

    # =========================================================================
    # TABLE: random_questionnaires
    # =========================================================================
    op.create_table(
        "random_questionnaires",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("personality_id", sa.Integer(), nullable=True),
        sa.Column("question_number", sa.Integer(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=10), nullable=True),
        sa.Column("key", sa.Boolean(), nullable=True),
        sa.Column("answer", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_random_questionnaires_personality_id",
        "random_questionnaires",
        ["personality_id"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_random_questionnaires_question_number",
        "random_questionnaires",
        ["question_number"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_random_questionnaires_category",
        "random_questionnaires",
        ["category"],
        schema=target_schema,
    )

    # =========================================================================
    # TABLE: experiments_groups (with server-side timestamp)
    # =========================================================================
    op.create_table(
        "experiments_groups",
        sa.Column("experiments_group_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "created",
            sa.String(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("system_role", sa.String(), nullable=True),
        sa.Column("base_prompt", sa.String(), nullable=True),
        sa.Column("concluded", sa.Boolean(), nullable=True),
        sa.Column("processed", sa.Boolean(), nullable=True),
        sa.Column("translated", sa.Boolean(), nullable=True),
        sa.Column("temperature", sa.Float(), nullable=True),
        sa.Column("top_p", sa.Float(), nullable=True),
        sa.Column("ii_repeated", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("experiments_group_id"),
        schema=target_schema,
    )

    # =========================================================================
    # TABLE: experiments_list (with server-side timestamp)
    # =========================================================================
    op.create_table(
        "experiments_list",
        sa.Column("experiment_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiments_group_id", sa.Integer(), nullable=True),
        sa.Column("repeated", sa.Integer(), nullable=True),
        sa.Column(
            "created",
            sa.String(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=True,
        ),
        sa.Column("questionnaire", sa.String(), nullable=True),
        sa.Column("language_instructions", sa.String(), nullable=True),
        sa.Column("language_questionnaire", sa.String(), nullable=True),
        sa.Column("model_provider", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("population", sa.String(), nullable=True),
        sa.Column("personality_id", sa.Integer(), nullable=True),
        sa.Column("succeeded", sa.Boolean(), nullable=True),
        sa.Column("llm_explanation", sa.Text(), nullable=True),
        sa.Column("repo_sha", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("experiment_id"),
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_experiments_list_experiments_group_id",
        "experiments_list",
        ["experiments_group_id"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_experiments_list_personality_id",
        "experiments_list",
        ["personality_id"],
        schema=target_schema,
    )

    # =========================================================================
    # TABLE: eval_questionnaires
    # =========================================================================
    op.create_table(
        "eval_questionnaires",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("question_number", sa.Integer(), nullable=True),
        sa.Column("answer", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_eval_questionnaires_experiment_id",
        "eval_questionnaires",
        ["experiment_id"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_eval_questionnaires_question_number",
        "eval_questionnaires",
        ["question_number"],
        schema=target_schema,
    )

    # =========================================================================
    # TABLE: questionnaire
    # =========================================================================
    op.create_table(
        "questionnaire",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("language_questionnaire", sa.String(length=50), nullable=True),
        sa.Column("question_number", sa.Integer(), nullable=True),
        sa.Column("questionnaire", sa.String(), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=True),
        sa.Column("key", sa.Integer(), nullable=True),
        sa.Column("question", sa.Text(), nullable=True),
        sa.Column("reverse", sa.Boolean(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_questionnaire_language_questionnaire",
        "questionnaire",
        ["language_questionnaire"],
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_questionnaire_question_number",
        "questionnaire",
        ["question_number"],
        schema=target_schema,
    )

    # =========================================================================
    # TABLE: experiment_request_metadata (with server-side timestamp)
    # =========================================================================
    op.create_table(
        "experiment_request_metadata",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "created",
            sa.DateTime(),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column("experiment_id", sa.Integer(), nullable=True),
        sa.Column("request_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("response_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("request_metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=target_schema,
    )
    op.create_index(
        f"ix_{target_schema}_exp_req_meta_exp_id",
        "experiment_request_metadata",
        ["experiment_id"],
        schema=target_schema,
    )


def downgrade() -> None:
    """Drop all tables and schema."""
    from alembic import context

    target_schema = None

    try:
        config = context.config
        if hasattr(config, "attributes") and config.attributes:
            target_schema = config.attributes.get("target_schema")
    except Exception:
        pass

    if not target_schema:
        try:
            from personas_backend.utils.config import ConfigManager

            cfg = ConfigManager()
            target_schema = cfg.pg_config.target_schema or cfg.pg_config.default_schema
        except Exception:
            pass

    if not target_schema:
        target_schema = "personality_trap"

    # Drop all tables in reverse order
    op.drop_table("experiment_request_metadata", schema=target_schema)
    op.drop_table("questionnaire", schema=target_schema)
    op.drop_table("eval_questionnaires", schema=target_schema)
    op.drop_table("experiments_list", schema=target_schema)
    op.drop_table("experiments_groups", schema=target_schema)
    op.drop_table("random_questionnaires", schema=target_schema)
    op.drop_table("reference_questionnaires", schema=target_schema)
    op.drop_table("personas", schema=target_schema)

    # Drop schema
    op.execute(f"DROP SCHEMA IF EXISTS {target_schema} CASCADE")
