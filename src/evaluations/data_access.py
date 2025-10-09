"""Data access helpers for evaluations.

Provides simple functions to load population and questionnaire data into
Pandas DataFrames. Prefer passing `schema` explicitly to avoid hardcoding.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd  # type: ignore
from sqlalchemy.engine.base import Connection  # type: ignore


def load_population(
    conn: Connection,
    *,
    schema: str,
    table: str = "personas",
    drop_suffix_original: bool = True,
) -> pd.DataFrame:
    """Load population view/table as DataFrame.

    Args:
        conn: SQLAlchemy connection/engine.
        schema: PostgreSQL schema to use.
        table: Table or view name.
        drop_suffix_original: Drop columns ending with "_original" if present.

    Returns:
        DataFrame with population rows.
    """
    df = pd.read_sql_table(table, con=conn, schema=str(schema))
    if drop_suffix_original:
        original_cols = [c for c in df.columns if c.endswith("_original")]
        if original_cols:
            df = df.drop(columns=original_cols)
    return df


def load_reference_questionnaires(conn: Connection, *, schema: str) -> pd.DataFrame:
    """Load reference questionnaire data."""

    query = f"""
    SELECT *, CASE WHEN key = answer THEN 1 ELSE 0 END as ref_eval
    FROM {schema}.reference_questionnaires
    """
    return pd.read_sql(query, con=conn)


def load_questionnaire_experiments(
    conn: Connection,
    *,
    schema: str,
    experiment_groups: Optional[list[int]] = None,
    questionnaires: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load questionnaire evaluation experiments from experiments_evals view.

    Note: This function loads questionnaire evaluation data only.
    For demographic analysis (Tables 1-3), use load_population() instead.

    Args:
        conn: Database connection
        schema: Schema name (e.g., 'personality_trap')
        experiment_groups: List of experiment group IDs to filter
        questionnaires: List of questionnaire types to filter
                       (e.g., ['epqra', 'bigfive']).
                       If None, defaults to ['epqra']

    Returns:
        DataFrame with questionnaire evaluation results from view
    """
    # Default to epqra if no questionnaires specified
    if questionnaires is None:
        questionnaires = ["epqra"]

    # Build WHERE clause for questionnaires
    if len(questionnaires) == 1:
        questionnaire_filter = f"questionnaire = '{questionnaires[0]}'"
    else:
        questionnaire_list = "', '".join(questionnaires)
        questionnaire_filter = f"questionnaire IN ('{questionnaire_list}')"

    base_query = f"""
    SELECT experiments_group_id, model_provider, model, questionnaire,
           population, personality_id, repeated, experiment_id,
           question_number, answer, category, key, eval,
           model_clean, population_mapped, population_display
    FROM {schema}.experiments_evals
    WHERE {questionnaire_filter}
    """

    if experiment_groups:
        exp_groups_str = ",".join(map(str, experiment_groups))
        base_query += f" AND experiments_group_id IN ({exp_groups_str})"

    print(f"Loading questionnaire data from {schema}.experiments_evals...")
    df = pd.read_sql(base_query, con=conn)
    print(f"Loaded {len(df)} questionnaire records")
    print(f"Models: {df['model_clean'].unique()}")
    print(f"Populations: {df['population_mapped'].unique()}")

    return df


def load_reference_enpl_scores(conn: Connection, *, schema: str) -> pd.DataFrame:
    """Load reference ENPL scores from reference_questionnaires table.

    Computes individual ENPL scores by summing correct answers per
    personality_id and category.

    Args:
        conn: Database connection
        schema: Schema name (e.g., 'personality_trap')

    Returns:
        DataFrame with columns: personality_id, category, ref_score
        where ref_score is the total score for that category
    """
    query = f"""
    SELECT personality_id, category,
           SUM(CASE WHEN key = answer THEN 1 ELSE 0 END) as ref_score
    FROM {schema}.reference_questionnaires
    GROUP BY personality_id, category
    """
    return pd.read_sql(query, con=conn)


def load_random_enpl_scores(conn: Connection, *, schema: str) -> pd.DataFrame:
    """Load random ENPL scores from random_questionnaires table.

    Computes individual ENPL scores by summing correct answers per
    personality_id and category from the randomly generated baseline.

    Args:
        conn: Database connection
        schema: Schema name (e.g., 'personality_trap')

    Returns:
        DataFrame with columns: personality_id, category, random_score
        where random_score is the total score for that category
    """
    query = f"""
    SELECT personality_id, category,
           SUM(CASE WHEN key = answer THEN 1 ELSE 0 END) as random_score
    FROM {schema}.random_questionnaires
    GROUP BY personality_id, category
    """
    return pd.read_sql(query, con=conn)


def load_experiment_enpl_scores(
    conn: Connection,
    *,
    schema: str,
    experiment_groups: Optional[list[int]] = None,
    questionnaires: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Load experiment ENPL scores from experiments_evals view.

    Computes ENPL scores by summing correct answers (eval) per experiment,
    grouped by model, population, personality, and category.

    Args:
        conn: Database connection
        schema: Schema name (e.g., 'personality_trap')
        experiment_groups: List of experiment group IDs to filter
        questionnaires: List of questionnaire types to filter
                       (e.g., ['epqra', 'bigfive']).
                       If None, defaults to ['epqra']

    Returns:
        DataFrame with columns:
        - experiments_group_id: Experiment group ID
        - model_clean: Clean model name
        - population_mapped: Mapped population name
        - population_display: Display name (Base/MaxN/MaxP)
        - experiment_id: Experiment ID
        - personality_id: Personality ID
        - repeated: Repetition number
        - category: ENPL category (E, N, P, L)
        - score: Sum of correct answers for this category
    """
    # Default to epqra if no questionnaires specified
    if questionnaires is None:
        questionnaires = ["epqra"]

    # Build WHERE clause for questionnaires
    if len(questionnaires) == 1:
        questionnaire_filter = f"questionnaire = '{questionnaires[0]}'"
    else:
        questionnaire_list = "', '".join(questionnaires)
        questionnaire_filter = f"questionnaire IN ('{questionnaire_list}')"

    base_query = f"""
    SELECT
        experiments_group_id,
        model_clean,
        population_mapped,
        population_display,
        experiment_id,
        personality_id,
        repeated,
        category,
        SUM(eval) as score
    FROM {schema}.experiments_evals
    WHERE {questionnaire_filter}
    """

    if experiment_groups:
        exp_groups_str = ",".join(map(str, experiment_groups))
        base_query += f" AND experiments_group_id IN ({exp_groups_str})"

    base_query += """
    GROUP BY
        experiments_group_id,
        model_clean,
        population_mapped,
        population_display,
        experiment_id,
        personality_id,
        repeated,
        category
    ORDER BY
        experiments_group_id,
        model_clean,
        population_mapped,
        personality_id,
        repeated,
        category
    """

    print(f"Loading experiment ENPL scores from {schema}.experiments_evals...")
    df = pd.read_sql(base_query, con=conn)
    print(f"Loaded {len(df)} experiment score records")
    print(f"Models: {df['model_clean'].unique()}")
    print(f"Populations: {df['population_mapped'].unique()}")
    print(f"Categories: {sorted(df['category'].unique())}")

    return df
