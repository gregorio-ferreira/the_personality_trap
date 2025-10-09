"""Cronbach's Alpha table (Table 6): Internal consistency analysis.

This module provides functions to create Table 6 from the paper, which
measures the internal consistency (reliability) of the EPQR-A questionnaire
across different LLM models and population conditions using Cronbach's Alpha.

Cronbach's Alpha (α):
- α > 0.9: Excellent consistency
- α > 0.8: Good consistency
- α > 0.7: Acceptable consistency
- α > 0.6: Questionable consistency
- α < 0.6: Poor consistency

The analysis includes:
1. LLM-generated persona responses
2. Reference baseline (input personality profiles)
3. Random baseline (null hypothesis control)
"""

from __future__ import annotations

from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import pingouin as pg  # type: ignore


def calculate_cronbach_alpha(
    data_frame: pd.DataFrame,
    question_col: str,
    category_col: str,
    eval_col: str,
    experiment_col: str,
    *,
    model_col: Optional[str] = None,
    group_by: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Calculate Cronbach's Alpha for questionnaire responses.

    This function computes Cronbach's Alpha to measure the internal
    consistency of questionnaire responses. It pivots the data into a
    wide format (experiments × questions) and calculates alpha for each
    category and grouping combination.

    Args:
        data_frame: DataFrame containing questionnaire responses
        question_col: Column name for question identifiers
        category_col: Column name for category identifiers (E, N, P, L)
        eval_col: Column name for evaluation values (0 or 1)
        experiment_col: Column name for experiment/personality identifiers
        model_col: Column name for model identifiers (optional, deprecated)
        group_by: Columns to group by before calculating alpha
                 (e.g., ['model_clean', 'population_display'])

    Returns:
        DataFrame with Cronbach's Alpha values for each group and category.
        Columns depend on group_by parameter plus category columns
        (E, N, P, L).

    Example:
        >>> alpha_results = calculate_cronbach_alpha(
        ...     data_frame=epqra_data,
        ...     question_col='question_number',
        ...     category_col='category',
        ...     eval_col='eval',
        ...     experiment_col='experiment_id',
        ...     group_by=['model_clean', 'population_display']
        ... )
        >>> print(alpha_results[
        ...     ['model_clean', 'population_display', 'E', 'N', 'P', 'L']
        ... ])
    """
    # Convert eval to integer if needed
    data_frame = data_frame.copy()
    if data_frame[eval_col].dtype != "int":
        data_frame[eval_col] = data_frame[eval_col].astype(int)

    # Format question numbers to strings for consistent column naming
    if data_frame[question_col].dtype != "str":
        data_frame[question_col] = data_frame[question_col].apply(lambda x: f"q_{int(x):02d}")

    # Create pivot table for Cronbach's Alpha calculation
    # Rows: experiments (and grouping columns if specified)
    # Columns: questions
    pivot_columns = [experiment_col]
    if group_by:
        pivot_columns.extend(group_by)

    # Create one column per question
    pivot_df = pd.pivot_table(
        data_frame,
        values=eval_col,
        index=pivot_columns,
        columns=[question_col],
        aggfunc="first",
    )

    # Reset index to make the pivoted columns accessible
    pivot_df.reset_index(inplace=True)

    # Initialize results dictionary
    result_dict = {}

    # Get unique categories
    categories = data_frame[category_col].unique()

    # Define which questions belong to each category
    category_questions = {}
    for cat in categories:
        # Get question numbers for this category
        questions = data_frame[data_frame[category_col] == cat][question_col].unique()
        category_questions[cat] = questions

    if group_by:
        # Group by the specified columns and calculate alpha for each group
        for group_name, group_df in pivot_df.groupby(group_by):
            if not isinstance(group_name, tuple):
                group_name = (group_name,)

            group_key = tuple(zip(group_by, group_name, strict=False))
            result_dict[group_key] = {}

            # Calculate alpha for each category
            for cat, questions in category_questions.items():
                # Filter only questions that exist in the pivot table
                available_questions = [q for q in questions if q in pivot_df.columns]

                if len(available_questions) >= 2:  # Need at least 2 items for alpha calculation
                    # Extract data for alpha calculation
                    alpha_data = group_df[available_questions]

                    # Skip if all values are the same (no variance)
                    if alpha_data.std().sum() > 0:
                        alpha = pg.cronbach_alpha(data=alpha_data)[0]
                        result_dict[group_key][cat] = alpha
                    else:
                        result_dict[group_key][cat] = np.nan
                else:
                    result_dict[group_key][cat] = np.nan
    else:
        # Calculate alpha for each category across the entire dataset
        result_dict["all"] = {}

        for cat, questions in category_questions.items():
            # Filter only questions that exist in the pivot table
            available_questions = [q for q in questions if q in pivot_df.columns]

            if len(available_questions) >= 2:  # Need at least 2 items for alpha calculation
                # Extract data for alpha calculation
                alpha_data = pivot_df[available_questions]

                # Skip if all values are the same (no variance)
                if alpha_data.std().sum() > 0:
                    alpha = pg.cronbach_alpha(data=alpha_data)[0]
                    result_dict["all"][cat] = alpha
                else:
                    result_dict["all"][cat] = np.nan
            else:
                result_dict["all"][cat] = np.nan

    # Convert results to DataFrame
    if group_by:
        rows = []
        for group_key, category_alphas in result_dict.items():
            row = dict(group_key)
            row.update(category_alphas)
            rows.append(row)

        result_df = pd.DataFrame(rows)
    else:
        result_df = pd.DataFrame([result_dict["all"]])

    return result_df.round(2)


def create_cronbach_table(
    alpha_results: pd.DataFrame,
    *,
    model_col: str = "model_clean",
    population_col: str = "population_display",
    model_order: Optional[list[str]] = None,
    population_order: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Format Cronbach's Alpha results into publication-ready table.

    Sorts and formats the alpha results table with proper ordering of
    models and populations for presentation.

    Args:
        alpha_results: DataFrame from calculate_cronbach_alpha()
        model_col: Column name containing model identifiers
        population_col: Column name containing population identifiers
        model_order: Desired order of models in output table.
                    Default: ['GPT-4o', 'GPT-3.5', 'Claude-3.5-s',
                             'Llama3.1-70B', 'Llama3.2-3B']
        population_order: Desired order of populations in output table.
                         Default: ['Base', 'MaxN', 'MaxP']

    Returns:
        Sorted and formatted DataFrame ready for publication.

    Example:
        >>> table6 = create_cronbach_table(
        ...     alpha_results,
        ...     model_col='model_clean',
        ...     population_col='population_display'
        ... )
        >>> display(table6)
    """
    # Default orderings
    if model_order is None:
        model_order = [
            "GPT-4o",
            "GPT-3.5",
            "Claude-3.5-s",
            "Llama3.1-70B",
            "Llama3.2-3B",
        ]

    if population_order is None:
        population_order = ["Base", "MaxN", "MaxP"]

    # Create copy to avoid modifying input
    result = alpha_results.copy()

    # Create categorical columns for sorting
    result["_model_order"] = pd.Categorical(result[model_col], categories=model_order, ordered=True)
    result["_population_order"] = pd.Categorical(
        result[population_col], categories=population_order, ordered=True
    )

    # Sort by model and population
    result = result.sort_values(by=["_model_order", "_population_order"])

    # Drop temporary ordering columns
    result = result.drop(columns=["_model_order", "_population_order"])

    # Select and order final columns
    result = result[[model_col, population_col, "E", "N", "P", "L"]].reset_index(drop=True)

    return result


def compute_alpha_statistics(
    alpha_table: pd.DataFrame,
    categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute summary statistics for Cronbach's Alpha values.

    Calculates descriptive statistics (count, mean, std, min, max, etc.)
    for each category across all model-population combinations.

    Args:
        alpha_table: Formatted table from create_cronbach_table()
        categories: List of category column names to analyze.
                   Default: ['E', 'N', 'P', 'L']

    Returns:
        DataFrame with summary statistics for each category.

    Example:
        >>> stats = compute_alpha_statistics(table6)
        >>> print(stats)
    """
    if categories is None:
        categories = ["E", "N", "P", "L"]

    return alpha_table[categories].describe().round(2)


def compute_alpha_quality_distribution(
    alpha_table: pd.DataFrame,
    categories: Optional[list[str]] = None,
) -> dict[str, int]:
    """Count alpha values by quality level.

    Categorizes all alpha values into quality levels based on standard
    thresholds and returns counts for each level.

    Quality Levels:
    - Excellent: α ≥ 0.9
    - Good: 0.8 ≤ α < 0.9
    - Acceptable: 0.7 ≤ α < 0.8
    - Questionable: 0.6 ≤ α < 0.7
    - Poor: α < 0.6

    Args:
        alpha_table: Formatted table from create_cronbach_table()
        categories: List of category column names to analyze.
                   Default: ['E', 'N', 'P', 'L']

    Returns:
        Dictionary with counts for each quality level.

    Example:
        >>> quality = compute_alpha_quality_distribution(table6)
        >>> print(f"Excellent: {quality['excellent']}")
    """
    if categories is None:
        categories = ["E", "N", "P", "L"]

    # Flatten all alpha values
    all_alphas = alpha_table[categories].values.flatten()

    # Count by quality level
    return {
        "excellent": int((all_alphas >= 0.9).sum()),
        "good": int(((all_alphas >= 0.8) & (all_alphas < 0.9)).sum()),
        "acceptable": int(((all_alphas >= 0.7) & (all_alphas < 0.8)).sum()),
        "questionable": int(((all_alphas >= 0.6) & (all_alphas < 0.7)).sum()),
        "poor": int((all_alphas < 0.6).sum()),
        "total": len(all_alphas),
    }


def create_baseline_comparison(
    llm_alpha: pd.DataFrame,
    reference_alpha: pd.DataFrame,
    random_alpha: Optional[pd.DataFrame] = None,
    categories: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Create comparison table of alpha values across data sources.

    Compares Cronbach's Alpha between:
    - LLM experiments (mean across all model-population combinations)
    - Reference baseline (input personality profiles)
    - Random baseline (null hypothesis control)

    Args:
        llm_alpha: Formatted LLM results from create_cronbach_table()
        reference_alpha: Reference baseline results from
            calculate_cronbach_alpha()
        random_alpha: Random baseline results (optional)
        categories: List of category columns. Default: ['E', 'N', 'P', 'L']

    Returns:
        Comparison DataFrame with one row per data source.

    Example:
        >>> comparison = create_baseline_comparison(
        ...     table6, reference_alpha, random_alpha
        ... )
        >>> display(comparison)
    """
    if categories is None:
        categories = ["E", "N", "P", "L"]

    # Calculate mean alpha values for LLM experiments
    llm_mean = llm_alpha[categories].mean().round(2)

    # Build comparison data
    comparison_data = {
        "Source": ["LLM Experiments (Mean)", "Reference Baseline"],
    }
    for cat in categories:
        comparison_data[cat] = [
            llm_mean[cat],
            reference_alpha[cat].values[0],
        ]

    # Add random baseline if available
    if random_alpha is not None:
        comparison_data["Source"].append("Random Baseline")
        for cat in categories:
            comparison_data[cat].append(random_alpha[cat].values[0])

    return pd.DataFrame(comparison_data)
