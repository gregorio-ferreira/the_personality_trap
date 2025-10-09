"""Accuracy and Error Metrics Tables (Tables A6 & A7).

This module provides functions to calculate and format accuracy metrics
(accuracy, precision, recall, specificity) and error metrics (MAE, RMSE)
for Tables A6 and A7 in the paper appendix.

Tables:
- Table A6: Base population accuracy metrics vs reference
- Table A7: MaxN/MaxP populations accuracy metrics vs reference
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from personas_backend.utils.formatting import custom_format


def calculate_accuracy_metrics(data: pd.DataFrame, group_by: List[str]) -> pd.DataFrame:
    """Calculate accuracy, precision, recall, and specificity with custom formatting.

    This matches the exact implementation from the original paper notebook.

    Args:
        data: DataFrame with binary classification columns (tp, tn, pred_pos, act_pos, etc.)
        group_by: List of columns to group by (e.g., ['model_clean', 'category'])

    Returns:
        DataFrame with calculated and formatted metrics (as strings)

    Example:
        >>> metrics = calculate_accuracy_metrics(
        ...     accuracy_df,
        ...     group_by=['model_clean', 'category']
        ... )
    """
    metrics = (
        data.groupby(group_by, as_index=False)
        .agg(
            accuracy=pd.NamedAgg(column="equal", aggfunc="mean"),
            tp=pd.NamedAgg(column="tp", aggfunc="sum"),
            pred_pos=pd.NamedAgg(column="pred_pos", aggfunc="sum"),
            act_pos=pd.NamedAgg(column="act_pos", aggfunc="sum"),
            tn=pd.NamedAgg(column="tn", aggfunc="sum"),
            pred_neg=pd.NamedAgg(column="pred_neg", aggfunc="sum"),
            act_neg=pd.NamedAgg(column="act_neg", aggfunc="sum"),
        )
        .reset_index(drop=True)
        .sort_values(by="accuracy", ascending=True)
    )

    metrics["accuracy"] = 100 * metrics["accuracy"]

    # Calculate precision (if denominator is zero, return 0)
    metrics["precision"] = metrics.apply(
        lambda row: 100 * row["tp"] / row["pred_pos"] if row["pred_pos"] > 0 else 0,
        axis=1,
    )

    # Calculate recall (if denominator is zero, return 0)
    metrics["recall"] = metrics.apply(
        lambda row: 100 * row["tp"] / row["act_pos"] if row["act_pos"] > 0 else 0,
        axis=1,
    )

    # Calculate specificity (if denominator is zero, return 0)
    metrics["specificity"] = metrics.apply(
        lambda row: 100 * row["tn"] / row["act_neg"] if row["act_neg"] > 0 else 0,
        axis=1,
    )

    # Format values using custom_format (matches paper formatting)
    metrics["accuracy"] = metrics["accuracy"].apply(custom_format)
    metrics["tp"] = metrics["tp"].apply(custom_format)
    metrics["pred_pos"] = metrics["pred_pos"].apply(custom_format)
    metrics["act_pos"] = metrics["act_pos"].apply(custom_format)
    metrics["precision"] = metrics["precision"].apply(custom_format)
    metrics["recall"] = metrics["recall"].apply(custom_format)
    metrics["specificity"] = metrics["specificity"].apply(custom_format)

    return metrics


def compute_error_metrics(data: pd.DataFrame, group_by_base: List[str]) -> pd.DataFrame:
    """Compute MAE and RMSE error metrics.

    Calculates scale-level errors by summing answers per personality and category,
    then computing mean absolute error and root mean squared error.

    Args:
        data: DataFrame with answer_ref and answer_exp columns
        group_by_base: Base grouping columns (e.g., ['model_clean', 'population_display'])

    Returns:
        DataFrame with MAE and RMSE metrics

    Example:
        >>> errors = compute_error_metrics(
        ...     accuracy_df,
        ...     group_by_base=['model_clean']
        ... )
    """
    # First group by personality to get scale scores
    enlp_ref_diff = data.groupby(
        group_by_base + ["personality_id", "category"], as_index=False
    ).agg(
        answer_ref=pd.NamedAgg(column="answer_ref", aggfunc="sum"),
        answer_exp=pd.NamedAgg(column="answer_exp", aggfunc="sum"),
    )

    enlp_ref_diff["diff"] = enlp_ref_diff["answer_exp"] - enlp_ref_diff["answer_ref"]
    enlp_ref_diff["absolute_error"] = enlp_ref_diff["diff"].abs()
    enlp_ref_diff["squared_error"] = enlp_ref_diff["diff"] ** 2

    # Aggregate error metrics
    errors_df = enlp_ref_diff.groupby(group_by_base + ["category"], as_index=False).agg(
        mae=pd.NamedAgg(column="absolute_error", aggfunc="mean"),
        mse=pd.NamedAgg(column="squared_error", aggfunc="mean"),
    )

    # Calculate RMSE
    errors_df["rmse"] = np.sqrt(errors_df["mse"])
    errors_df = errors_df.drop(columns=["mse"])

    return errors_df


def format_metrics_table(
    metrics_df: pd.DataFrame,
    model_col: str = "model_clean",
    population_col: Optional[str] = "population_display",
) -> pd.DataFrame:
    """Format metrics into publication-ready table.

    Sorts by predefined model, population, and category orders.

    Args:
        metrics_df: DataFrame with calculated metrics
        model_col: Column name for model
        population_col: Column name for population (None if not present)

    Returns:
        Formatted and sorted DataFrame

    Example:
        >>> formatted = format_metrics_table(
        ...     metrics,
        ...     model_col='model_clean',
        ...     population_col=None  # For Table A6 (Base only)
        ... )
    """
    # Define display order
    model_order = ["GPT-4o", "GPT-3.5", "Claude-3.5-s", "Llama3.1-70B", "Llama3.2-3B"]
    population_order = ["Base", "MaxN", "MaxP"]
    category_order = ["E", "N", "P", "L"]

    # Create categorical columns for sorting
    result = metrics_df.copy()

    if model_col in result.columns:
        result["_model_order"] = pd.Categorical(
            result[model_col], categories=model_order, ordered=True
        )

    if population_col in result.columns and population_col is not None:
        result["_pop_order"] = pd.Categorical(
            result[population_col], categories=population_order, ordered=True
        )

    if "category" in result.columns:
        result["_cat_order"] = pd.Categorical(
            result["category"], categories=category_order, ordered=True
        )

    # Sort by ordering columns
    sort_cols = [c for c in ["_model_order", "_pop_order", "_cat_order"] if c in result.columns]
    if sort_cols:
        result = result.sort_values(by=sort_cols)

    # Drop temporary ordering columns
    result = result.drop(columns=[c for c in result.columns if c.startswith("_")], errors="ignore")

    return result.reset_index(drop=True)


def prepare_accuracy_data(experiment_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare merged data with binary classification metrics.

    Merges experiment and reference data, then calculates equality,
    true positives, true negatives, and other confusion matrix components.

    Args:
        experiment_df: DataFrame with experiment answers (must have columns:
            model_clean, population_display, personality_id, question_number,
            answer, category, key)
        reference_df: DataFrame with reference answers (must have columns:
            personality_id, question_number, answer, key)

    Returns:
        DataFrame with binary classification columns added:
            equal, tp, tn, pred_pos, act_pos, pred_neg, act_neg

    Example:
        >>> accuracy_df = prepare_accuracy_data(epqra_data, reference_data)
        >>> # Now ready for calculate_accuracy_metrics()
    """
    # Select relevant columns
    exp_cols = [
        "experiments_group_id",
        "model_clean",
        "population_display",
        "personality_id",
        "experiment_id",
        "question_number",
        "answer",
        "category",
        "key",
    ]

    ref_cols = ["personality_id", "question_number", "answer", "key"]

    # Merge experiment data with reference data
    accuracy_df = pd.merge(
        experiment_df[exp_cols],
        reference_df[ref_cols],
        on=["personality_id", "question_number"],
        how="left",
        suffixes=("_exp", "_ref"),
    )

    # Calculate equality (simple accuracy)
    accuracy_df["equal"] = accuracy_df["answer_exp"] == accuracy_df["answer_ref"]

    # Compute true positives, predicted positives, and actual positives
    accuracy_df["tp"] = (
        (accuracy_df["answer_exp"] == True) & (accuracy_df["answer_ref"] == True)
    ).astype(int)
    accuracy_df["pred_pos"] = (accuracy_df["answer_exp"] == True).astype(int)
    accuracy_df["act_pos"] = (accuracy_df["answer_ref"] == True).astype(int)

    # Compute true negatives, predicted negatives, and actual negatives
    accuracy_df["tn"] = (
        (accuracy_df["answer_exp"] == False) & (accuracy_df["answer_ref"] == False)
    ).astype(int)
    accuracy_df["pred_neg"] = (accuracy_df["answer_exp"] == False).astype(int)
    accuracy_df["act_neg"] = (accuracy_df["answer_ref"] == False).astype(int)

    return accuracy_df


def create_table_a6(experiment_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Create Table A6: Base population accuracy and error metrics.

    Args:
        experiment_df: DataFrame with EPQR-A experiment data
        reference_df: DataFrame with reference questionnaire data

    Returns:
        Formatted DataFrame for Table A6 with columns:
            model_clean, category, accuracy, precision, recall, specificity, mae, rmse

    Example:
        >>> table_a6 = create_table_a6(epqra_data, reference_data)
        >>> table_a6.to_csv('table_a6_base_accuracy.csv', index=False)
    """
    # Prepare data with binary classification metrics
    accuracy_df = prepare_accuracy_data(experiment_df, reference_df)

    # Filter to Base population only
    base_data = accuracy_df[accuracy_df["population_display"] == "Base"].copy()

    # Calculate accuracy metrics
    base_metrics = calculate_accuracy_metrics(base_data, group_by=["model_clean", "category"])

    # Calculate error metrics (MAE, RMSE)
    base_errors = compute_error_metrics(base_data, group_by_base=["model_clean"])

    # Merge metrics with errors
    table_a6 = pd.merge(base_metrics, base_errors, on=["model_clean", "category"], how="left")

    # Format for display
    table_a6 = format_metrics_table(table_a6, model_col="model_clean", population_col=None)

    # Select key columns for display
    display_cols = [
        "model_clean",
        "category",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "mae",
        "rmse",
    ]
    return table_a6[display_cols]


def create_table_a7(experiment_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """Create Table A7: MaxN/MaxP populations accuracy and error metrics.

    Args:
        experiment_df: DataFrame with EPQR-A experiment data
        reference_df: DataFrame with reference questionnaire data

    Returns:
        Formatted DataFrame for Table A7 with columns:
            model_clean, population_display, category, accuracy, precision,
            recall, specificity, mae, rmse

    Example:
        >>> table_a7 = create_table_a7(epqra_data, reference_data)
        >>> table_a7.to_csv('table_a7_borderline_accuracy.csv', index=False)
    """
    # Prepare data with binary classification metrics
    accuracy_df = prepare_accuracy_data(experiment_df, reference_df)

    # Filter to MaxN and MaxP populations
    borderline_data = accuracy_df[accuracy_df["population_display"].isin(["MaxN", "MaxP"])].copy()

    # Calculate accuracy metrics
    borderline_metrics = calculate_accuracy_metrics(
        borderline_data, group_by=["model_clean", "population_display", "category"]
    )

    # Calculate error metrics (MAE, RMSE)
    borderline_errors = compute_error_metrics(
        borderline_data, group_by_base=["model_clean", "population_display"]
    )

    # Merge metrics with errors
    table_a7 = pd.merge(
        borderline_metrics,
        borderline_errors,
        on=["model_clean", "population_display", "category"],
        how="left",
    )

    # Format for display
    table_a7 = format_metrics_table(
        table_a7, model_col="model_clean", population_col="population_display"
    )

    # Select key columns for display
    display_cols = [
        "model_clean",
        "population_display",
        "category",
        "accuracy",
        "precision",
        "recall",
        "specificity",
        "mae",
        "rmse",
    ]
    return table_a7[display_cols]
