"""Personality scores table (Table 4): EPQR-A scores with statistical testing.

This module provides functions to create Table 4 from the paper, which
compares EPQR-A personality scores between LLM-generated personas and
reference profiles.

Statistical Methodology:
- Dual-level testing: Individual (paired t-test) + Population
  (one-sample t-test)
- Individual-level: Compares same personalities (model vs reference)
- Population-level: Tests if mean differences differ from zero
- Effect sizes: Cohen's d for practical significance assessment

Significance Markers:
- Individual-level (paired t-test):
  * = p < 0.05
  † = p < 0.01
- Population-level (one-sample t-test):
  § = p < 0.05
  ¶ = p < 0.01
"""

from __future__ import annotations

from typing import Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import scipy.stats as stats  # type: ignore


def perform_enpl_tests(
    data: pd.DataFrame,
    reference_data: pd.DataFrame,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Perform dual-level statistical testing for EPQR-A scores.

    Implements the statistical methodology for Table 4, performing two
    complementary tests to assess whether LLM-generated personas exhibit
    personality traits consistent with their input personality profiles.

    Statistical Tests:
    ------------------
    1. INDIVIDUAL-LEVEL: Paired t-test
       - Compares scores for the SAME personalities (model vs reference)
       - Tests: "Do individual LLM personas match their reference profiles?"
       - Use case: Validates persona-level consistency
       - Markers: * (p<0.05), † (p<0.01)

    2. POPULATION-LEVEL: One-sample t-test on differences
       - Tests if mean difference across population differs from zero
       - Tests: "Does the model systematically over/under-represent traits?"
       - Use case: Detects systematic bias at population level
       - Markers: § (p<0.05), ¶ (p<0.01)

    Parameters:
    -----------
    data : pd.DataFrame
        Merged experiment data with columns:
        - model_clean: Clean model name (e.g., 'GPT-4o')
        - population_display: Population condition (Base/MaxN/MaxP)
        - population_mapped: Mapped population name
        - category: ENPL category
        - score: LLM experiment score
        - ref_score: Reference baseline score
        - personality_id: Personality identifier

    reference_data : pd.DataFrame
        Reference scores with columns:
        - personality_id: Personality identifier
        - category: ENPL category
        - ref_score: Reference score

    alpha : float, default=0.05
        Significance threshold for hypothesis testing

    Returns:
    --------
    pd.DataFrame
        Results containing:
        - model: Model name
        - population: Population condition
        - category: ENPL category
        - n_samples: Sample size
        - model_mean: Mean model score
        - model_std: Standard deviation of model scores
        - ref_mean: Mean reference score
        - mean_diff: Mean difference (model - reference)
        - 95%_CI_lower: Lower confidence interval bound
        - 95%_CI_upper: Upper confidence interval bound
        - t_stat_paired: t-statistic for paired test
        - p_value_paired: p-value for paired test
        - marker_individual: Significance marker for individual test
        - t_stat_pop: t-statistic for population test
        - p_value_pop: p-value for population test
        - marker_population: Significance marker for population test
        - significance_markers: Combined markers
        - cohens_d: Effect size (Cohen's d)

    Example:
    --------
    >>> results = perform_enpl_tests(merged_data, reference_data)
    >>> # Results include dual-level significance markers:
    >>> # '†§' means: individual p<0.01 AND population p<0.05
    """
    results = []

    # Get unique model-population combinations to test
    model_pops = (
        data[["model_clean", "population_mapped", "population_display"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Iterate through each model-population-category combination
    for _, row in model_pops.iterrows():
        model = row["model_clean"]
        population = row["population_display"]

        for category in ["E", "N", "P", "L"]:
            # Filter data for this specific combination
            subset = data[
                (data["model_clean"] == model)
                & (data["population_display"] == population)
                & (data["category"] == category)
            ].copy()

            # Skip if insufficient samples for reliable testing
            if len(subset) < 5:
                continue

            # ─────────────────────────────────────────────────────────────
            # TEST 1: Individual-level paired t-test
            # ─────────────────────────────────────────────────────────────
            # Compares model scores vs reference scores for SAME personalities
            # H0: Mean difference between paired observations equals zero
            # H1: Mean difference differs from zero
            t_stat_paired, p_value_paired = stats.ttest_rel(subset["score"], subset["ref_score"])

            # ─────────────────────────────────────────────────────────────
            # TEST 2: Population-level one-sample t-test on differences
            # ─────────────────────────────────────────────────────────────
            # Tests if mean difference across population differs from zero
            # H0: Population mean difference equals zero
            # H1: Population mean difference differs from zero
            differences = subset["score"] - subset["ref_score"]
            t_stat_pop, p_value_pop = stats.ttest_1samp(differences, 0)

            # ─────────────────────────────────────────────────────────────
            # Effect Size: Cohen's d for paired samples
            # ─────────────────────────────────────────────────────────────
            # Measures practical significance (standardized mean difference)
            # Interpretation: |d| < 0.2 (small), 0.5 (medium), 0.8 (large)
            mean_diff = differences.mean()
            std_diff = differences.std()
            d = mean_diff / std_diff if std_diff > 0 else 0

            # ─────────────────────────────────────────────────────────────
            # 95% Confidence Interval for mean difference
            # ─────────────────────────────────────────────────────────────
            ci_lower = mean_diff - 1.96 * std_diff / np.sqrt(len(subset))
            ci_upper = mean_diff + 1.96 * std_diff / np.sqrt(len(subset))

            # ─────────────────────────────────────────────────────────────
            # Assign significance markers based on p-values
            # ─────────────────────────────────────────────────────────────

            # Individual-level markers (paired t-test)
            if p_value_paired < 0.01:
                marker_individual = "†"  # Highly significant
            elif p_value_paired < 0.05:
                marker_individual = "*"  # Significant
            else:
                marker_individual = ""  # Not significant

            # Population-level markers (one-sample t-test)
            if p_value_pop < 0.01:
                marker_population = "¶"  # Highly significant
            elif p_value_pop < 0.05:
                marker_population = "§"  # Significant
            else:
                marker_population = ""  # Not significant

            # Combine markers for display (e.g., '†§' or '*¶')
            significance_markers = marker_individual + marker_population

            # ─────────────────────────────────────────────────────────────
            # Store results for this combination
            # ─────────────────────────────────────────────────────────────
            results.append(
                {
                    "model": model,
                    "population": population,
                    "category": category,
                    "n_samples": len(subset),
                    "model_mean": subset["score"].mean(),
                    "model_std": subset["score"].std(),
                    "ref_mean": subset["ref_score"].mean(),
                    "mean_diff": mean_diff,
                    "95%_CI_lower": ci_lower,
                    "95%_CI_upper": ci_upper,
                    "t_stat_paired": t_stat_paired,
                    "p_value_paired": p_value_paired,
                    "marker_individual": marker_individual,
                    "t_stat_pop": t_stat_pop,
                    "p_value_pop": p_value_pop,
                    "marker_population": marker_population,
                    "significance_markers": significance_markers,
                    "cohens_d": d,
                }
            )

    # Convert results to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["model", "population", "category"])

    return results_df


def format_enpl_scores(row: pd.Series) -> str:
    """Format EPQR-A scores as mean ± std with significance markers.

    Parameters:
    -----------
    row : pd.Series
        Row with model_mean, model_std, and significance_markers columns

    Returns:
    --------
    str
        Formatted string like "3.45 ± 1.23†§"

    Examples:
    ---------
    - "3.45 ± 1.23†§"  (both individual and population significant)
    - "4.12 ± 0.89*"   (only individual significant)
    - "2.98 ± 1.45"    (not significant)
    """
    mean_str = f"{row['model_mean']:.2f}"
    std_str = f"{row['model_std']:.2f}"
    markers = row["significance_markers"]
    return f"{mean_str} ± {std_str}{markers}"


def create_personality_table(
    test_results: pd.DataFrame,
    *,
    model_order: Optional[list[str]] = None,
    population_order: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Create publication-ready EPQR-A personality table (Table 4).

    Transforms statistical test results into a formatted table with:
    - Rows: Model × Population combinations
    - Columns: E, N, P, L personality dimensions
    - Values: "mean ± std" with significance markers

    Parameters:
    -----------
    test_results : pd.DataFrame
        Output from perform_enpl_tests() containing statistical results

    model_order : list[str], optional
        Preferred ordering for models. Default: GPT-4o, GPT-3.5, Claude-3.5-s,
        Llama3.2-3B, Llama3.1-70B

    population_order : list[str], optional
        Preferred ordering for populations. Default: Base, MaxN, MaxP

    Returns:
    --------
    pd.DataFrame
        Formatted table with rows (model, population) and columns (E, N, P, L)

    Example:
    --------
    >>> results = perform_enpl_tests(merged_data, reference_data)
    >>> table = create_personality_table(results)
    >>> table.to_csv('table4_personality_scores.csv')
    """
    # Apply formatting to create display column
    df = test_results.copy()
    df["content"] = df.apply(format_enpl_scores, axis=1)

    # Create pivot table in paper format: rows = model×population, cols = ENPL
    pivot_results = pd.pivot_table(
        df,
        values=["content"],
        index=["model", "population"],
        columns=["category"],
        aggfunc="first",  # Take first (only) value for each combination
    )

    # Flatten MultiIndex column names
    pivot_results.columns = [f"{cat}" for _, cat in pivot_results.columns]
    pivot_results = pivot_results.reset_index()

    # Apply consistent ordering for publication
    if model_order is None:
        model_order = [
            "GPT-4o",
            "GPT-3.5",
            "Claude-3.5-s",
            "Llama3.2-3B",
            "Llama3.1-70B",
        ]

    if population_order is None:
        population_order = ["Base", "MaxN", "MaxP"]

    enpl_order = ["E", "N", "P", "L"]

    # Convert to categorical types for proper sorting
    pivot_results["model"] = pd.Categorical(
        pivot_results["model"], categories=model_order, ordered=True
    )
    pivot_results["population"] = pd.Categorical(
        pivot_results["population"], categories=population_order, ordered=True
    )

    # Sort: first by model, then by population within each model
    pivot_results = pivot_results.sort_values(["model", "population"])

    # Reorder columns: model, population, then E, N, P, L
    column_order = ["model", "population"] + enpl_order
    available_cols = [col for col in column_order if col in pivot_results.columns]
    pivot_results = pivot_results[available_cols]

    return pivot_results


def compute_baseline_statistics(
    scores_data: pd.DataFrame,
    score_column: str = "ref_score",
    *,
    name: str = "Reference",
) -> pd.DataFrame:
    """Compute mean ± std statistics for baseline (reference or random) scores.

    Parameters:
    -----------
    scores_data : pd.DataFrame
        DataFrame with columns: category, {score_column}

    score_column : str, default='ref_score'
        Name of the score column to aggregate

    name : str, default='Reference'
        Name for the baseline type (e.g., 'Reference', 'Random')

    Returns:
    --------
    pd.DataFrame
        Summary table with columns:
        - Category: E, N, P, L
        - {name} Score (Mean ± SD): Formatted scores

    Example:
    --------
    >>> ref_summary = compute_baseline_statistics(
    ...     reference_data, name='Reference'
    ... )
    >>> random_summary = compute_baseline_statistics(random_data,
    ...     score_column='random_score', name='Random')
    """
    # Aggregate by category
    summary = scores_data.groupby(["category"]).agg({score_column: ["mean", "std"]}).reset_index()

    # Round and format
    summary[score_column] = summary[score_column].round(2)
    summary["formatted"] = (
        summary[score_column]["mean"].astype(str) + " ± " + summary[score_column]["std"].astype(str)
    )

    # Select display columns
    result = summary[["category", "formatted"]].copy()
    result.columns = ["Category", f"{name} Score (Mean ± SD)"]

    return result
