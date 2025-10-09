"""Demographics tables (Tables 1–3): stats by condition for selected models.

This module computes, per model and condition (Base/MaxN/MaxP), the mean±std of
selected demographic proportions across replicate samples and marks significant
differences vs Base using a z-test for proportions (binary demographic analysis).

Binary Demographic Analysis Approach:
- Each demographic value is treated as an independent binary variable (0/1)
- All repetitions are combined per population (concatenated data)
- Z-test for proportions compares binary outcomes between populations
- Edge cases (zero variance, empty groups) are handled appropriately

Use the `models` parameter to select which models to include (e.g., GPT, Llama,
Claude).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

try:  # prefer scipy if present
    from scipy.stats import ttest_ind  # type: ignore
except ImportError:  # pragma: no cover
    ttest_ind = None  # type: ignore[assignment]

try:
    import pingouin as pg  # type: ignore
except ImportError:  # pragma: no cover
    pg = None  # type: ignore[assignment]

try:
    from statsmodels.stats.proportion import proportions_ztest  # type: ignore
except ImportError:  # pragma: no cover
    proportions_ztest = None  # type: ignore[assignment]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    title: str


# Order matches paper Table 1 groups and labels
METRICS: List[MetricSpec] = [
    # Gender
    MetricSpec("gender_female", "Females"),
    MetricSpec("gender_male", "Male"),
    MetricSpec("gender_non_binary", "Non-bin."),
    MetricSpec("gender_other", "Other"),
    # Political Orientation
    MetricSpec("pol_centre", "Centre"),
    MetricSpec("pol_conservative", "Con."),
    MetricSpec("pol_progressive", "Prog."),
    MetricSpec("pol_other", "Others"),
    # Race
    MetricSpec("race_asia", "Asian"),
    MetricSpec("race_black", "Black"),
    MetricSpec("race_latin", "Latin"),
    MetricSpec("race_white", "White"),
    MetricSpec("race_other", "Other"),
    # Religious Belief
    MetricSpec("relig_christian", "Christian"),
    MetricSpec("relig_agnostic", "Agnostic"),
    MetricSpec("relig_atheist", "Atheist"),
    MetricSpec("relig_other", "Others"),
    # Sexual Orientation
    MetricSpec("sex_hetero", "Hetero."),
    MetricSpec("sex_lgbtq", "LGBTQ+"),
    MetricSpec("sex_unspecified", "Unspe."),
]


def _label_model(model: str) -> str | None:
    m = (model or "").lower()
    if "gpt-4o" in m or "gpt4o" in m:
        return "GPT-4o"
    if "gpt-3.5" in m or "gpt35" in m or "gpt-3.5-turbo" in m:
        return "GPT-3.5"
    # Llama models - handle the actual model names from data
    if "llama323b" in m or ("llama" in m and "323b" in m):
        return "Llama3.2-3B"
    if "llama3170b" in m or ("llama" in m and "3170b" in m):
        return "Llama3.1-70B"
    # Also handle more general Llama patterns
    if "llama" in m and ("3.2" in m or "3_2" in m):
        # 3.2 3B variant
        if "3b" in m or "3-b" in m or "3 b" in m:
            return "Llama3.2-3B"
        return "Llama3.2-3B"
    if "llama" in m and ("3.1" in m or "3_1" in m) and ("70" in m or "70b" in m):
        return "Llama3.1-70B"
    # Claude 3.5 Sonnet variants
    # Be generous about version tokens (e.g., 'claude-3-5-sonnet-20240620')
    if ("claude35sonnet" in m) or ("claude" in m and ("sonnet" in m or "son" in m)):
        return "Claude-3.5-s"
    return None


def _label_condition(population: str) -> str | None:
    p = (population or "").lower()
    # Normalize separators to catch variants like max-n, max_n, max n
    norm = p.replace("-", " ").replace("_", " ")
    if ("maxn" in p) or ("max n" in norm):
        return "MaxN"
    if ("maxp" in p) or ("max p" in norm) or ("prob" in p):
        return "MaxP"
    # Some datasets label the numeric-constraint variant simply as "Max"
    tokens = norm.split()
    if "max" in tokens:
        return "MaxN"
    # Base: be robust to naming differences
    if "base" in p or "generated" in p or "generated_" in p or "spain826" in p or "baseline" in p:
        return "Base"
    return None


def _category_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Produce boolean flags for all categories used in Table 1.

    Assumes categorical columns: gender, political_orientation, race,
    religious_belief, sexual_orientation. Missing columns are treated as empty.
    """

    def to_low(series: pd.Series) -> pd.Series:
        return series.astype("string").str.lower()

    have_gender = "gender" in df
    have_pol = "political_orientation" in df
    have_race = "race" in df
    have_relig = "religious_belief" in df
    have_sex = "sexual_orientation" in df

    gender = to_low(df["gender"]) if have_gender else pd.Series(index=df.index, dtype="string")
    pol = (
        to_low(df["political_orientation"])
        if have_pol
        else pd.Series(index=df.index, dtype="string")
    )
    race = to_low(df["race"]) if have_race else pd.Series(index=df.index, dtype="string")
    relig = (
        to_low(df["religious_belief"]) if have_relig else pd.Series(index=df.index, dtype="string")
    )
    sex = (
        to_low(df["sexual_orientation"]) if have_sex else pd.Series(index=df.index, dtype="string")
    )

    out = pd.DataFrame(index=df.index)

    # Gender
    if have_gender:
        out["gender_female"] = gender.isin(["female", "woman", "f"])
        out["gender_male"] = gender.isin(["male", "man", "m"])
        out["gender_non_binary"] = gender.isin(["non-binary", "nonbinary", "nb"])
        out["gender_other"] = ~(
            out["gender_female"] | out["gender_male"] | out["gender_non_binary"]
        )
    else:
        out["gender_female"] = False
        out["gender_male"] = False
        out["gender_non_binary"] = False
        out["gender_other"] = False

    # Political orientation
    if have_pol:
        out["pol_conservative"] = pol.isin(["conservative", "right", "right-wing"])
        out["pol_progressive"] = pol.isin(["progressive", "left", "left-wing", "liberal"])
        out["pol_centre"] = pol.isin(["centre", "center", "centrist", "moderate"])
        out["pol_other"] = ~(out["pol_conservative"] | out["pol_progressive"] | out["pol_centre"])
    else:
        out["pol_conservative"] = False
        out["pol_progressive"] = False
        out["pol_centre"] = False
        out["pol_other"] = False

    # Race
    if have_race:
        out["race_asia"] = race.isin(["asian", "asia"])
        out["race_black"] = race.str.contains("black", na=False)
        out["race_white"] = race.isin(["white"])
        out["race_latin"] = race.str.contains("latino|latinx|hispanic", na=False)
        out["race_other"] = ~(
            out["race_asia"] | out["race_black"] | out["race_white"] | out["race_latin"]
        )
    else:
        out["race_asia"] = False
        out["race_black"] = False
        out["race_white"] = False
        out["race_latin"] = False
        out["race_other"] = False

    # Religious belief (map others by exclusion). Note: prepare_population may
    # map Buddhist/Hinduist to Others already.
    if have_relig:
        out["relig_christian"] = relig.str.contains("christian", na=False)
        out["relig_agnostic"] = relig.str.contains("agnostic", na=False)
        out["relig_atheist"] = relig.str.contains("atheist", na=False)
        out["relig_other"] = ~(
            out["relig_christian"] | out["relig_agnostic"] | out["relig_atheist"]
        )
    else:
        out["relig_christian"] = False
        out["relig_agnostic"] = False
        out["relig_atheist"] = False
        out["relig_other"] = False

    # Sexual orientation
    if have_sex:
        out["sex_hetero"] = sex.isin(["heterosexual", "hetero"])
        out["sex_lgbtq"] = sex.str.contains(
            "gay|lesbian|bi|bisexual|pan|pansexual|queer|lgbt|lgbtq", na=False
        )
        out["sex_unspecified"] = sex.isin(["unspecified", "unknown", "none", "na"]) | sex.isna()
    else:
        out["sex_hetero"] = False
        out["sex_lgbtq"] = False
        out["sex_unspecified"] = False

    return out


def _format_mean_std(mean: float, std: float, marker: str = "") -> str:
    base = f"{mean:.2f} ± {std:.2f}"
    return base + (marker if marker else "")


def _proportion_ztest(a: np.ndarray, b: np.ndarray) -> float:
    """Perform z-test for proportions on binary data.

    Binary demographic analysis: each demographic value is treated as an
    independent binary variable (0/1). This tests whether proportions differ
    significantly between two groups.

    Args:
        a: Binary array (0/1) for group 1
        b: Binary array (0/1) for group 2

    Returns:
        p-value from two-sided z-test for proportions
    """
    if a.size == 0 or b.size == 0:
        return np.nan

    count_a = int(a.sum())
    count_b = int(b.sum())
    n_a = len(a)
    n_b = len(b)

    # Handle edge cases: no variance
    # Use np.unique to get unique values in numpy arrays
    unique_a = len(np.unique(a))
    unique_b = len(np.unique(b))

    if unique_a <= 1 and unique_b <= 1:
        prop_a = a.mean()
        prop_b = b.mean()
        if prop_a == prop_b:
            # Same proportion, no variance -> not significant
            return 1.0
        else:
            # Different proportions, no variance -> perfect separation
            return 0.0

    # Use statsmodels proportions_ztest if available
    if proportions_ztest is not None:
        try:
            counts = [count_a, count_b]
            nobs = [n_a, n_b]
            _, p_value = proportions_ztest(counts, nobs)
            return float(p_value)
        except Exception:
            pass

    # Fallback to t-test if proportions_ztest unavailable
    if ttest_ind is not None:
        try:
            res = ttest_ind(a, b, equal_var=False)
            p = getattr(res, "pvalue", np.nan)
            return float(p)
        except (TypeError, ValueError):
            pass

    return np.nan


def _mark(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "‡"
    if p < 0.01:
        return "†"
    if p < 0.05:
        return "*"
    return ""


def _to_float(val: object) -> float:
    """Convert pandas/NumPy scalar to float, returning NaN on failure.

    This avoids static typing issues with pandas Scalar types.
    """
    try:
        return float(np.asarray(val))
    except (TypeError, ValueError):
        return math.nan


def compute_table_demographics(
    df: pd.DataFrame,
    *,
    models: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute demographics tables (Tables 1–3) from a personas DataFrame.

    Binary demographic analysis: treats each demographic value as an
    independent binary variable (0/1) and uses z-test for proportions
    to compare against the Base condition.

    Expects columns at least: model, population, repetitions, gender,
    political_orientation, sexual_orientation.

    Returns:
    (tidy, pivot) where tidy has columns:
    [model, condition, metric, mean, std, p_value, marker, formatted]
    and pivot is shaped with rows (model, condition) and columns metric
    titles with formatted strings. Statistical significance is marked
    using z-test for proportions (not t-test).
    """
    work = df.copy()
    work["model_label"] = work["model"].map(_label_model)
    work["condition"] = work["population"].map(_label_condition)
    default_models = ["GPT-4o", "GPT-3.5"] if models is None else list(models)
    work = work[
        work["model_label"].isin(default_models) & work["condition"].isin(["Base", "MaxN", "MaxP"])
    ].copy()

    # Early exit if no rows for selected models/conditions
    if work.empty:
        tidy_cols = [
            "model",
            "condition",
            "metric",
            "mean",
            "std",
            "p_value",
            "marker",
            "formatted",
        ]
        tidy = pd.DataFrame(columns=tidy_cols)
        pivot_cols = ["model", "condition"] + [m.title for m in METRICS]
        pivot = pd.DataFrame(columns=pivot_cols)
        return tidy, pivot

    flags = _category_flags(work)
    work = pd.concat([work, flags], axis=1)

    # Compute replicate-level proportions per metric
    idx = (
        ["model_label", "condition", "repetitions"]
        if "repetitions" in work
        else ["model_label", "condition"]
    )
    metric_cols = [m.name for m in METRICS if m.name in work.columns]
    if not metric_cols:
        tidy_cols = [
            "model",
            "condition",
            "metric",
            "mean",
            "std",
            "p_value",
            "marker",
            "formatted",
        ]
        tidy = pd.DataFrame(columns=tidy_cols)
        pivot_cols = ["model", "condition"] + [m.title for m in METRICS]
        pivot = pd.DataFrame(columns=pivot_cols)
        return tidy, pivot
    per_rep = work.groupby(idx, as_index=False)[metric_cols].mean()

    if per_rep.empty:
        tidy_cols = [
            "model",
            "condition",
            "metric",
            "mean",
            "std",
            "p_value",
            "marker",
            "formatted",
        ]
        tidy = pd.DataFrame(columns=tidy_cols)
        pivot_cols = ["model", "condition"] + [m.title for m in METRICS]
        pivot = pd.DataFrame(columns=pivot_cols)
        return tidy, pivot

    # Aggregate across reps for display (mean ± std)
    agg_idx = ["model_label", "condition"]
    means = per_rep.groupby(agg_idx).mean(numeric_only=True)
    stds = per_rep.groupby(agg_idx).std(numeric_only=True)

    # Z-tests for proportions vs Base per model+metric
    # Binary demographic analysis: combine all raw data (not per-replicate)
    # This matches the original notebook approach
    pvals: Dict[Tuple[str, str, str], float] = {}
    for model_label in means.index.get_level_values(0).unique():
        # Get ALL raw data for Base condition (all repetitions combined)
        base_raw = work[(work["model_label"] == model_label) & (work["condition"] == "Base")]
        for cond in ["MaxN", "MaxP"]:
            # Get ALL raw data for this condition (all repetitions combined)
            other_raw = work[(work["model_label"] == model_label) & (work["condition"] == cond)]
            for m in METRICS:
                # Use z-test for proportions on raw binary data (0/1)
                # Not on aggregated per-replicate proportions
                if m.name in base_raw.columns and m.name in other_raw.columns:
                    p = _proportion_ztest(
                        base_raw[m.name].to_numpy(),
                        other_raw[m.name].to_numpy(),
                    )
                else:
                    p = np.nan
                pvals[(model_label, cond, m.name)] = p

    # Build tidy — ensure all three conditions per present model, allowing
    # zero-fills when some (model, condition) combos are missing entirely.
    records: List[Dict[str, object]] = []
    present_models = work["model_label"].dropna().unique().tolist()
    target_conditions = ["Base", "MaxN", "MaxP"]
    pairs_df = pd.DataFrame(
        [{"model_label": ml, "condition": c} for ml in present_models for c in target_conditions]
    )

    for model_label, condition in pairs_df.itertuples(index=False):
        for m in METRICS:
            mu = 0.0
            sd = 0.0
            has_mean = (model_label, condition) in means.index and m.name in means.columns
            if has_mean:
                mu_val = means.loc[(model_label, condition), m.name]
                mu = _to_float(mu_val)
                if not np.isfinite(mu):
                    mu = 0.0
            has_std = (model_label, condition) in stds.index and m.name in stds.columns
            if has_std:
                sd_val = stds.loc[(model_label, condition), m.name]
                sd = _to_float(sd_val)
                if not np.isfinite(sd):
                    sd = 0.0
            p = (
                pvals.get((model_label, condition, m.name), np.nan)
                if condition in ("MaxN", "MaxP")
                else np.nan
            )
            marker = _mark(p) if condition in ("MaxN", "MaxP") else ""
            # present as percentage points
            formatted = _format_mean_std(mu * 100, sd * 100, marker)
            records.append(
                {
                    "model": model_label,
                    "condition": condition,
                    "metric": m.title,
                    "mean": mu,
                    "std": sd,
                    "p_value": p,
                    "marker": marker,
                    "formatted": formatted,
                }
            )

    tidy = pd.DataFrame.from_records(
        records,
        columns=[
            "model",
            "condition",
            "metric",
            "mean",
            "std",
            "p_value",
            "marker",
            "formatted",
        ],
    )
    pivot = tidy.pivot_table(
        index=["model", "condition"],
        columns="metric",
        values="formatted",
        aggfunc="first",
    ).reset_index()
    # Ensure consistent column order
    cols = ["model", "condition"] + [m.title for m in METRICS]
    pivot = pivot.reindex(columns=cols)
    return tidy, pivot


def create_paper_table(
    df: pd.DataFrame,
    *,
    models: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Create multi-index demographic table matching the paper format.

    Binary demographic analysis: each demographic value is treated as an
    independent binary variable (0/1) and z-test for proportions is used
    for statistical significance testing against Base condition.

    Returns a DataFrame with:
    - Multi-index columns: (Model, Condition)
    - Multi-index rows: (Category, Subcategory)
    - Values: formatted mean ± std with significance markers
    - Significance markers: * (p<0.05), † (p<0.01), ‡ (p<0.001)
    """
    # Get the tidy data using existing function
    tidy, _ = compute_table_demographics(df, models=models)

    if tidy.empty:
        return pd.DataFrame()

    # Define demographic categories and their mapping
    category_mapping = {
        "Females": ("Gender", "Female"),
        "Male": ("Gender", "Male"),
        "Non-bin.": ("Gender", "Non-bin."),
        "Centre": ("Pol. Or.", "Centre"),
        "Con.": ("Pol. Or.", "Con."),
        "Prog.": ("Pol. Or.", "Prog."),
        "Asian": ("Race", "Asian"),
        "Black": ("Race", "Black"),
        "Latin": ("Race", "Latin"),
        "White": ("Race", "White"),
        "Christian": ("Relig. Beli.", "Christian"),
        "Agnostic": ("Relig. Beli.", "Agnostic"),
        "Atheist": ("Relig. Beli.", "Atheist"),
        "Hetero.": ("Sex. Or.", "Hetero."),
        "LGBTQ+": ("Sex. Or.", "LGBTQ+"),
        "Unspe.": ("Sex. Or.", "Unspe."),
    }

    # Handle "Others" category which appears in multiple groups
    def get_category_info(metric: str) -> Tuple[str, str]:
        if metric == "Others":
            # Find which category this "Others" belongs to by metric order
            if any(m.title == metric for m in METRICS[4:8]):  # Political
                return ("Pol. Or.", "Others")
            elif any(m.title == metric for m in METRICS[15:19]):  # Religious
                return ("Relig. Beli.", "Others")
        elif metric == "Other":
            # Find which category this "Other" belongs to by metric order
            if any(m.title == metric for m in METRICS[0:4]):  # Gender
                return ("Gender", "Other")
            elif any(m.title == metric for m in METRICS[8:13]):  # Race
                return ("Race", "Other")
        return category_mapping.get(metric, ("Unknown", metric))

    # Add category information to tidy data
    tidy["category"] = tidy["metric"].map(lambda x: get_category_info(x)[0])
    tidy["subcategory"] = tidy["metric"].map(lambda x: get_category_info(x)[1])

    # Create pivot with multi-index columns and rows
    pivot = tidy.pivot_table(
        index=["category", "subcategory"],
        columns=["model", "condition"],
        values="formatted",
        aggfunc="first",
    )

    # Ensure consistent column ordering (Base, MaxN, MaxP for each model)
    # Preserve the order of models from the input argument
    if not pivot.empty:
        # Use the input models list to preserve order, or fall back to
        # unique values
        if models is not None:
            models_ordered = list(models)
        else:
            models_ordered = pivot.columns.get_level_values(0).unique().tolist()

        conditions = ["Base", "MaxN", "MaxP"]

        # Create ordered column multi-index in the order of input models
        ordered_columns = []
        for model in models_ordered:
            for condition in conditions:
                if (model, condition) in pivot.columns:
                    ordered_columns.append((model, condition))

        pivot = pivot.reindex(columns=ordered_columns)

    # Ensure consistent row ordering
    category_order = ["Gender", "Pol. Or.", "Race", "Relig. Beli.", "Sex. Or."]
    if not pivot.empty:
        # Sort by category order, then by subcategory alphabetically
        sorted_index = sorted(
            pivot.index,
            key=lambda x: (category_order.index(x[0]) if x[0] in category_order else 999, x[1]),
        )
        pivot = pivot.reindex(index=sorted_index)

    return pivot
