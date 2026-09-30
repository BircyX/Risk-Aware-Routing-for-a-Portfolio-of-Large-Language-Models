from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

FIXED_FILE = RESULTS_DIR / "additional_fixed_model_baselines.csv"
BEST_FIXED_FILE = RESULTS_DIR / "additional_best_fixed_utility_test.csv"
CATEGORY_FILE = RESULTS_DIR / "category_router_train_selected_test.csv"
FULL_ROUTER_FILE = RESULTS_DIR / "final_test_results.csv"
NO_CATEGORY_FILE = RESULTS_DIR / "ablation_no_category" / "test_results.csv"

COMPARISON_OUT = RESULTS_DIR / "final_policy_comparison.csv"
PARETO_OUT = RESULTS_DIR / "final_accuracy_cost_pareto.csv"
PARETO_UNIQUE_OUT = RESULTS_DIR / "final_accuracy_cost_pareto_unique.csv"

ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]

ACCURACY_TOLERANCE = 1e-6
COST_TOLERANCE = 1e-10


def first_existing_column(df, candidates, file_name):
    for column in candidates:
        if column in df.columns:
            return column

    raise ValueError(
        f"Could not find any of {candidates} "
        f"in {file_name}. "
        f"Available columns: {list(df.columns)}"
    )


def optional_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def standardise_metrics(df, file_name):
    """
    Map result files with slightly different column names
    into one common schema.
    """
    alpha_col = first_existing_column(df, ["alpha"], file_name)

    accuracy_col = first_existing_column(df, ["accuracy", "acc"], file_name)

    cost_col = first_existing_column(
        df, ["average_cost", "avg_cost", "cost"], file_name
    )

    utility_col = first_existing_column(
        df,
        ["mean_realized_utility", "mean_utility", "utility"],
        file_name,
    )

    regret_col = first_existing_column(df, ["mean_regret", "regret"], file_name)

    oracle_col = optional_column(df, ["mean_oracle_utility", "oracle_utility"])

    latency_col = optional_column(df, ["average_latency", "avg_latency", "latency"])

    n_col = optional_column(df, ["num_queries", "n_queries"])

    lambda_col = optional_column(df, ["lambda", "lambda_value"])

    result = pd.DataFrame(
        {
            "alpha": df[alpha_col].astype(float),
            "accuracy": df[accuracy_col].astype(float),
            "average_cost": df[cost_col].astype(float),
            "mean_realized_utility": df[utility_col].astype(float),
            "mean_regret": df[regret_col].astype(float),
        }
    )

    if oracle_col is not None:
        result["mean_oracle_utility"] = df[oracle_col].astype(float)
    else:
        result["mean_oracle_utility"] = np.nan

    if latency_col is not None:
        result["average_latency"] = df[latency_col].astype(float)
    else:
        result["average_latency"] = np.nan

    if n_col is not None:
        result["num_queries"] = df[n_col].astype(int)
    else:
        result["num_queries"] = 486

    if lambda_col is not None:
        result["lambda"] = pd.to_numeric(df[lambda_col], errors="coerce")
    else:
        result["lambda"] = np.nan

    return result


def load_fixed_models():
    df = pd.read_csv(FIXED_FILE)

    model_col = first_existing_column(
        df,
        ["model", "selected_model", "router", "policy", "method"],
        FIXED_FILE.name,
    )

    standard = standardise_metrics(df, FIXED_FILE.name)

    model_values = df[model_col].astype(str).str.strip().str.lower()

    rows = []

    for i in range(len(df)):
        model = model_values.iloc[i]

        for candidate in ["gpt", "claude", "gemini", "qwen"]:
            if candidate in model:
                model = candidate
                break

        row = standard.iloc[i].to_dict()

        row["policy"] = f"Fixed {model.upper()}"
        row["policy_family"] = "fixed_model"
        row["selected_model"] = model

        rows.append(row)

    return pd.DataFrame(rows)


def load_best_fixed():
    df = pd.read_csv(BEST_FIXED_FILE)

    standard = standardise_metrics(df, BEST_FIXED_FILE.name)

    model_col = optional_column(df, ["selected_model", "best_model", "model"])

    standard["policy"] = "Best Fixed Utility"
    standard["policy_family"] = "best_fixed"

    if model_col is not None:
        standard["selected_model"] = df[model_col].astype(str).str.strip().str.lower()
    else:
        standard["selected_model"] = ""

    return standard


def load_category_router():
    df = pd.read_csv(CATEGORY_FILE)

    standard = standardise_metrics(df, CATEGORY_FILE.name)

    standard["policy"] = "Category-only"
    standard["policy_family"] = "category_router"
    standard["selected_model"] = ""

    return standard


def load_full_router():
    df = pd.read_csv(FULL_ROUTER_FILE)

    router_col = first_existing_column(
        df, ["router", "policy", "method"], FULL_ROUTER_FILE.name
    )

    standard = standardise_metrics(df, FULL_ROUTER_FILE.name)

    rows = []

    for i in range(len(df)):
        router = str(df.iloc[i][router_col]).strip().lower()

        if "utility" in router and "risk" not in router:
            policy = "Full XGBoost Utility-only"
        elif "risk" in router:
            policy = "Full XGBoost Risk-aware"
        else:
            policy = f"Full XGBoost {router}"

        row = standard.iloc[i].to_dict()

        row["policy"] = policy
        row["policy_family"] = "full_xgboost"
        row["selected_model"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def load_no_category():
    df = pd.read_csv(NO_CATEGORY_FILE)

    router_col = first_existing_column(
        df, ["router", "policy", "method"], NO_CATEGORY_FILE.name
    )

    standard = standardise_metrics(df, NO_CATEGORY_FILE.name)

    rows = []

    for i in range(len(df)):
        router = str(df.iloc[i][router_col]).strip().lower()

        if "utility" in router and "risk" not in router:
            policy = "No-category XGBoost Utility-only"
        elif "risk" in router:
            policy = "No-category XGBoost Risk-aware"
        else:
            policy = f"No-category XGBoost {router}"

        row = standard.iloc[i].to_dict()

        row["policy"] = policy
        row["policy_family"] = "no_category_xgboost"
        row["selected_model"] = ""

        rows.append(row)

    return pd.DataFrame(rows)


def add_pareto_status(df):
    """
    Mark accuracy-cost Pareto-efficient rows.

    A point is dominated when another point has at least
    the same accuracy and no greater cost, with a meaningful
    improvement in at least one dimension.
    """
    result = df.copy().reset_index(drop=True)

    flags = []

    for i, row_i in result.iterrows():
        dominated = False

        for j, row_j in result.iterrows():
            if i == j:
                continue

            accuracy_no_worse = (
                row_j["accuracy"] >= row_i["accuracy"] - ACCURACY_TOLERANCE
            )

            cost_no_worse = (
                row_j["average_cost"] <= row_i["average_cost"] + COST_TOLERANCE
            )

            accuracy_strictly_better = (
                row_j["accuracy"] > row_i["accuracy"] + ACCURACY_TOLERANCE
            )

            cost_strictly_better = (
                row_j["average_cost"] < row_i["average_cost"] - COST_TOLERANCE
            )

            if (
                accuracy_no_worse
                and cost_no_worse
                and (accuracy_strictly_better or cost_strictly_better)
            ):
                dominated = True
                break

        flags.append(not dominated)

    result["accuracy_cost_pareto"] = flags

    return result


def build_unique_pareto_points(pareto_df):
    """
    Collapse repeated Pareto rows that represent the same
    accuracy-cost operating point.

    Utility and regret remain alpha-dependent, so their
    ranges are retained.
    """
    if pareto_df.empty:
        return pareto_df.copy()

    df = pareto_df.copy()

    df["_accuracy_key"] = df["accuracy"].round(6)
    df["_cost_key"] = df["average_cost"].round(9)

    rows = []

    grouped = df.groupby(
        ["policy", "_accuracy_key", "_cost_key"],
        sort=False,
        dropna=False,
    )

    for (policy, _, _), group in grouped:
        alpha_values = sorted(group["alpha"].astype(float).tolist())

        alpha_label = ",".join(str(alpha) for alpha in alpha_values)

        rows.append(
            {
                "policy": policy,
                "alpha_values": alpha_label,
                "accuracy": float(group["accuracy"].iloc[0]),
                "average_cost": float(group["average_cost"].iloc[0]),
                "mean_utility_min": float(group["mean_realized_utility"].min()),
                "mean_utility_max": float(group["mean_realized_utility"].max()),
                "mean_regret_min": float(group["mean_regret"].min()),
                "mean_regret_max": float(group["mean_regret"].max()),
                "num_alpha_settings": int(len(alpha_values)),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by=["average_cost", "accuracy"], ascending=[True, True])
        .reset_index(drop=True)
    )


def main():
    fixed_df = load_fixed_models()
    best_fixed_df = load_best_fixed()
    category_df = load_category_router()
    full_df = load_full_router()
    no_category_df = load_no_category()

    comparison = pd.concat(
        [fixed_df, best_fixed_df, category_df, full_df, no_category_df],
        ignore_index=True,
        sort=False,
    )

    core_columns = [
        "accuracy",
        "average_cost",
        "mean_realized_utility",
        "mean_regret",
    ]

    for column in core_columns:
        if comparison[column].isna().any():
            raise ValueError(f"Missing values detected in core metric: {column}")

    comparison = add_pareto_status(comparison)

    family_order = {
        "fixed_model": 0,
        "best_fixed": 1,
        "category_router": 2,
        "full_xgboost": 3,
        "no_category_xgboost": 4,
    }

    comparison["_family_order"] = comparison["policy_family"].map(family_order)

    comparison = (
        comparison.sort_values(by=["alpha", "_family_order", "policy"])
        .drop(columns=["_family_order"])
        .reset_index(drop=True)
    )

    pareto = (
        comparison[comparison["accuracy_cost_pareto"]]
        .copy()
        .sort_values(
            by=["average_cost", "accuracy", "policy", "alpha"],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )

    unique_pareto = build_unique_pareto_points(pareto)

    comparison.to_csv(COMPARISON_OUT, index=False)
    pareto.to_csv(PARETO_OUT, index=False)
    unique_pareto.to_csv(PARETO_UNIQUE_OUT, index=False)

    print(f"Saved final comparison and Pareto results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()