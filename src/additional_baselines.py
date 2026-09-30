from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PATH = RESULTS_DIR / "utility_outputs.csv"

QUERY_COL = "query_id"
SPLIT_COL = "split"
MODEL_COL = "model"
CORRECT_COL = "correct"
LATENCY_COL = "latency"

# Imputed but otherwise unclipped realised monetary cost.
REAL_COST_COL = "cost_clean"

MODELS = ["gpt", "claude", "gemini", "qwen"]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]


def utility_column(alpha: float) -> str:
    return f"utility_alpha_{alpha}"


def compute_oracle(split_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Compute the realised oracle utility for each query.
    """
    util_col = utility_column(alpha)

    return (
        split_df.groupby(QUERY_COL, as_index=False)[util_col]
        .max()
        .rename(columns={util_col: "oracle_utility"})
    )


def evaluate_policy(
    full_split_df: pd.DataFrame,
    selected_rows: pd.DataFrame,
    alpha: float,
    policy_name: str,
) -> dict:
    """
    Evaluate a routing policy using accuracy, realised cost,
    mean utility and regret relative to the per-query oracle.
    """
    util_col = utility_column(alpha)

    selected = selected_rows.merge(
        compute_oracle(full_split_df, alpha),
        on=QUERY_COL,
        how="left",
        validate="one_to_one",
    )

    selected["regret"] = selected["oracle_utility"] - selected[util_col]

    result = {
        "alpha": alpha,
        "policy": policy_name,
        "n_queries": len(selected),
        "accuracy": selected[CORRECT_COL].mean(),
        "avg_cost": selected[REAL_COST_COL].mean(),
        "avg_latency": selected[LATENCY_COL].mean(),
        "mean_utility": selected[util_col].mean(),
        "oracle_utility": selected["oracle_utility"].mean(),
        "mean_regret": selected["regret"].mean(),
    }

    selection_counts = selected[MODEL_COL].value_counts()

    for model in MODELS:
        result[f"selected_{model}"] = int(selection_counts.get(model, 0))

    return result


def fixed_model_rows(split_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Select the same model for every query.
    """
    return split_df[split_df[MODEL_COL] == model].copy()


# Fixed-model baselines
def run_fixed_model_baselines(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate each of the four candidate models as a fixed policy
    on the test split.
    """
    results = []

    for alpha in ALPHAS:
        for model in MODELS:
            selected = fixed_model_rows(test_df, model)

            result = evaluate_policy(
                full_split_df=test_df,
                selected_rows=selected,
                alpha=alpha,
                policy_name=f"Fixed {model.capitalize()}",
            )

            results.append(result)

    return pd.DataFrame(results)


# Best fixed-utility baseline
def run_best_fixed_utility(val_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    For each alpha, select the fixed model with the highest
    mean utility on validation, freeze that choice, and then
    evaluate it on test.
    """
    test_results = []
    selection_records = []

    for alpha in ALPHAS:
        util_col = utility_column(alpha)

        validation_means = (
            val_df.groupby(MODEL_COL)[util_col].mean().sort_values(ascending=False)
        )

        best_model = validation_means.index[0]
        best_utility = validation_means.iloc[0]

        selection_record = {
            "alpha": alpha,
            "selected_model": best_model,
            "validation_mean_utility": best_utility,
        }

        for model in MODELS:
            selection_record[f"validation_utility_{model}"] = validation_means.loc[
                model
            ]

        selection_records.append(selection_record)

        selected_test_rows = fixed_model_rows(test_df, best_model)

        result = evaluate_policy(
            full_split_df=test_df,
            selected_rows=selected_test_rows,
            alpha=alpha,
            policy_name="Best Fixed Utility",
        )

        result["validation_selected_model"] = best_model
        result["validation_mean_utility"] = best_utility

        test_results.append(result)

    return pd.DataFrame(test_results), pd.DataFrame(selection_records)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Normalise fields used for split and model matching.
    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()
    df[MODEL_COL] = df[MODEL_COL].astype(str).str.strip().str.lower()

    val_df = df[df[SPLIT_COL].isin(["validation", "val"])].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()

    fixed_results = run_fixed_model_baselines(test_df)

    fixed_results.to_csv(
        RESULTS_DIR / "additional_fixed_model_baselines.csv",
        index=False,
    )

    best_fixed_results, best_fixed_selection = run_best_fixed_utility(val_df, test_df)

    best_fixed_results.to_csv(
        RESULTS_DIR / "additional_best_fixed_utility_test.csv",
        index=False,
    )

    best_fixed_selection.to_csv(
        RESULTS_DIR / "additional_best_fixed_utility_selection.csv",
        index=False,
    )

    # Convenience summary containing both fixed-baseline families.
    pd.concat(
        [fixed_results, best_fixed_results],
        ignore_index=True,
        sort=False,
    ).to_csv(
        RESULTS_DIR / "additional_baselines_summary.csv",
        index=False,
    )

    print(f"Saved baseline results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()