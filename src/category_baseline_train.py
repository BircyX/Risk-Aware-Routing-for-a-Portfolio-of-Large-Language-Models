from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_PATH = RESULTS_DIR / "utility_outputs.csv"

QUERY_COL = "query_id"
SPLIT_COL = "split"
MODEL_COL = "model"
CORRECT_COL = "correct"
CATEGORY_COL = "category"
LATENCY_COL = "latency"

# Imputed but otherwise unclipped realised monetary cost.
REAL_COST_COL = "cost_clean"

MODELS = ["gpt", "claude", "gemini", "qwen"]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]


def utility_column(alpha: float) -> str:
    return f"utility_alpha_{alpha}"


def compute_oracle(split_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """
    Compute realised oracle utility for each query.
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
    Evaluate a routing policy against the realised
    per-query oracle.
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

    counts = selected[MODEL_COL].value_counts()

    for model in MODELS:
        result[f"selected_{model}"] = int(counts.get(model, 0))

    return result


def learn_category_mapping(train_df: pd.DataFrame, alpha: float):
    """
    Learn category -> best model mapping using TRAIN only.

    For each category, select the model with the highest
    mean realised utility in the training split.
    """
    util_col = utility_column(alpha)

    category_model_means = train_df.groupby(
        [CATEGORY_COL, MODEL_COL], as_index=False
    )[util_col].mean()

    best_indices = category_model_means.groupby(CATEGORY_COL)[util_col].idxmax()

    best_rows = (
        category_model_means.loc[best_indices]
        .sort_values(CATEGORY_COL)
        .reset_index(drop=True)
    )

    mapping = dict(zip(best_rows[CATEGORY_COL], best_rows[MODEL_COL]))

    global_model_means = (
        train_df.groupby(MODEL_COL)[util_col].mean().sort_values(ascending=False)
    )

    fallback_model = global_model_means.index[0]

    return mapping, fallback_model, best_rows


def apply_category_mapping(
    split_df: pd.DataFrame,
    mapping: dict,
    fallback_model: str,
) -> pd.DataFrame:
    """
    Apply the frozen category mapping to a new split.
    """
    query_categories = split_df[[QUERY_COL, CATEGORY_COL]].drop_duplicates().copy()

    expected_queries = split_df[QUERY_COL].nunique()

    if len(query_categories) != expected_queries:
        raise ValueError(
            "At least one query has inconsistent category labels across model rows."
        )

    query_categories["selected_model"] = (
        query_categories[CATEGORY_COL].map(mapping).fillna(fallback_model)
    )

    selected = split_df.merge(
        query_categories[[QUERY_COL, "selected_model"]],
        on=QUERY_COL,
        how="left",
    )

    selected = selected[selected[MODEL_COL] == selected["selected_model"]].copy()

    selected.drop(columns=["selected_model"], inplace=True)

    if len(selected) != expected_queries:
        raise ValueError(
            f"Selected {len(selected)} rows, expected {expected_queries}."
        )

    return selected


def run_category_router(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Learn the category mapping on training data,
    freeze it, and evaluate it on validation and test.
    """
    validation_results = []
    test_results = []
    mapping_records = []

    for alpha in ALPHAS:
        mapping, fallback_model, best_rows = learn_category_mapping(
            train_df=train_df, alpha=alpha
        )

        util_col = utility_column(alpha)

        for _, row in best_rows.iterrows():
            mapping_records.append(
                {
                    "alpha": alpha,
                    "category": row[CATEGORY_COL],
                    "selected_model": row[MODEL_COL],
                    "training_mean_utility": row[util_col],
                    "fallback_model": fallback_model,
                }
            )

        selected_val = apply_category_mapping(
            split_df=val_df,
            mapping=mapping,
            fallback_model=fallback_model,
        )

        validation_results.append(
            evaluate_policy(
                full_split_df=val_df,
                selected_rows=selected_val,
                alpha=alpha,
                policy_name="Category Router (train-selected)",
            )
        )

        selected_test = apply_category_mapping(
            split_df=test_df,
            mapping=mapping,
            fallback_model=fallback_model,
        )

        test_results.append(
            evaluate_policy(
                full_split_df=test_df,
                selected_rows=selected_test,
                alpha=alpha,
                policy_name="Category Router (train-selected)",
            )
        )

    return (
        pd.DataFrame(validation_results),
        pd.DataFrame(test_results),
        pd.DataFrame(mapping_records),
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    df[SPLIT_COL] = df[SPLIT_COL].astype(str).str.strip().str.lower()
    df[MODEL_COL] = df[MODEL_COL].astype(str).str.strip().str.lower()
    df[CATEGORY_COL] = df[CATEGORY_COL].astype(str).str.strip()

    train_df = df[df[SPLIT_COL] == "train"].copy()
    val_df = df[df[SPLIT_COL].isin(["validation", "val"])].copy()
    test_df = df[df[SPLIT_COL] == "test"].copy()

    if train_df.empty:
        raise ValueError("Training split is empty.")

    if val_df.empty:
        raise ValueError("Validation split is empty.")

    if test_df.empty:
        raise ValueError("Test split is empty.")

    validation_results, test_results, mapping_results = run_category_router(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )

    validation_results.to_csv(
        RESULTS_DIR / "category_router_train_selected_validation.csv",
        index=False,
    )

    test_results.to_csv(
        RESULTS_DIR / "category_router_train_selected_test.csv",
        index=False,
    )

    mapping_results.to_csv(
        RESULTS_DIR / "category_router_train_selected_mapping.csv",
        index=False,
    )

    print(f"Saved train-selected category baseline results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()