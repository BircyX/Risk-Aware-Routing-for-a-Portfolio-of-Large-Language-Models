from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

PREDICTIONS_FILE = RESULTS_DIR / "final_test_predictions_aligned.csv"

UTILITY_FILE = RESULTS_DIR / "utility_outputs.csv"

OUTPUT_DIR = RESULTS_DIR / "prediction_error_analysis"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt", "claude", "gemini", "qwen"]

ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]


def error_summary(group: pd.DataFrame) -> pd.Series:
    error = group["prediction_error"]
    abs_error = group["abs_prediction_error"]

    return pd.Series(
        {
            "n": len(group),
            "mae": abs_error.mean(),
            "median_abs_error": abs_error.median(),
            "p90_abs_error": abs_error.quantile(0.90),
            "rmse": np.sqrt(np.mean(error**2)),
            "mean_signed_error": error.mean(),
            "mean_risk": group["predictive_risk"].mean(),
        }
    )


def load_data():
    pred = pd.read_csv(PREDICTIONS_FILE)
    util = pd.read_csv(UTILITY_FILE)

    if "split" in util.columns:
        util["split"] = util["split"].astype(str).str.lower()
        util_test = util[util["split"] == "test"].copy()
    else:
        test_ids = set(pred["query_id"].unique())
        util_test = util[util["query_id"].isin(test_ids)].copy()

    return pred, util_test


def build_prediction_long(pred):
    rows = []

    for _, row in pred.iterrows():
        alpha = float(row["alpha"])

        for model in MODELS:
            mean_col = f"mean_utility_{model}"
            risk_col = f"risk_{model}"

            if mean_col not in pred.columns or risk_col not in pred.columns:
                raise ValueError(f"Missing prediction columns for model {model}.")

            rows.append(
                {
                    "query_id": row["query_id"],
                    "dataset": row.get("dataset", np.nan),
                    "category": row.get("category", np.nan),
                    "alpha": alpha,
                    "model": model,
                    "predicted_utility": row[mean_col],
                    "predictive_risk": row[risk_col],
                }
            )

    return pd.DataFrame(rows)


def build_utility_long(util_test):
    rows = []

    for alpha in ALPHAS:
        utility_col = f"utility_alpha_{alpha}"

        if utility_col not in util_test.columns:
            raise ValueError(f"Missing utility column: {utility_col}")

        tmp = util_test[
            ["query_id", "dataset", "category", "model", utility_col, "correct", "cost"]
        ].copy()

        tmp["alpha"] = alpha
        tmp = tmp.rename(columns={utility_col: "realised_utility"})

        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)


def build_error_table(pred_long, utility_long):
    df = pred_long.merge(
        utility_long[
            ["query_id", "model", "alpha", "realised_utility", "correct", "cost"]
        ],
        on=["query_id", "model", "alpha"],
        how="left",
        validate="one_to_one",
    )

    if df["realised_utility"].isna().any():
        raise ValueError(
            "Some predicted rows could not be matched to realised utility."
        )

    df["prediction_error"] = df["predicted_utility"] - df["realised_utility"]
    df["abs_prediction_error"] = df["prediction_error"].abs()
    df["squared_prediction_error"] = df["prediction_error"] ** 2

    return df


def build_ranking_analysis(df):
    ranking_rows = []

    for (alpha, query_id), group in df.groupby(["alpha", "query_id"]):
        if len(group) != len(MODELS):
            continue

        group = group.sort_values("predicted_utility", ascending=False).reset_index(
            drop=True
        )

        pred_top = group.iloc[0]
        pred_second = group.iloc[1]

        predicted_model = pred_top["model"]
        predicted_margin = (
            pred_top["predicted_utility"] - pred_second["predicted_utility"]
        )

        realised_sorted = group.sort_values(
            "realised_utility", ascending=False
        ).reset_index(drop=True)

        oracle_row = realised_sorted.iloc[0]
        oracle_model = oracle_row["model"]
        oracle_utility = oracle_row["realised_utility"]

        selected_realised_utility = pred_top["realised_utility"]
        regret = oracle_utility - selected_realised_utility

        inversion_count = 0
        valid_pair_count = 0
        reversed_pairs = []

        for model_a, model_b in combinations(MODELS, 2):
            a = group[group["model"] == model_a].iloc[0]
            b = group[group["model"] == model_b].iloc[0]

            pred_diff = a["predicted_utility"] - b["predicted_utility"]
            real_diff = a["realised_utility"] - b["realised_utility"]

            # Exact ties do not define a strict ranking.
            if pred_diff == 0 or real_diff == 0:
                continue

            valid_pair_count += 1

            if np.sign(pred_diff) != np.sign(real_diff):
                inversion_count += 1
                reversed_pairs.append(f"{model_a}>{model_b}")

        inversion_rate = (
            inversion_count / valid_pair_count if valid_pair_count > 0 else np.nan
        )

        ranking_rows.append(
            {
                "query_id": query_id,
                "alpha": alpha,
                "dataset": pred_top["dataset"],
                "category": pred_top["category"],
                "predicted_top_model": predicted_model,
                "realised_top_model": oracle_model,
                "top1_match": int(predicted_model == oracle_model),
                "predicted_top_utility": pred_top["predicted_utility"],
                "predicted_second_utility": pred_second["predicted_utility"],
                "predicted_margin": predicted_margin,
                "selected_prediction_error": pred_top["prediction_error"],
                "selected_abs_error": pred_top["abs_prediction_error"],
                "selected_risk": pred_top["predictive_risk"],
                "mean_abs_error_all_models": group["abs_prediction_error"].mean(),
                "max_abs_error_all_models": group["abs_prediction_error"].max(),
                "realised_regret": regret,
                "material_failure": int(regret > 0.05),
                "pairwise_inversions": inversion_count,
                "valid_pairwise_comparisons": valid_pair_count,
                "pairwise_inversion_rate": inversion_rate,
                "reversed_pairs": ";".join(reversed_pairs),
            }
        )

    return pd.DataFrame(ranking_rows)


def build_pairwise_reversals(df):
    rows = []

    for alpha, alpha_df in df.groupby("alpha"):
        for model_a, model_b in combinations(MODELS, 2):
            a = alpha_df[alpha_df["model"] == model_a][
                ["query_id", "predicted_utility", "realised_utility"]
            ].rename(
                columns={"predicted_utility": "pred_a", "realised_utility": "real_a"}
            )

            b = alpha_df[alpha_df["model"] == model_b][
                ["query_id", "predicted_utility", "realised_utility"]
            ].rename(
                columns={"predicted_utility": "pred_b", "realised_utility": "real_b"}
            )

            merged = a.merge(b, on="query_id", how="inner")

            merged["pred_diff"] = merged["pred_a"] - merged["pred_b"]
            merged["real_diff"] = merged["real_a"] - merged["real_b"]

            valid_df = merged[
                (merged["pred_diff"] != 0) & (merged["real_diff"] != 0)
            ].copy()

            if valid_df.empty:
                reversal_count = 0
                reversal_rate = np.nan
            else:
                reversal_mask = np.sign(valid_df["pred_diff"]) != np.sign(
                    valid_df["real_diff"]
                )
                reversal_count = int(reversal_mask.sum())
                reversal_rate = reversal_count / len(valid_df)

            rows.append(
                {
                    "alpha": alpha,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_queries": len(valid_df),
                    "n_reversals": reversal_count,
                    "reversal_rate": reversal_rate,
                    "mean_predicted_gap": valid_df["pred_diff"].abs().mean(),
                    "mean_realised_gap": valid_df["real_diff"].abs().mean(),
                }
            )

    return pd.DataFrame(rows)


def main():
    pred, util_test = load_data()

    pred_long = build_prediction_long(pred)
    utility_long = build_utility_long(util_test)

    df = build_error_table(pred_long, utility_long)

    df.to_csv(OUTPUT_DIR / "prediction_error_long.csv", index=False)

    error_by_model = (
        df.groupby(["alpha", "model"], dropna=False)
        .apply(error_summary, include_groups=False)
        .reset_index()
    )

    error_by_model.to_csv(
        OUTPUT_DIR / "prediction_error_by_model_alpha.csv", index=False
    )

    error_by_category = (
        df.groupby(["alpha", "category"], dropna=False)
        .apply(error_summary, include_groups=False)
        .reset_index()
    )

    error_by_category.to_csv(
        OUTPUT_DIR / "prediction_error_by_category_alpha.csv", index=False
    )

    error_by_model_category = (
        df.groupby(["alpha", "category", "model"], dropna=False)
        .apply(error_summary, include_groups=False)
        .reset_index()
    )

    error_by_model_category.to_csv(
        OUTPUT_DIR / "prediction_error_by_model_category_alpha.csv", index=False
    )

    ranking = build_ranking_analysis(df)

    ranking.to_csv(OUTPUT_DIR / "ranking_error_analysis.csv", index=False)

    summary_rows = []

    for alpha, group in ranking.groupby("alpha"):
        ranking_correct = group[group["top1_match"] == 1]
        ranking_wrong = group[group["top1_match"] == 0]

        summary_rows.append(
            {
                "alpha": alpha,
                "n_queries": len(group),
                "top1_match_rate": group["top1_match"].mean(),
                "top1_ranking_error_rate": 1 - group["top1_match"].mean(),
                "mean_pairwise_inversion_rate": group["pairwise_inversion_rate"].mean(),
                "mean_predicted_margin": group["predicted_margin"].mean(),
                "ranking_correct_mean_margin": ranking_correct[
                    "predicted_margin"
                ].mean(),
                "ranking_wrong_mean_margin": ranking_wrong["predicted_margin"].mean(),
                "ranking_correct_mean_abs_error": ranking_correct[
                    "selected_abs_error"
                ].mean(),
                "ranking_wrong_mean_abs_error": ranking_wrong[
                    "selected_abs_error"
                ].mean(),
                "ranking_correct_mean_risk": ranking_correct["selected_risk"].mean(),
                "ranking_wrong_mean_risk": ranking_wrong["selected_risk"].mean(),
                "ranking_correct_mean_regret": ranking_correct[
                    "realised_regret"
                ].mean(),
                "ranking_wrong_mean_regret": ranking_wrong["realised_regret"].mean(),
                "material_failure_rate": group["material_failure"].mean(),
            }
        )

    ranking_summary = pd.DataFrame(summary_rows)

    ranking_summary.to_csv(OUTPUT_DIR / "ranking_summary_by_alpha.csv", index=False)

    ranking_by_category = (
        ranking.groupby(["alpha", "category"], dropna=False)
        .agg(
            n_queries=("query_id", "size"),
            top1_match_rate=("top1_match", "mean"),
            mean_predicted_margin=("predicted_margin", "mean"),
            mean_selected_abs_error=("selected_abs_error", "mean"),
            mean_selected_risk=("selected_risk", "mean"),
            mean_regret=("realised_regret", "mean"),
            material_failure_rate=("material_failure", "mean"),
            mean_pairwise_inversion_rate=("pairwise_inversion_rate", "mean"),
        )
        .reset_index()
    )

    ranking_by_category.to_csv(
        OUTPUT_DIR / "ranking_error_by_category_alpha.csv", index=False
    )

    pairwise = build_pairwise_reversals(df)

    pairwise.to_csv(OUTPUT_DIR / "pairwise_ranking_reversals.csv", index=False)

    margin_frames = []

    for alpha, group in ranking.groupby("alpha"):
        group = group.copy()

        group["margin_quintile"] = pd.qcut(
            group["predicted_margin"],
            q=5,
            labels=["Q1 smallest", "Q2", "Q3", "Q4", "Q5 largest"],
            duplicates="drop",
        )

        margin_frames.append(group)

    margin_df = pd.concat(margin_frames, ignore_index=True)

    margin_summary = (
        margin_df.groupby(["alpha", "margin_quintile"], observed=True)
        .agg(
            n_queries=("query_id", "size"),
            mean_margin=("predicted_margin", "mean"),
            top1_match_rate=("top1_match", "mean"),
            mean_abs_error=("selected_abs_error", "mean"),
            mean_risk=("selected_risk", "mean"),
            mean_regret=("realised_regret", "mean"),
            material_failure_rate=("material_failure", "mean"),
            mean_pairwise_inversion_rate=("pairwise_inversion_rate", "mean"),
        )
        .reset_index()
    )

    margin_summary.to_csv(OUTPUT_DIR / "ranking_by_margin_quintile.csv", index=False)

    print(f"Saved prediction and ranking error analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()