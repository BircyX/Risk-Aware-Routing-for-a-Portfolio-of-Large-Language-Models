import csv
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

PREDICTION_FILE = RESULTS_DIR / "final_test_predictions_aligned.csv"
UTILITY_FILE = RESULTS_DIR / "utility_outputs.csv"

OUTPUT_FILE = RESULTS_DIR / "final_risk_mechanism_analysis.csv"
SUMMARY_FILE = RESULTS_DIR / "final_risk_mechanism_summary.csv"

ALPHA = 0.5

MODELS = ["gpt", "claude", "gemini", "qwen"]


def load_predictions():
    """
    Load test queries for which the utility-only and
    risk-aware routers make different decisions.
    """
    rows = []

    with open(PREDICTION_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            alpha = float(row["alpha"])

            if abs(alpha - ALPHA) > 1e-9:
                continue

            if int(row["decision_changed"]) != 1:
                continue

            rows.append(row)

    return rows


def load_utility_data():
    """
    Load realised test outcomes for all candidate models.
    """
    data = {}

    with open(UTILITY_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["split"] != "test":
                continue

            query_id = row["query_id"]
            model = row["model"]

            if query_id not in data:
                data[query_id] = {
                    "dataset": row.get("dataset", ""),
                    "category": row.get("category", ""),
                    "question": row.get("question") or "",
                    "models": {},
                }

            data[query_id]["models"][model] = {
                "correct": int(row["correct"]),
                "cost": float(row["cost_clean"]),
                "utility": float(row[f"utility_alpha_{ALPHA}"]),
            }

    return data


def classify_transition(utility_correct, risk_correct):
    if utility_correct == 0 and risk_correct == 1:
        return "wrong_to_correct"

    if utility_correct == 1 and risk_correct == 0:
        return "correct_to_wrong"

    if utility_correct == 1 and risk_correct == 1:
        return "both_correct"

    return "both_wrong"


def main():
    prediction_rows = load_predictions()
    utility_data = load_utility_data()

    output_rows = []

    for row in prediction_rows:
        query_id = row["query_id"]

        lambda_value = float(row["lambda"])

        utility_model = row["utility_only_model"]
        risk_model = row["risk_aware_model"]

        query = utility_data[query_id]
        actual_models = query["models"]

        # Predicted utility and predictive risk of both selections.
        utility_mean = float(row[f"mean_utility_{utility_model}"])
        risk_mean = float(row[f"mean_utility_{risk_model}"])

        utility_risk = float(row[f"risk_{utility_model}"])
        risk_risk = float(row[f"risk_{risk_model}"])

        utility_penalized_score = float(row[f"risk_adjusted_score_{utility_model}"])
        risk_penalized_score = float(row[f"risk_adjusted_score_{risk_model}"])

        utility_correct = actual_models[utility_model]["correct"]
        risk_correct = actual_models[risk_model]["correct"]

        utility_realized = actual_models[utility_model]["utility"]
        risk_realized = actual_models[risk_model]["utility"]

        correctness_transition = classify_transition(utility_correct, risk_correct)

        lower_risk_selected = int(risk_risk < utility_risk)
        lower_mean_selected = int(risk_mean < utility_mean)

        utility_penalty = lambda_value * utility_risk
        risk_penalty = lambda_value * risk_risk

        mean_ranking = sorted(
            MODELS,
            key=lambda model: float(row[f"mean_utility_{model}"]),
            reverse=True,
        )

        score_ranking = sorted(
            MODELS,
            key=lambda model: float(row[f"risk_adjusted_score_{model}"]),
            reverse=True,
        )

        output_row = {
            "query_id": query_id,
            "dataset": query["dataset"],
            "category": query["category"],
            "question": query["question"],
            "alpha": ALPHA,
            "lambda": lambda_value,
            "utility_only_model": utility_model,
            "risk_aware_model": risk_model,
            "correctness_transition": correctness_transition,
            "utility_only_predicted_mean": utility_mean,
            "utility_only_risk": utility_risk,
            "utility_only_penalty": utility_penalty,
            "utility_only_risk_adjusted_score": utility_penalized_score,
            "utility_only_correct": utility_correct,
            "utility_only_realized_utility": utility_realized,
            "risk_aware_predicted_mean": risk_mean,
            "risk_aware_risk": risk_risk,
            "risk_aware_penalty": risk_penalty,
            "risk_aware_risk_adjusted_score": risk_penalized_score,
            "risk_aware_correct": risk_correct,
            "risk_aware_realized_utility": risk_realized,
            "predicted_mean_difference_risk_minus_utility": risk_mean - utility_mean,
            "risk_difference_risk_minus_utility": risk_risk - utility_risk,
            "penalty_difference_risk_minus_utility": risk_penalty - utility_penalty,
            "score_difference_risk_minus_utility": (
                risk_penalized_score - utility_penalized_score
            ),
            "realized_utility_difference_risk_minus_utility": (
                risk_realized - utility_realized
            ),
            "lower_risk_selected": lower_risk_selected,
            "lower_mean_selected": lower_mean_selected,
            "mean_ranking": " > ".join(mean_ranking),
            "risk_adjusted_ranking": " > ".join(score_ranking),
        }

        # Preserve candidate-level states for detailed inspection.
        for model in MODELS:
            output_row[f"{model}_predicted_mean"] = float(row[f"mean_utility_{model}"])
            output_row[f"{model}_risk"] = float(row[f"risk_{model}"])
            output_row[f"{model}_risk_adjusted_score"] = float(
                row[f"risk_adjusted_score_{model}"]
            )
            output_row[f"{model}_correct"] = actual_models[model]["correct"]
            output_row[f"{model}_realized_utility"] = actual_models[model]["utility"]

        output_rows.append(output_row)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    detailed_df = pd.DataFrame(output_rows)

    detailed_df = detailed_df.sort_values(
        "realized_utility_difference_risk_minus_utility",
        ascending=False,
    )

    detailed_df.to_csv(OUTPUT_FILE, index=False)

    # Aggregate mechanism summary used in the report.
    n = len(detailed_df)

    lower_risk_count = int(detailed_df["lower_risk_selected"].sum())
    lower_mean_count = int(detailed_df["lower_mean_selected"].sum())

    correctness_counts = Counter(detailed_df["correctness_transition"])

    summary = {
        "alpha": ALPHA,
        "lambda": float(detailed_df["lambda"].iloc[0]),
        "changed_decisions": n,
        "lower_risk_selected_count": lower_risk_count,
        "lower_risk_selected_rate": lower_risk_count / n,
        "lower_mean_selected_count": lower_mean_count,
        "lower_mean_selected_rate": lower_mean_count / n,
        "mean_predicted_mean_difference": float(
            detailed_df["predicted_mean_difference_risk_minus_utility"].mean()
        ),
        "mean_risk_difference": float(
            detailed_df["risk_difference_risk_minus_utility"].mean()
        ),
        "mean_penalty_difference": float(
            detailed_df["penalty_difference_risk_minus_utility"].mean()
        ),
        "mean_score_margin": float(
            detailed_df["score_difference_risk_minus_utility"].mean()
        ),
        "mean_realized_utility_difference": float(
            detailed_df["realized_utility_difference_risk_minus_utility"].mean()
        ),
        "wrong_to_correct": correctness_counts["wrong_to_correct"],
        "correct_to_wrong": correctness_counts["correct_to_wrong"],
        "both_correct": correctness_counts["both_correct"],
        "both_wrong": correctness_counts["both_wrong"],
    }

    pd.DataFrame([summary]).to_csv(SUMMARY_FILE, index=False)

    print(f"Saved risk-mechanism analysis to {RESULTS_DIR}")


if __name__ == "__main__":
    main()