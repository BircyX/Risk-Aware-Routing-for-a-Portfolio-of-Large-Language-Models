import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import xgboost as xgb

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_FILE = RESULTS_DIR / "query_features.csv"
UTILITY_FILE = RESULTS_DIR / "utility_outputs.csv"

MODEL_DIR = RESULTS_DIR / "models" / "risk_aware_aligned"
BEST_LAMBDA_FILE = RESULTS_DIR / "risk_aware_best_lambdas_aligned.json"
OUTPUT_FILE = RESULTS_DIR / "final_test_results.csv"
DECISION_FILE = RESULTS_DIR / "final_test_decisions.csv"
PREDICTION_FILE = RESULTS_DIR / "final_test_predictions_aligned.csv"

MODELS = ["gpt", "claude", "gemini", "qwen"]
ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]

ENSEMBLE_SIZE = 10

NON_FEATURE_COLUMNS = {
    "query_id",
    "dataset",
    "category",
    "split",
    "question",
}


def load_features():
    feature_data = {}

    with open(FEATURE_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        feature_columns = [
            column for column in reader.fieldnames if column not in NON_FEATURE_COLUMNS
        ]

        for row in reader:
            feature_data[row["query_id"]] = {
                "dataset": row["dataset"],
                "category": row["category"],
                "split": row["split"],
                "features": np.array(
                    [float(row[column]) for column in feature_columns],
                    dtype=np.float32,
                ),
            }

    return feature_data


def load_utility_data():
    utility_data = {}

    with open(UTILITY_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            query_id = row["query_id"]
            model = row["model"]

            if query_id not in utility_data:
                utility_data[query_id] = {"split": row["split"], "models": {}}

            utility_data[query_id]["models"][model] = {
                "correct": int(row["correct"]),
                "cost": float(row["cost_clean"]),
                "latency": float(row["latency"]),
                "utilities": {
                    alpha: float(row[f"utility_alpha_{alpha}"]) for alpha in ALPHAS
                },
            }

    return utility_data


def load_best_lambdas():
    """
    Load lambda values selected on validation data.
    """
    with open(BEST_LAMBDA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    return {float(alpha): float(row["lambda"]) for alpha, row in data.items()}


def get_test_ids(feature_data):
    return sorted(
        query_id for query_id, row in feature_data.items() if row["split"] == "test"
    )


def build_x_test(test_ids, feature_data):
    return np.stack([feature_data[query_id]["features"] for query_id in test_ids])


def load_ensemble_predictions(alpha, x_test):
    """
    Load predictions from the trained aligned bootstrap ensemble.
    No training or model selection is performed here.
    """
    predictions = {member: {} for member in range(ENSEMBLE_SIZE)}

    dtest = xgb.DMatrix(x_test)

    for member in range(ENSEMBLE_SIZE):
        for model in MODELS:
            model_path = (
                MODEL_DIR / f"alpha_{alpha}" / f"member_{member + 1}" / f"{model}.json"
            )

            if not model_path.exists():
                raise FileNotFoundError(f"Missing model: {model_path}")

            booster = xgb.Booster()
            booster.load_model(model_path)

            best_iteration = getattr(booster, "best_iteration", None)

            if best_iteration is not None:
                prediction = booster.predict(
                    dtest, iteration_range=(0, best_iteration + 1)
                )
            else:
                prediction = booster.predict(dtest)

            predictions[member][model] = prediction

    return predictions


def compute_stats(test_ids, predictions):
    """
    Compute ensemble mean prediction and predictive risk.

    Risk is the population standard deviation across the
    K ensemble members.
    """
    stats = {}

    for i, query_id in enumerate(test_ids):
        stats[query_id] = {}

        for model in MODELS:
            member_values = np.array(
                [predictions[member][model][i] for member in range(ENSEMBLE_SIZE)],
                dtype=float,
            )

            stats[query_id][model] = {
                "mean": float(np.mean(member_values)),
                "risk": float(np.std(member_values, ddof=0)),
            }

    return stats


def build_prediction_rows(test_ids, stats, feature_data, alpha, lambda_value):
    """
    Save the complete four-model routing state for
    mechanism analysis without changing test decisions.
    """
    rows = []

    for query_id in test_ids:
        query_stats = stats[query_id]

        row = {
            "query_id": query_id,
            "dataset": feature_data[query_id]["dataset"],
            "category": feature_data[query_id]["category"],
            "alpha": alpha,
            "lambda": lambda_value,
        }

        for model in MODELS:
            mean_utility = query_stats[model]["mean"]
            risk = query_stats[model]["risk"]
            risk_adjusted_score = mean_utility - lambda_value * risk

            row[f"mean_utility_{model}"] = mean_utility
            row[f"risk_{model}"] = risk
            row[f"risk_adjusted_score_{model}"] = risk_adjusted_score

        utility_only_model = max(MODELS, key=lambda model: query_stats[model]["mean"])

        risk_aware_model = max(
            MODELS,
            key=lambda model: (
                query_stats[model]["mean"] - lambda_value * query_stats[model]["risk"]
            ),
        )

        row["utility_only_model"] = utility_only_model
        row["risk_aware_model"] = risk_aware_model
        row["decision_changed"] = int(utility_only_model != risk_aware_model)

        utility_mean = query_stats[utility_only_model]["mean"]
        utility_risk = query_stats[utility_only_model]["risk"]
        risk_mean = query_stats[risk_aware_model]["mean"]
        risk_risk = query_stats[risk_aware_model]["risk"]

        row["utility_only_selected_mean"] = utility_mean
        row["utility_only_selected_risk"] = utility_risk
        row["risk_aware_selected_mean"] = risk_mean
        row["risk_aware_selected_risk"] = risk_risk

        row["selected_mean_difference_risk_minus_utility"] = risk_mean - utility_mean
        row["selected_risk_difference_risk_minus_utility"] = risk_risk - utility_risk

        utility_score = utility_mean - lambda_value * utility_risk
        risk_score = risk_mean - lambda_value * risk_risk

        row["utility_only_model_risk_adjusted_score"] = utility_score
        row["risk_aware_model_risk_adjusted_score"] = risk_score
        row["risk_adjusted_score_margin"] = risk_score - utility_score

        rows.append(row)

    return rows


def evaluate_router(test_ids, stats, utility_data, alpha, lambda_value):
    correctness = []
    costs = []
    latencies = []

    realized_utilities = []
    oracle_utilities = []
    regrets = []

    selected_models = []
    decisions = []

    for query_id in test_ids:
        query_stats = stats[query_id]

        scores = {
            model: (
                query_stats[model]["mean"] - lambda_value * query_stats[model]["risk"]
            )
            for model in MODELS
        }

        selected_model = max(MODELS, key=lambda model: scores[model])

        actual_models = utility_data[query_id]["models"]

        realized_utility = actual_models[selected_model]["utilities"][alpha]

        oracle_model = max(
            MODELS, key=lambda model: actual_models[model]["utilities"][alpha]
        )

        oracle_utility = actual_models[oracle_model]["utilities"][alpha]

        regret = oracle_utility - realized_utility

        selected_models.append(selected_model)
        correctness.append(actual_models[selected_model]["correct"])
        costs.append(actual_models[selected_model]["cost"])
        latencies.append(actual_models[selected_model]["latency"])
        realized_utilities.append(realized_utility)
        oracle_utilities.append(oracle_utility)
        regrets.append(regret)

        decisions.append(
            {
                "query_id": query_id,
                "alpha": alpha,
                "lambda": lambda_value,
                "selected_model": selected_model,
                "selected_mean_utility": query_stats[selected_model]["mean"],
                "selected_risk": query_stats[selected_model]["risk"],
                "realized_utility": realized_utility,
                "oracle_model": oracle_model,
                "oracle_utility": oracle_utility,
                "regret": regret,
                "correct": actual_models[selected_model]["correct"],
                "cost": actual_models[selected_model]["cost"],
                "latency": actual_models[selected_model]["latency"],
            }
        )

    counts = Counter(selected_models)

    summary = {
        "alpha": alpha,
        "lambda": lambda_value,
        "num_queries": len(test_ids),
        "accuracy": float(np.mean(correctness)),
        "average_cost": float(np.mean(costs)),
        "average_latency": float(np.mean(latencies)),
        "mean_realized_utility": float(np.mean(realized_utilities)),
        "mean_oracle_utility": float(np.mean(oracle_utilities)),
        "mean_regret": float(np.mean(regrets)),
        "gpt_selected": counts["gpt"],
        "claude_selected": counts["claude"],
        "gemini_selected": counts["gemini"],
        "qwen_selected": counts["qwen"],
    }

    return summary, decisions


def save_csv(rows, path):
    if not rows:
        raise ValueError(f"No rows to save for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            output_row = {
                key: (round(value, 8) if isinstance(value, float) else value)
                for key, value in row.items()
            }

            writer.writerow(output_row)


def main():
    feature_data = load_features()
    utility_data = load_utility_data()
    best_lambdas = load_best_lambdas()

    test_ids = get_test_ids(feature_data)
    x_test = build_x_test(test_ids, feature_data)

    summary_rows = []
    decision_rows = []
    prediction_rows = []

    for alpha in ALPHAS:
        predictions = load_ensemble_predictions(alpha, x_test)
        stats = compute_stats(test_ids, predictions)

        # Utility-only uses the same ensemble with lambda = 0.
        utility_summary, utility_decisions = evaluate_router(
            test_ids, stats, utility_data, alpha, 0.0
        )

        utility_summary["router"] = "utility_only"

        for row in utility_decisions:
            row["router"] = "utility_only"

        summary_rows.append(utility_summary)
        decision_rows.extend(utility_decisions)

        # Risk-aware uses lambda selected and frozen on validation.
        best_lambda = best_lambdas[alpha]

        risk_summary, risk_decisions = evaluate_router(
            test_ids, stats, utility_data, alpha, best_lambda
        )

        risk_summary["router"] = "risk_aware"

        for row in risk_decisions:
            row["router"] = "risk_aware"

        summary_rows.append(risk_summary)
        decision_rows.extend(risk_decisions)

        prediction_rows.extend(
            build_prediction_rows(
                test_ids=test_ids,
                stats=stats,
                feature_data=feature_data,
                alpha=alpha,
                lambda_value=best_lambda,
            )
        )

    save_csv(summary_rows, OUTPUT_FILE)
    save_csv(decision_rows, DECISION_FILE)
    save_csv(prediction_rows, PREDICTION_FILE)

    print(f"Saved final test evaluation to {RESULTS_DIR}")


if __name__ == "__main__":
    main()