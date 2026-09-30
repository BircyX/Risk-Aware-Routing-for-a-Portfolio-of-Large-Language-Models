import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from xgboost import XGBRegressor

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_FILE = RESULTS_DIR / "query_features.csv"
UTILITY_FILE = RESULTS_DIR / "utility_outputs.csv"

MODEL_DIR = RESULTS_DIR / "models" / "risk_aware_aligned"

MEMBER_PREDICTION_FILE = RESULTS_DIR / "risk_aware_member_predictions.csv"

PREDICTION_FILE = RESULTS_DIR / "risk_aware_validation_predictions_aligned.csv"

DECISION_FILE = RESULTS_DIR / "risk_aware_validation_decisions_aligned.csv"

SUMMARY_FILE = RESULTS_DIR / "risk_aware_validation_summary_aligned.csv"

BEST_LAMBDA_FILE = RESULTS_DIR / "risk_aware_best_lambdas_aligned.json"

METADATA_FILE = RESULTS_DIR / "risk_aware_router_metadata_aligned.json"

MODELS = ["gpt", "claude", "gemini", "qwen"]

ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]

LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0]

ENSEMBLE_SIZE = 10
SEED = 42

NON_FEATURE_COLUMNS = {"query_id", "dataset", "category", "split", "question"}

XGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "n_jobs": 4,
    "early_stopping_rounds": 50,
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

    return feature_data, feature_columns


def load_utility_data():
    utility_data = {}

    with open(UTILITY_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            query_id = row["query_id"]
            model = row["model"]

            if query_id not in utility_data:
                utility_data[query_id] = {
                    "dataset": row["dataset"],
                    "category": row["category"],
                    "split": row["split"],
                    "models": {},
                }

            utility_data[query_id]["models"][model] = {
                "correct": int(row["correct"]),
                "cost": float(row["cost_clean"]),
                "latency": float(row["latency"]),
                "utilities": {
                    alpha: float(row[f"utility_alpha_{alpha}"]) for alpha in ALPHAS
                },
            }

    return utility_data


def validate_data(feature_data, utility_data):
    if set(feature_data) != set(utility_data):
        raise ValueError("Feature/utility IDs do not match.")

    for query_id in feature_data:
        if feature_data[query_id]["split"] != utility_data[query_id]["split"]:
            raise ValueError(f"Split mismatch: {query_id}")

        models = set(utility_data[query_id]["models"])

        if models != set(MODELS):
            raise ValueError(f"Incomplete model set for {query_id}")


def get_query_ids(feature_data, split):
    return sorted(
        query_id for query_id, row in feature_data.items() if row["split"] == split
    )


def build_x_matrix(query_ids, feature_data):
    return np.stack([feature_data[query_id]["features"] for query_id in query_ids])


def build_y_vector(query_ids, utility_data, model, alpha):
    return np.array(
        [
            utility_data[query_id]["models"][model]["utilities"][alpha]
            for query_id in query_ids
        ],
        dtype=np.float32,
    )


def create_aligned_bootstrap_indices(n_train):
    """
    Use the same bootstrap query sample for all candidate
    models within each ensemble member.
    """
    bootstrap_samples = []

    for member in range(ENSEMBLE_SIZE):
        rng = np.random.default_rng(SEED + member)
        indices = rng.integers(0, n_train, size=n_train)
        bootstrap_samples.append(indices)

    return bootstrap_samples


def train_ensembles(feature_data, utility_data):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ids = get_query_ids(feature_data, "train")
    validation_ids = get_query_ids(feature_data, "validation")

    x_train = build_x_matrix(train_ids, feature_data)
    x_val = build_x_matrix(validation_ids, feature_data)

    bootstrap_samples = create_aligned_bootstrap_indices(len(train_ids))

    predictions = {
        alpha: {member: {} for member in range(ENSEMBLE_SIZE)} for alpha in ALPHAS
    }

    for alpha in ALPHAS:
        y_by_model = {
            model: build_y_vector(train_ids, utility_data, model, alpha)
            for model in MODELS
        }

        y_val_by_model = {
            model: build_y_vector(validation_ids, utility_data, model, alpha)
            for model in MODELS
        }

        for member in range(ENSEMBLE_SIZE):
            bootstrap_indices = bootstrap_samples[member]
            x_bootstrap = x_train[bootstrap_indices]

            for model_index, model in enumerate(MODELS):
                y_bootstrap = y_by_model[model][bootstrap_indices]

                params = XGB_PARAMS.copy()
                params["random_state"] = SEED + member + model_index * 1000

                regressor = XGBRegressor(**params)

                regressor.fit(
                    x_bootstrap,
                    y_bootstrap,
                    eval_set=[(x_val, y_val_by_model[model])],
                    verbose=False,
                )

                predictions[alpha][member][model] = regressor.predict(x_val)

                model_dir = MODEL_DIR / f"alpha_{alpha}" / f"member_{member + 1}"
                model_dir.mkdir(parents=True, exist_ok=True)

                regressor.get_booster().save_model(model_dir / f"{model}.json")

    return validation_ids, predictions


def build_member_prediction_rows(validation_ids, predictions, feature_data):
    rows = []

    for alpha in ALPHAS:
        for member in range(ENSEMBLE_SIZE):
            for i, query_id in enumerate(validation_ids):
                model_predictions = {
                    model: float(predictions[alpha][member][model][i])
                    for model in MODELS
                }

                selected_model = max(MODELS, key=lambda model: model_predictions[model])

                row = {
                    "query_id": query_id,
                    "dataset": feature_data[query_id]["dataset"],
                    "category": feature_data[query_id]["category"],
                    "alpha": alpha,
                    "member": member + 1,
                    "selected_model": selected_model,
                }

                for model in MODELS:
                    row[f"predicted_utility_{model}"] = model_predictions[model]

                rows.append(row)

    return rows


def compute_statistics(validation_ids, predictions, feature_data, utility_data):
    """
    Compute ensemble mean utility prediction, population
    standard-deviation risk, and diagnostic vote measures.
    """
    stats = {alpha: {} for alpha in ALPHAS}

    rows = []

    for alpha in ALPHAS:
        for i, query_id in enumerate(validation_ids):
            stats[alpha][query_id] = {}

            votes = Counter()

            row = {
                "query_id": query_id,
                "dataset": feature_data[query_id]["dataset"],
                "category": feature_data[query_id]["category"],
                "alpha": alpha,
                "split": "validation",
            }

            for model in MODELS:
                member_values = np.array(
                    [
                        predictions[alpha][member][model][i]
                        for member in range(ENSEMBLE_SIZE)
                    ],
                    dtype=float,
                )

                mean_utility = float(np.mean(member_values))
                risk = float(np.std(member_values, ddof=0))

                stats[alpha][query_id][model] = {"mean": mean_utility, "risk": risk}

                row[f"mean_utility_{model}"] = mean_utility
                row[f"risk_{model}"] = risk
                row[f"actual_utility_{model}"] = utility_data[query_id]["models"][
                    model
                ]["utilities"][alpha]

            for member in range(ENSEMBLE_SIZE):
                member_choice = max(
                    MODELS, key=lambda model: predictions[alpha][member][model][i]
                )

                votes[member_choice] += 1

            max_vote_fraction = max(votes.values()) / ENSEMBLE_SIZE
            decision_disagreement = 1.0 - max_vote_fraction

            probabilities = np.array(
                [votes[model] / ENSEMBLE_SIZE for model in MODELS], dtype=float
            )

            nonzero = probabilities[probabilities > 0]

            entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(len(MODELS)))

            row["decision_disagreement"] = decision_disagreement
            row["decision_entropy"] = entropy

            for model in MODELS:
                row[f"vote_fraction_{model}"] = votes[model] / ENSEMBLE_SIZE

            stats[alpha][query_id]["decision_disagreement"] = decision_disagreement
            stats[alpha][query_id]["decision_entropy"] = entropy

            rows.append(row)

    return stats, rows


def evaluate_lambdas(validation_ids, stats, utility_data):
    decision_rows = []
    summary_rows = []

    for alpha in ALPHAS:
        for lambda_value in LAMBDAS:
            selected_models = []

            correctness = []
            costs = []
            latencies = []

            realized_utilities = []
            oracle_utilities = []
            regrets = []

            selected_risks = []
            decision_disagreements = []

            for query_id in validation_ids:
                query_stats = stats[alpha][query_id]

                scores = {
                    model: query_stats[model]["mean"]
                    - lambda_value * query_stats[model]["risk"]
                    for model in MODELS
                }

                selected_model = max(MODELS, key=lambda model: scores[model])

                actual_models = utility_data[query_id]["models"]

                realized_utility = actual_models[selected_model]["utilities"][alpha]

                oracle_model = max(
                    MODELS,
                    key=lambda model: actual_models[model]["utilities"][alpha],
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
                selected_risks.append(query_stats[selected_model]["risk"])
                decision_disagreements.append(query_stats["decision_disagreement"])

                decision_rows.append(
                    {
                        "query_id": query_id,
                        "alpha": alpha,
                        "lambda": lambda_value,
                        "selected_model": selected_model,
                        "selected_mean_utility": query_stats[selected_model]["mean"],
                        "selected_risk": query_stats[selected_model]["risk"],
                        "decision_disagreement": query_stats["decision_disagreement"],
                        "decision_entropy": query_stats["decision_entropy"],
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

            summary_rows.append(
                {
                    "alpha": alpha,
                    "lambda": lambda_value,
                    "num_queries": len(validation_ids),
                    "accuracy": float(np.mean(correctness)),
                    "average_cost": float(np.mean(costs)),
                    "average_latency": float(np.mean(latencies)),
                    "mean_realized_utility": float(np.mean(realized_utilities)),
                    "mean_oracle_utility": float(np.mean(oracle_utilities)),
                    "mean_regret": float(np.mean(regrets)),
                    "mean_selected_risk": float(np.mean(selected_risks)),
                    "mean_decision_disagreement": float(
                        np.mean(decision_disagreements)
                    ),
                    "gpt_selected": counts["gpt"],
                    "claude_selected": counts["claude"],
                    "gemini_selected": counts["gemini"],
                    "qwen_selected": counts["qwen"],
                }
            )

    return decision_rows, summary_rows


def choose_best_lambdas(summary_rows):
    """
    Select lambda using validation mean regret only.
    Smaller lambda breaks exact ties.
    """
    best = {}

    for alpha in ALPHAS:
        candidates = [row for row in summary_rows if row["alpha"] == alpha]

        winner = min(candidates, key=lambda row: (row["mean_regret"], row["lambda"]))

        best[str(alpha)] = winner

    return best


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


def save_json(object_, path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(object_, file, indent=4)


def main():
    feature_data, feature_columns = load_features()

    utility_data = load_utility_data()

    validate_data(feature_data, utility_data)

    validation_ids, predictions = train_ensembles(feature_data, utility_data)

    member_rows = build_member_prediction_rows(
        validation_ids, predictions, feature_data
    )

    stats, prediction_rows = compute_statistics(
        validation_ids, predictions, feature_data, utility_data
    )

    decision_rows, summary_rows = evaluate_lambdas(validation_ids, stats, utility_data)

    best_lambdas = choose_best_lambdas(summary_rows)

    save_csv(member_rows, MEMBER_PREDICTION_FILE)
    save_csv(prediction_rows, PREDICTION_FILE)
    save_csv(decision_rows, DECISION_FILE)
    save_csv(summary_rows, SUMMARY_FILE)
    save_json(best_lambdas, BEST_LAMBDA_FILE)

    metadata = {
        "ensemble_size": ENSEMBLE_SIZE,
        "aligned_bootstrap": True,
        "risk_definition": (
            "population standard deviation "
            "of model-level bootstrap "
            "utility predictions"
        ),
        "decision_disagreement_definition": ("1 - maximum ensemble vote fraction"),
        "decision_entropy_definition": (
            "normalized entropy of ensemble model-selection votes"
        ),
        "alphas": ALPHAS,
        "lambdas": LAMBDAS,
        "training_split": "train",
        "selection_split": "validation",
        "test_used": False,
        "feature_count": len(feature_columns),
        "random_seed": SEED,
        "xgboost_parameters": XGB_PARAMS,
    }

    save_json(metadata, METADATA_FILE)

    print(f"Saved risk-aware validation outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()