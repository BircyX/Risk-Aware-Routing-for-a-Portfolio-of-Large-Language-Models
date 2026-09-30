import csv
import json
import math
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

MODEL_DIR = RESULTS_DIR / "models" / "utility_only"

PREDICTION_FILE = RESULTS_DIR / "utility_router_validation_predictions.csv"

REGRESSION_METRICS_FILE = RESULTS_DIR / "utility_predictor_validation_metrics.csv"

ROUTING_SUMMARY_FILE = RESULTS_DIR / "utility_router_validation_summary.csv"

METADATA_FILE = RESULTS_DIR / "utility_router_metadata.json"

MODELS = ["gpt", "claude", "gemini", "qwen"]

ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]

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
    "random_state": SEED,
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
    feature_ids = set(feature_data)
    utility_ids = set(utility_data)

    if feature_ids != utility_ids:
        raise ValueError("Feature/utility query IDs do not match.")

    for query_id in feature_ids:
        if feature_data[query_id]["split"] != utility_data[query_id]["split"]:
            raise ValueError(f"Split mismatch for {query_id}")

        models = set(utility_data[query_id]["models"])

        if models != set(MODELS):
            raise ValueError(f"Incomplete model data for {query_id}: {models}")


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


def regression_metrics(y_true, y_pred):
    errors = y_true - y_pred

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    return mae, rmse


def train_models(feature_data, utility_data):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_ids = get_query_ids(feature_data, "train")
    validation_ids = get_query_ids(feature_data, "validation")

    x_train = build_x_matrix(train_ids, feature_data)
    x_validation = build_x_matrix(validation_ids, feature_data)

    predictions = {alpha: {} for alpha in ALPHAS}

    metric_rows = []

    for alpha in ALPHAS:
        for model_name in MODELS:
            y_train = build_y_vector(train_ids, utility_data, model_name, alpha)
            y_validation = build_y_vector(
                validation_ids, utility_data, model_name, alpha
            )

            regressor = XGBRegressor(**XGB_PARAMS)

            regressor.fit(
                x_train,
                y_train,
                eval_set=[(x_validation, y_validation)],
                verbose=False,
            )

            validation_prediction = regressor.predict(x_validation)

            mae, rmse = regression_metrics(y_validation, validation_prediction)

            best_iteration = getattr(regressor, "best_iteration", None)
            best_score = getattr(regressor, "best_score", None)

            regressor.get_booster().save_model(
                MODEL_DIR / f"alpha_{alpha}_{model_name}.json"
            )

            metric_rows.append(
                {
                    "alpha": alpha,
                    "model": model_name,
                    "mae": mae,
                    "rmse": rmse,
                    "best_iteration": best_iteration,
                    "best_score": best_score,
                }
            )

            for query_id, prediction in zip(validation_ids, validation_prediction):
                predictions[alpha].setdefault(query_id, {})
                predictions[alpha][query_id][model_name] = float(prediction)

    return validation_ids, predictions, metric_rows


def evaluate_router(validation_ids, predictions, feature_data, utility_data):
    prediction_rows = []
    summary_rows = []

    for alpha in ALPHAS:
        selected_models = []

        realized_utilities = []
        oracle_utilities = []
        regrets = []

        correctness = []
        costs = []
        latencies = []

        for query_id in validation_ids:
            model_predictions = predictions[alpha][query_id]

            selected_model = max(MODELS, key=lambda model: model_predictions[model])

            actual_models = utility_data[query_id]["models"]

            realized_utility = actual_models[selected_model]["utilities"][alpha]

            oracle_model = max(
                MODELS,
                key=lambda model: actual_models[model]["utilities"][alpha],
            )

            oracle_utility = actual_models[oracle_model]["utilities"][alpha]

            regret = oracle_utility - realized_utility

            selected_models.append(selected_model)
            realized_utilities.append(realized_utility)
            oracle_utilities.append(oracle_utility)
            regrets.append(regret)
            correctness.append(actual_models[selected_model]["correct"])
            costs.append(actual_models[selected_model]["cost"])
            latencies.append(actual_models[selected_model]["latency"])

            prediction_rows.append(
                {
                    "query_id": query_id,
                    "dataset": feature_data[query_id]["dataset"],
                    "category": feature_data[query_id]["category"],
                    "split": "validation",
                    "alpha": alpha,
                    "predicted_utility_gpt": model_predictions["gpt"],
                    "predicted_utility_claude": model_predictions["claude"],
                    "predicted_utility_gemini": model_predictions["gemini"],
                    "predicted_utility_qwen": model_predictions["qwen"],
                    "selected_model": selected_model,
                    "selected_predicted_utility": model_predictions[selected_model],
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
                "num_queries": len(validation_ids),
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
        )

    return prediction_rows, summary_rows


def save_csv(rows, path):
    if not rows:
        raise ValueError(f"No rows to save for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for row in rows:
            output_row = {}

            for key, value in row.items():
                if isinstance(value, float) and math.isfinite(value):
                    output_row[key] = round(value, 8)
                else:
                    output_row[key] = value

            writer.writerow(output_row)


def save_metadata(feature_columns):
    metadata = {
        "router_type": "utility-only regression",
        "training_split": "train",
        "validation_split": "validation",
        "test_used": False,
        "models": MODELS,
        "alphas": ALPHAS,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "xgboost_parameters": XGB_PARAMS,
        "decision_rule": "argmax predicted utility",
        "target_definition": ("utility = correctness - alpha * normalized_cost"),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def main():
    feature_data, feature_columns = load_features()

    utility_data = load_utility_data()

    validate_data(feature_data, utility_data)

    validation_ids, predictions, metric_rows = train_models(feature_data, utility_data)

    prediction_rows, summary_rows = evaluate_router(
        validation_ids, predictions, feature_data, utility_data
    )

    save_csv(metric_rows, REGRESSION_METRICS_FILE)
    save_csv(prediction_rows, PREDICTION_FILE)
    save_csv(summary_rows, ROUTING_SUMMARY_FILE)

    save_metadata(feature_columns)

    print(f"Saved utility-router validation outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()