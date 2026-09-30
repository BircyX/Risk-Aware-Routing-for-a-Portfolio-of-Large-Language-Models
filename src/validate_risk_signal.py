import csv
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    roc_auc_score = None

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "risk_aware_validation_predictions_aligned.csv"

OUTPUT_FILE = RESULTS_DIR / "risk_signal_validation_aligned.csv"

MODELS = ["gpt", "claude", "gemini", "qwen"]

ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]


def rankdata(values):
    values = np.asarray(values, dtype=float)

    order = np.argsort(values)

    ranks = np.empty(len(values), dtype=float)

    sorted_values = values[order]

    i = 0

    while i < len(values):
        j = i

        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1

        average_rank = (i + j) / 2.0

        ranks[order[i : j + 1]] = average_rank

        i = j + 1

    return ranks


def spearman_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return float("nan")

    x_rank = rankdata(x)
    y_rank = rankdata(y)

    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")

    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def load_rows():
    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            parsed = {
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "category": row["category"],
                "alpha": float(row["alpha"]),
                "decision_disagreement": float(row["decision_disagreement"]),
                "decision_entropy": float(row["decision_entropy"]),
                "models": {},
            }

            for model in MODELS:
                parsed["models"][model] = {
                    "mean": float(row[f"mean_utility_{model}"]),
                    "risk": float(row[f"risk_{model}"]),
                    "actual": float(row[f"actual_utility_{model}"]),
                }

            rows.append(parsed)

    return rows


def evaluate_model_level_uncertainty(rows):
    results = []

    for alpha in ALPHAS:
        alpha_rows = [row for row in rows if row["alpha"] == alpha]

        pooled_risks = []
        pooled_errors = []

        for row in alpha_rows:
            for model in MODELS:
                mean = row["models"][model]["mean"]
                actual = row["models"][model]["actual"]
                risk = row["models"][model]["risk"]

                pooled_risks.append(risk)
                pooled_errors.append(abs(mean - actual))

        results.append(
            {
                "analysis": "model_level_prediction_error",
                "alpha": alpha,
                "signal": "model_level_std_all_models",
                "n": len(pooled_risks),
                "spearman_rho": spearman_correlation(pooled_risks, pooled_errors),
            }
        )

        for model in MODELS:
            risks = []
            errors = []

            for row in alpha_rows:
                mean = row["models"][model]["mean"]
                actual = row["models"][model]["actual"]
                risk = row["models"][model]["risk"]

                risks.append(risk)
                errors.append(abs(mean - actual))

            risks = np.asarray(risks, dtype=float)
            errors = np.asarray(errors, dtype=float)

            order = np.argsort(risks)

            k = max(1, int(len(order) * 0.20))

            low = order[:k]
            high = order[-k:]

            results.append(
                {
                    "analysis": "model_level_prediction_error",
                    "alpha": alpha,
                    "signal": f"model_level_std_{model}",
                    "n": len(risks),
                    "spearman_rho": spearman_correlation(risks, errors),
                    "low_20_mean_error": float(np.mean(errors[low])),
                    "high_20_mean_error": float(np.mean(errors[high])),
                }
            )

    return results


def build_routing_records(rows):
    """
    Evaluate uncertainty around the utility-only
    routing decision, corresponding to lambda = 0.
    """
    routing_records = []

    for row in rows:
        model_stats = row["models"]

        selected_model = max(MODELS, key=lambda model: model_stats[model]["mean"])

        oracle_model = max(MODELS, key=lambda model: model_stats[model]["actual"])

        selected_actual = model_stats[selected_model]["actual"]
        oracle_actual = model_stats[oracle_model]["actual"]

        regret = oracle_actual - selected_actual

        routing_records.append(
            {
                "query_id": row["query_id"],
                "dataset": row["dataset"],
                "category": row["category"],
                "alpha": row["alpha"],
                "selected_model": selected_model,
                "oracle_model": oracle_model,
                "regret": regret,
                "selected_model_risk": model_stats[selected_model]["risk"],
                "decision_disagreement": row["decision_disagreement"],
                "decision_entropy": row["decision_entropy"],
            }
        )

    return routing_records


def evaluate_signal(values, regrets, failure_threshold):
    values = np.asarray(values, dtype=float)
    regrets = np.asarray(regrets, dtype=float)

    failures = (regrets > failure_threshold).astype(int)

    rho = spearman_correlation(values, regrets)

    auroc = float("nan")

    if roc_auc_score is not None and len(np.unique(failures)) == 2:
        auroc = float(roc_auc_score(failures, values))

    order = np.argsort(values)

    k = max(1, int(len(order) * 0.20))

    low = order[:k]
    high = order[-k:]

    return {
        "spearman_rho": rho,
        "auroc": auroc,
        "failure_rate": float(np.mean(failures)),
        "low_20_failure_rate": float(np.mean(failures[low])),
        "high_20_failure_rate": float(np.mean(failures[high])),
        "low_20_mean_regret": float(np.mean(regrets[low])),
        "high_20_mean_regret": float(np.mean(regrets[high])),
    }


def evaluate_routing_uncertainty(routing_records):
    results = []

    # Multiple thresholds distinguish any regret from
    # more practically meaningful routing failures.
    failure_thresholds = [0.0, 0.05, 0.10]

    signals = [
        ("selected_model_std", "selected_model_risk"),
        ("decision_disagreement", "decision_disagreement"),
        ("decision_entropy", "decision_entropy"),
    ]

    for alpha in ALPHAS:
        subset = [row for row in routing_records if row["alpha"] == alpha]

        regrets = [row["regret"] for row in subset]

        for signal_name, signal_column in signals:
            values = [row[signal_column] for row in subset]

            for threshold in failure_thresholds:
                metrics = evaluate_signal(values, regrets, threshold)

                results.append(
                    {
                        "analysis": "routing_uncertainty",
                        "alpha": alpha,
                        "signal": signal_name,
                        "failure_threshold": threshold,
                        "n": len(subset),
                        **metrics,
                    }
                )

    return results


def save_results(model_rows, routing_rows):
    rows = model_rows + routing_rows

    if not rows:
        raise ValueError("No risk-signal results to save.")

    columns = []

    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        writer.writeheader()

        for row in rows:
            output_row = {}

            for column in columns:
                value = row.get(column, "")

                if isinstance(value, float):
                    output_row[column] = round(value, 6)
                else:
                    output_row[column] = value

            writer.writerow(output_row)


def main():
    rows = load_rows()

    model_rows = evaluate_model_level_uncertainty(rows)

    routing_records = build_routing_records(rows)

    routing_rows = evaluate_routing_uncertainty(routing_records)

    save_results(model_rows, routing_rows)

    print(f"Saved risk-signal validation to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()