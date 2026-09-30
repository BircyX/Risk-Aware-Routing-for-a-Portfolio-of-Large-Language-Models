import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

UTILITY_FILE = RESULTS_DIR / "utility_outputs.csv"
ROUTER_DECISION_FILE = RESULTS_DIR / "final_test_decisions.csv"

OUTPUT_SUMMARY = RESULTS_DIR / "final_category_analysis.csv"
OUTPUT_TRANSITIONS = RESULTS_DIR / "final_category_transitions.csv"

ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]


def load_utility_data():
    """
    Load realised test outcomes for each query-model pair.
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
                    "models": {},
                }

            data[query_id]["models"][model] = {
                "correct": int(row["correct"]),
                "cost": float(row["cost_clean"]),
                "latency": float(row["latency"]),
                "utilities": {
                    alpha: float(row[f"utility_alpha_{alpha}"]) for alpha in ALPHAS
                },
            }

    return data


def load_router_decisions():
    """
    Load final utility-only and risk-aware model selections.
    """
    decisions = defaultdict(lambda: defaultdict(dict))

    with open(ROUTER_DECISION_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            query_id = row["query_id"]
            alpha = float(row["alpha"])

            router_name = (
                row.get("router") or row.get("method") or row.get("policy") or ""
            )

            router_name = router_name.lower().strip()

            selected_model = row["selected_model"]

            if "risk" in router_name:
                key = "risk_aware"
            elif (
                "utility" in router_name
                or "lambda=0" in router_name
                or "lambda_0" in router_name
            ):
                key = "utility_only"
            else:
                continue

            decisions[alpha][query_id][key] = selected_model

    return decisions


def get_group_labels(query_row):
    """
    Return the category groups used in the report:
    overall, dataset, and MMLU subject.
    """
    dataset = query_row["dataset"].strip()
    category = query_row["category"].strip()

    labels = [
        ("overall", "overall"),
        ("dataset", dataset),
    ]

    if dataset.lower() == "mmlu" and category:
        labels.append(("mmlu_subject", category))

    return labels


def evaluate_group(query_ids, alpha, utility_data, decisions):
    utility_correct = []
    risk_correct = []

    utility_cost = []
    risk_cost = []

    utility_latency = []
    risk_latency = []

    utility_values = []
    risk_values = []

    utility_regret = []
    risk_regret = []

    same_decision = 0
    changed_decision = 0

    utility_only_better = 0
    risk_aware_better = 0
    equal_utility = 0

    for query_id in query_ids:
        decision = decisions[alpha].get(query_id, {})

        if "utility_only" not in decision or "risk_aware" not in decision:
            continue

        models = utility_data[query_id]["models"]

        utility_model = decision["utility_only"]
        risk_model = decision["risk_aware"]

        utility_row = models[utility_model]
        risk_row = models[risk_model]

        utility_u = utility_row["utilities"][alpha]
        risk_u = risk_row["utilities"][alpha]

        oracle_utility = max(models[model]["utilities"][alpha] for model in models)

        utility_correct.append(utility_row["correct"])
        risk_correct.append(risk_row["correct"])

        utility_cost.append(utility_row["cost"])
        risk_cost.append(risk_row["cost"])

        utility_latency.append(utility_row["latency"])
        risk_latency.append(risk_row["latency"])

        utility_values.append(utility_u)
        risk_values.append(risk_u)

        utility_regret.append(oracle_utility - utility_u)
        risk_regret.append(oracle_utility - risk_u)

        if utility_model == risk_model:
            same_decision += 1
        else:
            changed_decision += 1

        if risk_u > utility_u:
            risk_aware_better += 1
        elif utility_u > risk_u:
            utility_only_better += 1
        else:
            equal_utility += 1

    n = len(utility_values)

    if n == 0:
        return None

    utility_accuracy = sum(utility_correct) / n
    risk_accuracy = sum(risk_correct) / n

    utility_avg_cost = sum(utility_cost) / n
    risk_avg_cost = sum(risk_cost) / n

    utility_avg_latency = sum(utility_latency) / n
    risk_avg_latency = sum(risk_latency) / n

    utility_mean = sum(utility_values) / n
    risk_mean = sum(risk_values) / n

    utility_mean_regret = sum(utility_regret) / n
    risk_mean_regret = sum(risk_regret) / n

    return {
        "n_queries": n,
        "utility_only_accuracy": utility_accuracy,
        "risk_aware_accuracy": risk_accuracy,
        "accuracy_difference_risk_minus_utility": risk_accuracy - utility_accuracy,
        "utility_only_average_cost": utility_avg_cost,
        "risk_aware_average_cost": risk_avg_cost,
        "cost_difference_risk_minus_utility": risk_avg_cost - utility_avg_cost,
        "utility_only_average_latency": utility_avg_latency,
        "risk_aware_average_latency": risk_avg_latency,
        "utility_only_mean_utility": utility_mean,
        "risk_aware_mean_utility": risk_mean,
        "utility_difference_risk_minus_utility": risk_mean - utility_mean,
        "utility_only_mean_regret": utility_mean_regret,
        "risk_aware_mean_regret": risk_mean_regret,
        "regret_difference_risk_minus_utility": (
            risk_mean_regret - utility_mean_regret
        ),
        "same_decision_count": same_decision,
        "changed_decision_count": changed_decision,
        "changed_decision_rate": changed_decision / n,
        "risk_aware_better_count": risk_aware_better,
        "utility_only_better_count": utility_only_better,
        "equal_utility_count": equal_utility,
    }


def build_transition_rows(utility_data, decisions):
    """
    Summarise model-to-model changes made by the risk penalty.
    """
    rows = []

    for alpha in ALPHAS:
        transition_counts = defaultdict(
            lambda: {
                "count": 0,
                "risk_better": 0,
                "utility_better": 0,
                "equal": 0,
                "mean_utility_delta_sum": 0.0,
                "mean_regret_delta_sum": 0.0,
            }
        )

        for query_id, query_row in utility_data.items():
            decision = decisions[alpha].get(query_id, {})

            if "utility_only" not in decision or "risk_aware" not in decision:
                continue

            utility_model = decision["utility_only"]
            risk_model = decision["risk_aware"]

            if utility_model == risk_model:
                continue

            models = query_row["models"]

            utility_u = models[utility_model]["utilities"][alpha]
            risk_u = models[risk_model]["utilities"][alpha]

            oracle_utility = max(models[model]["utilities"][alpha] for model in models)

            utility_regret = oracle_utility - utility_u
            risk_regret = oracle_utility - risk_u

            transition = (utility_model, risk_model)

            stats = transition_counts[transition]

            stats["count"] += 1

            delta_u = risk_u - utility_u
            delta_regret = risk_regret - utility_regret

            stats["mean_utility_delta_sum"] += delta_u
            stats["mean_regret_delta_sum"] += delta_regret

            if delta_u > 0:
                stats["risk_better"] += 1
            elif delta_u < 0:
                stats["utility_better"] += 1
            else:
                stats["equal"] += 1

        for (utility_model, risk_model), stats in transition_counts.items():
            count = stats["count"]

            rows.append(
                {
                    "alpha": alpha,
                    "utility_only_model": utility_model,
                    "risk_aware_model": risk_model,
                    "count": count,
                    "risk_better_count": stats["risk_better"],
                    "utility_better_count": stats["utility_better"],
                    "equal_count": stats["equal"],
                    "mean_utility_difference": (
                        stats["mean_utility_delta_sum"] / count
                    ),
                    "mean_regret_difference": (stats["mean_regret_delta_sum"] / count),
                }
            )

    return rows


def main():
    utility_data = load_utility_data()
    decisions = load_router_decisions()

    summary_rows = []

    for alpha in ALPHAS:
        groups = defaultdict(list)

        for query_id, query_row in utility_data.items():
            for group_type, group_name in get_group_labels(query_row):
                groups[(group_type, group_name)].append(query_id)

        for (group_type, group_name), query_ids in groups.items():
            result = evaluate_group(
                query_ids=query_ids,
                alpha=alpha,
                utility_data=utility_data,
                decisions=decisions,
            )

            if result is None:
                continue

            summary_rows.append(
                {
                    "alpha": alpha,
                    "group_type": group_type,
                    "group_name": group_name,
                    **result,
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    transition_df = pd.DataFrame(build_transition_rows(utility_data, decisions))
    transition_df.to_csv(OUTPUT_TRANSITIONS, index=False)

    print(f"Saved category-level analysis to {RESULTS_DIR}")


if __name__ == "__main__":
    main()