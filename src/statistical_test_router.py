import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

UTILITY_FILE = RESULTS_DIR / "utility_outputs.csv"
ROUTER_DECISION_FILE = RESULTS_DIR / "final_test_decisions.csv"
OUTPUT_FILE = RESULTS_DIR / "final_statistical_tests.csv"

ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]

N_BOOTSTRAP = 10000
SEED = 42


def load_utility_data():
    data = {}

    with open(UTILITY_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["split"] != "test":
                continue

            query_id = row["query_id"]
            model = row["model"]

            if query_id not in data:
                data[query_id] = {}

            data[query_id][model] = {
                "correct": int(row["correct"]),
                "cost": float(row["cost_clean"]),
                "utilities": {
                    alpha: float(row[f"utility_alpha_{alpha}"]) for alpha in ALPHAS
                },
            }

    return data


def load_router_decisions():
    decisions = {}

    with open(ROUTER_DECISION_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            query_id = row["query_id"]
            alpha = float(row["alpha"])

            router_name = row.get("router") or row.get("policy") or row.get("method")

            if router_name is None:
                raise KeyError(
                    "Could not find router/policy/method column in final_test_decisions.csv"
                )

            router_name = router_name.lower().strip()

            decisions.setdefault(alpha, {})
            decisions[alpha].setdefault(query_id, {})

            selected_model = row["selected_model"]

            if "risk" in router_name and "utility" not in router_name:
                key = "risk_aware"
            elif (
                "utility" in router_name
                or "lambda=0" in router_name
                or "lambda_0" in router_name
            ):
                key = "utility_only"
            elif router_name in {"risk_aware", "risk-aware"}:
                key = "risk_aware"
            else:
                continue

            decisions[alpha][query_id][key] = selected_model

    return decisions


def paired_bootstrap_ci(a, b, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """
    Return the observed mean difference b - a
    and its percentile bootstrap 95% confidence interval.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    differences = b - a

    observed = float(np.mean(differences))

    rng = np.random.default_rng(seed)

    n = len(differences)

    bootstrap_means = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = np.mean(differences[indices])

    lower = float(np.percentile(bootstrap_means, 2.5))
    upper = float(np.percentile(bootstrap_means, 97.5))

    return observed, lower, upper


def exact_mcnemar(utility_correct, risk_correct):
    utility_correct = np.asarray(utility_correct, dtype=int)
    risk_correct = np.asarray(risk_correct, dtype=int)

    # Discordant pairs only.
    utility_win = int(np.sum((utility_correct == 1) & (risk_correct == 0)))
    risk_win = int(np.sum((utility_correct == 0) & (risk_correct == 1)))

    discordant = utility_win + risk_win

    if discordant == 0:
        p_value = 1.0
    else:
        result = binomtest(
            min(utility_win, risk_win),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )

        p_value = float(result.pvalue)

    return utility_win, risk_win, p_value


def main():
    utility_data = load_utility_data()
    router_decisions = load_router_decisions()

    output_rows = []

    for alpha in ALPHAS:
        utility_correct = []
        risk_correct = []

        utility_values = []
        risk_values = []

        utility_regrets = []
        risk_regrets = []

        common_queries = []

        for query_id in sorted(router_decisions[alpha].keys()):
            decision = router_decisions[alpha][query_id]

            if "utility_only" not in decision or "risk_aware" not in decision:
                continue

            common_queries.append(query_id)

            utility_model = decision["utility_only"]
            risk_model = decision["risk_aware"]

            models = utility_data[query_id]

            oracle_utility = max(models[model]["utilities"][alpha] for model in models)

            utility_u = models[utility_model]["utilities"][alpha]
            risk_u = models[risk_model]["utilities"][alpha]

            utility_correct.append(models[utility_model]["correct"])
            risk_correct.append(models[risk_model]["correct"])

            utility_values.append(utility_u)
            risk_values.append(risk_u)

            utility_regrets.append(oracle_utility - utility_u)
            risk_regrets.append(oracle_utility - risk_u)

        accuracy_diff, accuracy_low, accuracy_high = paired_bootstrap_ci(
            utility_correct, risk_correct, seed=SEED
        )

        utility_diff, utility_low, utility_high = paired_bootstrap_ci(
            utility_values, risk_values, seed=SEED + 1
        )

        # Negative Risk - Utility regret means lower
        # regret for the risk-aware router.
        regret_diff, regret_low, regret_high = paired_bootstrap_ci(
            utility_regrets, risk_regrets, seed=SEED + 2
        )

        utility_win, risk_win, mcnemar_p = exact_mcnemar(utility_correct, risk_correct)

        output_rows.append(
            {
                "alpha": alpha,
                "n_queries": len(common_queries),
                "accuracy_difference_risk_minus_utility": accuracy_diff,
                "accuracy_ci_lower": accuracy_low,
                "accuracy_ci_upper": accuracy_high,
                "utility_difference_risk_minus_utility": utility_diff,
                "utility_ci_lower": utility_low,
                "utility_ci_upper": utility_high,
                "regret_difference_risk_minus_utility": regret_diff,
                "regret_ci_lower": regret_low,
                "regret_ci_upper": regret_high,
                "utility_only_accuracy_wins": utility_win,
                "risk_aware_accuracy_wins": risk_win,
                "mcnemar_p_value": mcnemar_p,
            }
        )

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(output_rows[0].keys()))

        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Saved statistical tests to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()