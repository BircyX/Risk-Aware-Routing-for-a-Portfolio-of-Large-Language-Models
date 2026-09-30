import csv
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILES = [
    RESULTS_DIR / "gsm8k_evaluated.csv",
    RESULTS_DIR / "mmlu_evaluated.csv",
]


def load_rows():
    rows = []

    for input_file in INPUT_FILES:
        with open(input_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                row["cost"] = float(row["cost"])
                row["total_tokens"] = int(row["total_tokens"])

                rows.append(row)

    return rows


def percentile(values, p):
    values = sorted(values)

    if not values:
        raise ValueError("Cannot compute percentile of an empty list.")

    if len(values) == 1:
        return values[0]

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def analyze(rows):
    costs = [row["cost"] for row in rows]

    zero_cost = [row for row in rows if row["cost"] == 0]

    zero_cost_counts = Counter((row["dataset"], row["model"]) for row in zero_cost)

    most_expensive = sorted(rows, key=lambda row: row["cost"], reverse=True)[:10]

    print(f"Total rows: {len(rows)}")

    print("\nCost statistics:")
    print(
        f"min={min(costs):.8f}, "
        f"median={percentile(costs, 0.5):.8f}, "
        f"P90={percentile(costs, 0.90):.8f}, "
        f"P95={percentile(costs, 0.95):.8f}, "
        f"P99={percentile(costs, 0.99):.8f}, "
        f"P99.9={percentile(costs, 0.999):.8f}, "
        f"max={max(costs):.8f}"
    )

    print(f"\nZero-cost rows: {len(zero_cost)}")

    for (dataset, model), count in sorted(zero_cost_counts.items()):
        print(f"{dataset} / {model}: {count}")

    print("\nTop 10 most expensive requests:")

    for row in most_expensive:
        print(
            f"{row['dataset']} | {row['model']} | "
            f"${row['cost']:.8f} | tokens={row['total_tokens']}"
        )


def main():
    rows = load_rows()

    if not rows:
        raise ValueError("No rows found in input files.")

    analyze(rows)


if __name__ == "__main__":
    main()