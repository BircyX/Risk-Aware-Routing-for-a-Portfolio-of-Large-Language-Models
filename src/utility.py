import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILES = [
    RESULTS_DIR / "gsm8k_evaluated.csv",
    RESULTS_DIR / "mmlu_evaluated.csv",
]

SPLIT_FILE = RESULTS_DIR / "query_splits.csv"

OUTPUT_FILE = RESULTS_DIR / "utility_outputs.csv"

PARAMS_FILE = RESULTS_DIR / "cost_preprocessing_params.json"

ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]

CLIP_PERCENTILE = 0.99


def percentile(values, q):
    """
    Compute a linear-interpolated percentile.
    """
    values = sorted(values)

    if not values:
        raise ValueError("Cannot compute percentile of empty values.")

    if len(values) == 1:
        return values[0]

    position = (len(values) - 1) * q

    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return values[lower]

    weight = position - lower

    return values[lower] * (1 - weight) + values[upper] * weight


def load_split_mapping():
    split_mapping = {}

    with open(SPLIT_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            key = (row["dataset"], row["category"], row["question"])

            split_mapping[key] = {
                "query_id": row["query_id"],
                "split": row["split"],
            }

    return split_mapping


def load_rows(split_mapping):
    rows = []

    for input_file in INPUT_FILES:
        with open(input_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                key = (row["dataset"], row["category"], row["question"])

                if key not in split_mapping:
                    raise ValueError(f"Query not found in split file: {key}")

                row["query_id"] = split_mapping[key]["query_id"]
                row["split"] = split_mapping[key]["split"]
                row["raw_cost"] = float(row["cost"])
                row["correct"] = int(row["correct"])

                rows.append(row)

    return rows


def validate_split_integrity(rows):
    query_splits = defaultdict(set)
    query_models = defaultdict(set)

    for row in rows:
        query_id = row["query_id"]

        query_splits[query_id].add(row["split"])
        query_models[query_id].add(row["model"])

    for query_id, splits in query_splits.items():
        if len(splits) != 1:
            raise ValueError(f"Query {query_id} appears in multiple splits: {splits}")

    incomplete = [
        (query_id, models)
        for query_id, models in query_models.items()
        if len(models) != 4
    ]

    if incomplete:
        raise ValueError(
            f"{len(incomplete)} queries do not contain exactly four models."
        )


def compute_train_group_medians(rows):
    """
    Fit zero-cost imputation values on training data only.
    """
    grouped_costs = defaultdict(list)

    for row in rows:
        if row["split"] == "train" and row["raw_cost"] > 0:
            key = (row["dataset"], row["model"])
            grouped_costs[key].append(row["raw_cost"])

    return {key: median(values) for key, values in grouped_costs.items()}


def apply_cost_imputation(rows, train_medians):
    for row in rows:
        raw_cost = row["raw_cost"]

        row["cost_imputed"] = 0

        if raw_cost <= 0:
            key = (row["dataset"], row["model"])

            if key not in train_medians:
                raise ValueError(f"No training median available for {key}.")

            row["cost_clean"] = train_medians[key]
            row["cost_imputed"] = 1

        else:
            row["cost_clean"] = raw_cost


def compute_train_clip_threshold(rows):
    """
    Fit the P99 clipping threshold using cleaned train costs only.
    """
    train_costs = [row["cost_clean"] for row in rows if row["split"] == "train"]

    return percentile(train_costs, CLIP_PERCENTILE)


def apply_winsorization(rows, clip_threshold):
    for row in rows:
        clean_cost = row["cost_clean"]

        if clean_cost > clip_threshold:
            row["cost_clipped"] = clip_threshold
            row["cost_winsorized"] = 1
        else:
            row["cost_clipped"] = clean_cost
            row["cost_winsorized"] = 0


def compute_train_normalization_params(rows):
    """
    Fit min-max normalization using processed train costs only.
    """
    train_costs = [row["cost_clipped"] for row in rows if row["split"] == "train"]

    return min(train_costs), max(train_costs)


def apply_normalization(rows, train_min, train_max):
    """
    Apply train-derived normalization to every split.

    Validation and test values outside the training range
    are bounded to [0, 1].
    """
    denominator = train_max - train_min

    for row in rows:
        if denominator == 0:
            normalized = 0.0
        else:
            normalized = (row["cost_clipped"] - train_min) / denominator

        bounded = max(0.0, min(1.0, normalized))

        row["normalized_cost"] = bounded
        row["normalization_clipped"] = int(bounded != normalized)


def compute_utilities(rows):
    """
    Compute U = correctness - alpha * normalized_cost.
    """
    for row in rows:
        correct = row["correct"]
        normalized_cost = row["normalized_cost"]

        for alpha in ALPHAS:
            row[f"utility_alpha_{alpha}"] = correct - alpha * normalized_cost


def save_rows(rows):
    original_columns = [
        column
        for column in rows[0].keys()
        if column
        not in {
            "raw_cost",
            "cost_clean",
            "cost_clipped",
            "normalized_cost",
            "cost_imputed",
            "cost_winsorized",
            "normalization_clipped",
        }
        and not column.startswith("utility_alpha_")
    ]

    added_columns = [
        "raw_cost",
        "cost_imputed",
        "cost_clean",
        "cost_winsorized",
        "cost_clipped",
        "normalized_cost",
        "normalization_clipped",
    ]

    utility_columns = [f"utility_alpha_{alpha}" for alpha in ALPHAS]

    fieldnames = original_columns + added_columns + utility_columns

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for row in rows:
            output_row = row.copy()

            for column in ["raw_cost", "cost_clean", "cost_clipped"]:
                output_row[column] = round(float(output_row[column]), 8)

            output_row["normalized_cost"] = round(
                float(output_row["normalized_cost"]), 6
            )

            for column in utility_columns:
                output_row[column] = round(float(output_row[column]), 6)

            writer.writerow({field: output_row[field] for field in fieldnames})


def save_params(train_medians, clip_threshold, train_min, train_max):
    median_dict = {
        f"{dataset}__{model}": value
        for (dataset, model), value in sorted(train_medians.items())
    }

    params = {
        "preprocessing_fit_split": "train",
        "clip_percentile": CLIP_PERCENTILE,
        "clip_threshold": clip_threshold,
        "normalization_min": train_min,
        "normalization_max": train_max,
        "zero_cost_imputation": (
            "median cost of the same "
            "dataset-model group, "
            "estimated from training "
            "data only"
        ),
        "winsorization": "upper-tail clipping at training-set P99",
        "normalization": (
            "train-derived min-max "
            "normalization, with "
            "validation/test values "
            "bounded to [0, 1]"
        ),
        "utility_definition": "correct - alpha * normalized_cost",
        "alphas": ALPHAS,
        "train_group_medians": median_dict,
    }

    with open(PARAMS_FILE, "w", encoding="utf-8") as file:
        json.dump(params, file, indent=4)


def main():
    split_mapping = load_split_mapping()

    rows = load_rows(split_mapping)

    validate_split_integrity(rows)

    train_medians = compute_train_group_medians(rows)

    apply_cost_imputation(rows, train_medians)

    clip_threshold = compute_train_clip_threshold(rows)

    apply_winsorization(rows, clip_threshold)

    train_min, train_max = compute_train_normalization_params(rows)

    apply_normalization(rows, train_min, train_max)

    compute_utilities(rows)

    save_rows(rows)

    save_params(train_medians, clip_threshold, train_min, train_max)

    print(f"Saved utility data and preprocessing parameters to {RESULTS_DIR}")


if __name__ == "__main__":
    main()