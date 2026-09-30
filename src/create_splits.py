import csv
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILES = [
    RESULTS_DIR / "gsm8k_evaluated.csv",
    RESULTS_DIR / "mmlu_evaluated.csv",
]

OUTPUT_FILE = RESULTS_DIR / "query_splits.csv"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def make_query_id(dataset, category, question):
    """
    Create a stable identifier for each unique query.
    """
    text = f"{dataset}||{category}||{question}"

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_unique_queries():
    """
    Extract unique queries from the evaluated model outputs.
    """
    queries = {}

    for input_file in INPUT_FILES:
        with open(input_file, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            for row in reader:
                key = (row["dataset"], row["category"], row["question"])

                if key not in queries:
                    queries[key] = {
                        "query_id": make_query_id(
                            row["dataset"], row["category"], row["question"]
                        ),
                        "dataset": row["dataset"],
                        "category": row["category"],
                        "question": row["question"],
                        "ground_truth": row["ground_truth"],
                    }

    return list(queries.values())


def stratified_query_split(queries):
    """
    Split queries by (dataset, category), keeping all model
    outputs for the same query in the same split.
    """
    strata = defaultdict(list)

    for query in queries:
        stratum = (query["dataset"], query["category"])
        strata[stratum].append(query)

    split_queries = []
    rng = random.Random(SEED)

    for stratum in sorted(strata.keys()):
        group = sorted(strata[stratum], key=lambda row: row["query_id"])

        rng.shuffle(group)

        n = len(group)

        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = n - n_train - n_val

        train_queries = group[:n_train]
        val_queries = group[n_train : n_train + n_val]
        test_queries = group[n_train + n_val :]

        for query in train_queries:
            query["split"] = "train"
            split_queries.append(query)

        for query in val_queries:
            query["split"] = "validation"
            split_queries.append(query)

        for query in test_queries:
            query["split"] = "test"
            split_queries.append(query)

    return split_queries


def validate_splits(split_queries):
    query_ids = [row["query_id"] for row in split_queries]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query_id detected.")

    valid_splits = {"train", "validation", "test"}

    for row in split_queries:
        if row["split"] not in valid_splits:
            raise ValueError(f"Invalid split: {row['split']}")


def save_splits(split_queries):
    fieldnames = [
        "query_id",
        "dataset",
        "category",
        "question",
        "ground_truth",
        "split",
    ]

    split_order = {"train": 0, "validation": 1, "test": 2}

    split_queries = sorted(
        split_queries,
        key=lambda row: (
            split_order[row["split"]],
            row["dataset"],
            row["category"],
            row["query_id"],
        ),
    )

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(split_queries)


def main():
    queries = load_unique_queries()
    split_queries = stratified_query_split(queries)

    validate_splits(split_queries)
    save_splits(split_queries)

    counts = {
        split: sum(row["split"] == split for row in split_queries)
        for split in ["train", "validation", "test"]
    }

    print(f"Saved {len(split_queries)} queries to {OUTPUT_FILE}")

    print(
        f"train={counts['train']}, "
        f"validation={counts['validation']}, "
        f"test={counts['test']}"
    )


if __name__ == "__main__":
    main()