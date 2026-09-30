import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.load_dataset import load_gsm8k_samples, load_mmlu_samples, MMLU_SUBJECTS

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

EXPECTED_MODELS = {"gpt", "claude", "gemini", "qwen"}

DATASETS = {
    "gsm8k": {
        "file": RESULTS_DIR / "gsm8k_outputs.csv",
        "loader": lambda: load_gsm8k_samples(),
    },
    "mmlu": {
        "file": RESULTS_DIR / "mmlu_outputs.csv",
        "loader": lambda: load_mmlu_samples(MMLU_SUBJECTS),
    },
}


def check_dataset(dataset_name, samples, input_file):
    expected_keys = [(sample["category"], sample["question"]) for sample in samples]

    expected_counts = Counter(expected_keys)

    duplicate_expected = {
        key: count for key, count in expected_counts.items() if count > 1
    }

    question_models = defaultdict(list)

    total_rows = 0

    with open(input_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            key = (row["category"], row["question"])

            question_models[key].append(row["model"])

    missing_pairs = []

    for key in expected_counts:
        models_found = set(question_models.get(key, []))

        missing = EXPECTED_MODELS - models_found

        if missing:
            missing_pairs.append((key, missing))

    completely_missing = [key for key in expected_counts if key not in question_models]

    print(f"\n{dataset_name.upper()}")
    print(f"Expected samples: {len(samples)}")
    print(f"CSV rows: {total_rows}")
    print(f"Questions with missing models: {len(missing_pairs)}")
    print(f"Questions completely missing: {len(completely_missing)}")

    if duplicate_expected:
        print("\nDuplicate source questions:")

        for (category, question), count in duplicate_expected.items():
            print(f"[{category}] {count} occurrences: {question}")

    if missing_pairs:
        print("\nMissing model outputs:")

        for (category, question), missing in missing_pairs:
            print(f"[{category}] missing {', '.join(sorted(missing))}: {question}")

    if completely_missing:
        print("\nCompletely missing questions:")

        for category, question in completely_missing:
            print(f"[{category}] {question}")

    return {
        "expected_samples": len(samples),
        "csv_rows": total_rows,
        "missing_model_queries": len(missing_pairs),
        "completely_missing_queries": len(completely_missing),
    }


def main():
    summaries = {}

    for dataset_name, config in DATASETS.items():
        samples = config["loader"]()

        summaries[dataset_name] = check_dataset(
            dataset_name=dataset_name,
            samples=samples,
            input_file=config["file"],
        )

    print("\nOverall completeness")

    for dataset_name, summary in summaries.items():
        print(
            f"{dataset_name.upper()}: "
            f"{summary['csv_rows']} rows, "
            f"{summary['missing_model_queries']} queries with missing models"
        )


if __name__ == "__main__":
    main()