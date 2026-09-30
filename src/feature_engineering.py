import csv
import json
import math
import re
import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "query_splits.csv"
OUTPUT_FILE = RESULTS_DIR / "query_features.csv"
METADATA_FILE = RESULTS_DIR / "feature_metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EXPECTED_QUERY_COUNT = 3194


def count_words(text):
    return len(re.findall(r"\b[\w'-]+\b", text))


def count_numbers(text):
    return len(re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text))


def count_math_symbols(text):
    symbols = set("+-*/=<>^%$")
    return sum(char in symbols for char in text)


def count_options(text):
    return len(re.findall(r"(?m)^[ABCD]\.\s", text))


def count_sentences(text):
    matches = re.findall(r"[.!?]+", text)
    return max(1, len(matches))


def average_word_length(text):
    words = re.findall(r"\b[\w'-]+\b", text)

    if not words:
        return 0.0

    return sum(len(word) for word in words) / len(words)


def extract_structural_features(question, tokenizer):
    """
    Extract features available before target-model inference.
    """
    option_count = count_options(question)

    token_count = len(
        tokenizer.encode(question, add_special_tokens=True, truncation=False)
    )

    return {
        "char_count": len(question),
        "word_count": count_words(question),
        "token_count": token_count,
        "line_count": max(1, len(question.splitlines())),
        "sentence_count": count_sentences(question),
        "number_count": count_numbers(question),
        "math_symbol_count": count_math_symbols(question),
        "question_mark_count": question.count("?"),
        "option_count": option_count,
        "is_multiple_choice": int(option_count >= 4),
        "avg_word_length": average_word_length(question),
    }


def load_queries():
    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            rows.append(
                {
                    "query_id": row["query_id"],
                    "dataset": row["dataset"],
                    "category": row["category"],
                    "question": row["question"],
                    "split": row["split"],
                }
            )

    return rows


def validate_queries(rows):
    query_ids = [row["query_id"] for row in rows]

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Duplicate query IDs detected.")

    if len(rows) != EXPECTED_QUERY_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_QUERY_COUNT} queries but found {len(rows)}."
        )

    valid_splits = {"train", "validation", "test"}

    for row in rows:
        if row["split"] not in valid_splits:
            raise ValueError(f"Invalid split: {row['split']}")


def sanitize_name(text):
    value = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return value.strip("_")


def add_one_hot_features(output_row, dataset, category, datasets, categories):
    for dataset_name in datasets:
        column = "dataset_" + sanitize_name(dataset_name)
        output_row[column] = int(dataset == dataset_name)

    for category_name in categories:
        column = "category_" + sanitize_name(category_name)
        output_row[column] = int(category == category_name)


def generate_embeddings(questions):
    """
    Generate normalized MiniLM embeddings for all queries.
    """
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")

    embeddings = model.encode(
        questions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return model, embeddings


def build_features(rows, model, embeddings):
    datasets = sorted({row["dataset"] for row in rows})
    categories = sorted({row["category"] for row in rows})

    tokenizer = model.tokenizer

    feature_rows = []

    for index, row in enumerate(rows):
        question = row["question"]

        output_row = {
            "query_id": row["query_id"],
            "dataset": row["dataset"],
            "category": row["category"],
            "split": row["split"],
            "question": question,
        }

        output_row.update(extract_structural_features(question, tokenizer))

        add_one_hot_features(
            output_row=output_row,
            dataset=row["dataset"],
            category=row["category"],
            datasets=datasets,
            categories=categories,
        )

        for dimension, value in enumerate(embeddings[index]):
            output_row[f"embedding_{dimension:03d}"] = float(value)

        feature_rows.append(output_row)

    return feature_rows, datasets, categories


def check_numeric_features(feature_rows):
    non_feature_columns = {"query_id", "dataset", "category", "split", "question"}

    feature_columns = [
        column for column in feature_rows[0] if column not in non_feature_columns
    ]

    for row in feature_rows:
        for column in feature_columns:
            value = float(row[column])

            if not math.isfinite(value):
                raise ValueError(
                    f"Non-finite feature detected: {row['query_id']} {column}={value}"
                )


def save_features(feature_rows):
    fieldnames = list(feature_rows[0].keys())

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in feature_rows:
            output_row = row.copy()

            output_row["avg_word_length"] = round(
                float(output_row["avg_word_length"]), 6
            )

            for column in fieldnames:
                if column.startswith("embedding_"):
                    output_row[column] = round(float(output_row[column]), 8)

            writer.writerow(output_row)


def save_metadata(feature_rows, datasets, categories, embedding_dimension):
    non_feature_columns = ["query_id", "dataset", "category", "split", "question"]

    feature_columns = [
        column for column in feature_rows[0] if column not in non_feature_columns
    ]

    structural_columns = [
        "char_count",
        "word_count",
        "token_count",
        "line_count",
        "sentence_count",
        "number_count",
        "math_symbol_count",
        "question_mark_count",
        "option_count",
        "is_multiple_choice",
        "avg_word_length",
    ]

    embedding_columns = [
        column for column in feature_columns if column.startswith("embedding_")
    ]

    dataset_columns = [
        column for column in feature_columns if column.startswith("dataset_")
    ]

    category_columns = [
        column for column in feature_columns if column.startswith("category_")
    ]

    metadata = {
        "input_file": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(PROJECT_ROOT)),
        "num_queries": len(feature_rows),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimension": embedding_dimension,
        "embedding_normalized": True,
        "datasets": datasets,
        "categories": categories,
        "non_feature_columns": non_feature_columns,
        "structural_feature_columns": structural_columns,
        "dataset_feature_columns": dataset_columns,
        "category_feature_columns": category_columns,
        "embedding_feature_columns": embedding_columns,
        "total_feature_count": len(feature_columns),
        "leakage_policy": (
            "Only information available before target LLM inference is included. "
            "Ground truth, model answer, correctness, cost, latency, completion "
            "tokens and utility are excluded."
        ),
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_queries()
    validate_queries(rows)

    questions = [row["question"] for row in rows]

    model, embeddings = generate_embeddings(questions)

    if len(embeddings) != len(rows):
        raise ValueError("Embedding/query count mismatch.")

    embedding_dimension = embeddings.shape[1]

    feature_rows, datasets, categories = build_features(rows, model, embeddings)

    check_numeric_features(feature_rows)

    save_features(feature_rows)
    save_metadata(feature_rows, datasets, categories, embedding_dimension)

    print(
        f"Saved {len(feature_rows)} queries "
        f"with {len(feature_rows[0]) - 5} numeric features to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()