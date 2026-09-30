import csv
import re
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def extract_gsm8k_answer(answer: str):
    match = re.search(r"Final Answer:\s*(.*)", answer, re.IGNORECASE)

    text = match.group(1).strip() if match else answer.strip()

    number_match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)

    if number_match:
        return number_match.group(0).replace(",", "")

    return text.strip()


def compute_correct_gsm8k(prediction: str, ground_truth: str):
    try:
        pred_number = Decimal(prediction)
        truth_number = Decimal(ground_truth.replace(",", "").strip())

        return int(pred_number == truth_number)

    except InvalidOperation:
        return int(prediction.strip().lower() == ground_truth.strip().lower())


def extract_mmlu_answer(answer: str):
    match = re.search(r"Final Answer:\s*([ABCD])\b", answer, re.IGNORECASE)

    if match:
        return match.group(1).upper()

    # Allow responses containing only a single option letter.
    stripped = answer.strip().upper()

    if stripped in {"A", "B", "C", "D"}:
        return stripped

    return ""


def compute_correct_mmlu(prediction: str, ground_truth: str):
    return int(prediction.strip().upper() == ground_truth.strip().upper())


def evaluate_outputs(input_file, output_file):
    model_results = defaultdict(list)

    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", newline="", encoding="utf-8") as outfile,
    ):
        reader = csv.DictReader(infile)

        fieldnames = reader.fieldnames + ["prediction", "correct"]

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            dataset = row["dataset"]
            ground_truth = row["ground_truth"]

            if dataset == "gsm8k":
                prediction = extract_gsm8k_answer(row["answer"])
                correct = compute_correct_gsm8k(prediction, ground_truth)

            elif dataset == "mmlu":
                prediction = extract_mmlu_answer(row["answer"])
                correct = compute_correct_mmlu(prediction, ground_truth)

            else:
                raise ValueError(f"Unsupported dataset: {dataset}")

            row["prediction"] = prediction
            row["correct"] = correct

            writer.writerow(row)

            model_results[row["model"]].append(correct)

    print(f"Saved evaluated data to {output_file}")

    for model_name, results in model_results.items():
        accuracy = sum(results) / len(results)
        print(f"{model_name}: accuracy={accuracy:.3f}")


def main():
    evaluate_outputs(
        RESULTS_DIR / "gsm8k_outputs.csv",
        RESULTS_DIR / "gsm8k_evaluated.csv",
    )

    evaluate_outputs(
        RESULTS_DIR / "mmlu_outputs.csv",
        RESULTS_DIR / "mmlu_evaluated.csv",
    )


if __name__ == "__main__":
    main()