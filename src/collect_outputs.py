import csv
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from config.models import MODELS
from config.pricing import PRICING
from config.prompts import COLLECTION_PROMPT
from src.load_dataset import MMLU_SUBJECTS, load_gsm8k_samples, load_mmlu_samples

csv.field_size_limit(sys.maxsize)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


def calculate_cost(model_name, prompt_tokens, completion_tokens):
    pricing = PRICING[model_name]

    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]

    return round(input_cost + output_cost, 8)


def query_model(question: str, model_name: str, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            start = time.perf_counter()

            response = client.chat.completions.create(
                model=MODELS[model_name],
                messages=[
                    {
                        "role": "user",
                        "content": f"{question}\n\n{COLLECTION_PROMPT}",
                    }
                ],
                temperature=0,
                timeout=60,
            )

            latency = time.perf_counter() - start

            return response, latency

        except Exception as error:
            print(f"{model_name} attempt {attempt}/{max_retries} failed: {error}")

            if attempt < max_retries:
                time.sleep(5)

    return None, None


def collect_outputs(samples, output_file):

    completed = set()

    file_exists = output_file.exists() and output_file.stat().st_size > 0

    if file_exists:
        with open(output_file, "r", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)

            for row in reader:
                completed.add((row["dataset"], row["question"], row["model"]))

    mode = "a" if file_exists else "w"

    with open(output_file, mode, newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(
                [
                    "dataset",
                    "category",
                    "question",
                    "ground_truth",
                    "model",
                    "answer",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "latency",
                    "cost",
                ]
            )

        for sample_index, sample in enumerate(samples, start=1):
            dataset = sample["dataset"]
            category = sample["category"]
            question = sample["question"]
            ground_truth = sample["ground_truth"]

            for model_name in MODELS:
                key = (dataset, question, model_name)

                if key in completed:
                    continue

                print(f"[{sample_index}/{len(samples)}] Running {model_name}...")

                response, latency = query_model(question, model_name)

                if response is None:
                    print(
                        f"[{sample_index}/{len(samples)}] "
                        f"{model_name}: failed after retries"
                    )
                    continue

                answer = response.choices[0].message.content
                usage = response.usage

                cost = calculate_cost(
                    model_name, usage.prompt_tokens, usage.completion_tokens
                )

                writer.writerow(
                    [
                        dataset,
                        category,
                        question,
                        ground_truth,
                        model_name,
                        answer,
                        usage.prompt_tokens,
                        usage.completion_tokens,
                        usage.total_tokens,
                        latency,
                        cost,
                    ]
                )

                file.flush()

                completed.add(key)

    print(f"Saved to {output_file}")


def collect_gsm8k():
    samples = load_gsm8k_samples()
    collect_outputs(samples, RESULTS_DIR / "gsm8k_outputs.csv")


def collect_mmlu():
    samples = load_mmlu_samples(MMLU_SUBJECTS)
    collect_outputs(samples, RESULTS_DIR / "mmlu_outputs.csv")


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    collect_gsm8k()
    collect_mmlu()