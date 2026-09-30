from datasets import load_dataset


MMLU_SUBJECTS = [
    "machine_learning",
    "computer_security",
    "college_mathematics",
    "abstract_algebra",
    "high_school_physics",
    "high_school_macroeconomics",
    "philosophy",
    "professional_psychology",
]


def load_gsm8k_samples(
    n=None,
):
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
    )

    split = dataset["test"]

    if n is None:
        n = len(split)

    if n > len(split):
        raise ValueError(
            f"Requested {n} samples, "
            f"but GSM8K test set only contains "
            f"{len(split)} samples."
        )

    samples = []

    for i in range(n):
        item = split[i]

        ground_truth = (
            item["answer"]
            .split("####")[-1]
            .strip()
        )

        samples.append({
            "dataset": "gsm8k",
            "category": "math_reasoning",
            "question": item["question"],
            "ground_truth": ground_truth,
        })

    return samples


def load_mmlu_samples(
    subjects,
    n_per_subject=None,
):
    samples = []

    answer_letters = [
        "A",
        "B",
        "C",
        "D",
    ]

    for subject in subjects:
        dataset = load_dataset(
            "cais/mmlu",
            subject,
        )

        split = dataset["test"]

        if n_per_subject is None:
            n = len(split)
        else:
            n = min(
                n_per_subject,
                len(split),
            )

        for i in range(n):
            item = split[i]

            question = (
                f"{item['question']}\n\n"
                f"A. {item['choices'][0]}\n"
                f"B. {item['choices'][1]}\n"
                f"C. {item['choices'][2]}\n"
                f"D. {item['choices'][3]}"
            )

            ground_truth = (
                answer_letters[
                    item["answer"]
                ]
            )

            samples.append({
                "dataset": "mmlu",
                "category": subject,
                "question": question,
                "ground_truth": ground_truth,
            })

    return samples