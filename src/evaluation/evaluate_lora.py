import json
import time
from collections import Counter

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_PATH = "experiments/smollm2-lora"
TEST_FILE = "data/splits/test.jsonl"


LABELS = [
    "PUBLIC",
    "INTERNAL",
    "CONFIDENTIAL",
    "RESTRICTED",
]

CLASSIFICATION_GUIDANCE = (
    "PUBLIC = approved information intended for external sharing.\n"
    "INTERNAL = routine company-only procedures or workflows.\n"
    "CONFIDENTIAL = private business, employee, customer, or contract information.\n"
    "RESTRICTED = credentials, keys, tokens, passwords, or access-bypass instructions."
)


def load_test_data():

    examples = []

    with open(
        TEST_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(line)
            )

    return examples


def extract_prediction(text):

    text = text.upper().strip()

    # Prefer exact label matches.
    for label in [
        "RESTRICTED",
        "CONFIDENTIAL",
        "INTERNAL",
        "PUBLIC",
    ]:

        if text.startswith(label):
            return label

    # Fallback for the current model's output format.
    for label in [
        "RESTRICTED",
        "CONFIDENTIAL",
        "INTERNAL",
        "PUBLIC",
    ]:

        if label in text:
            return label

    return "UNKNOWN"


def calculate_metrics(predictions):

    total = len(predictions)

    correct = sum(
        p["actual"] == p["prediction"]
        for p in predictions
    )

    accuracy = correct / total

    print("\n")
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy * 100:.2f}%")


    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    matrix = {
        actual: {
            predicted: 0
            for predicted in LABELS
        }
        for actual in LABELS
    }

    for item in predictions:

        actual = item["actual"]
        predicted = item["prediction"]

        if (
            actual in LABELS
            and predicted in LABELS
        ):
            matrix[actual][predicted] += 1

    print("\n")
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(
        f"{'Actual':<15}"
        + "".join(
            f"{label:<15}"
            for label in LABELS
        )
    )

    for actual in LABELS:

        print(
            f"{actual:<15}"
            + "".join(
                f"{matrix[actual][predicted]:<15}"
                for predicted in LABELS
            )
        )


    # ---------------------------------------------------------
    # Per-class metrics
    # ---------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)

    precisions = []
    recalls = []
    f1_scores = []

    for label in LABELS:

        true_positive = matrix[label][label]

        false_positive = sum(
            matrix[actual][label]
            for actual in LABELS
            if actual != label
        )

        false_negative = sum(
            matrix[label][predicted]
            for predicted in LABELS
            if predicted != label
        )

        if (
            true_positive + false_positive
            > 0
        ):

            precision = (
                true_positive
                /
                (
                    true_positive
                    + false_positive
                )
            )

        else:
            precision = 0.0


        if (
            true_positive + false_negative
            > 0
        ):

            recall = (
                true_positive
                /
                (
                    true_positive
                    + false_negative
                )
            )

        else:
            recall = 0.0


        if precision + recall > 0:

            f1 = (
                2
                * precision
                * recall
                /
                (precision + recall)
            )

        else:
            f1 = 0.0


        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)


        print(
            f"{label:<15}"
            f"Precision={precision * 100:6.2f}%  "
            f"Recall={recall * 100:6.2f}%  "
            f"F1={f1 * 100:6.2f}%"
        )


    macro_precision = (
        sum(precisions)
        / len(precisions)
    )

    macro_recall = (
        sum(recalls)
        / len(recalls)
    )

    macro_f1 = (
        sum(f1_scores)
        / len(f1_scores)
    )


    print("\n")
    print(
        f"Macro Precision: "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall: "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{macro_f1 * 100:.2f}%"
    )


    # ---------------------------------------------------------
    # Error analysis
    # ---------------------------------------------------------

    errors = [
        p
        for p in predictions
        if p["actual"] != p["prediction"]
    ]

    print("\n")
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print(
        f"Misclassified examples: "
        f"{len(errors)}"
    )

    if errors:

        for error in errors:

            print(
                f"Actual={error['actual']:<15}"
                f"Prediction={error['prediction']}"
            )

    else:

        print("No errors.")


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Loading model on {device}..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=(
            torch.float16
            if device == "cuda"
            else torch.float32
        ),
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.to(device)
    model.eval()

    examples = load_test_data()

    print(
        f"Loaded {len(examples)} test examples"
    )

    predictions = []
    latencies = []

    print("\nEvaluating...")
    print("--------------------------------")

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------

    warmup_prompt = (
        "Classify the sensitivity of the "
        "following company document.\n\n"
        "Document:\n"
        "This is a warm-up document.\n\n"
        "Classification:"
    )

    warmup_inputs = tokenizer(
        warmup_prompt,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():

        model.generate(
            **warmup_inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    if device == "cuda":
        torch.cuda.synchronize()


    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    for i, example in enumerate(
        examples,
        start=1,
    ):

        prompt = (
            "Classify the sensitivity of the "
            "following company document.\n\n"
            "Use these definitions:\n"
            f"{CLASSIFICATION_GUIDANCE}\n\n"
            f"Document:\n{example['text']}\n\n"
            "Classification:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
        ).to(device)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.no_grad():

            output = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        if device == "cuda":
            torch.cuda.synchronize()

        latency = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency)

        generated_tokens = output[
            0
        ][inputs["input_ids"].shape[1]:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        prediction = extract_prediction(
            response
        )

        actual = example["label"]

        predictions.append(
            {
                "actual": actual,
                "prediction": prediction,
                "text": example["text"],
                "latency": latency,
            }
        )

        print(
            f"{i:03d}. "
            f"Actual={actual:<12} "
            f"Prediction={prediction:<12} "
            f"Latency={latency:.2f} ms"
        )


    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    calculate_metrics(
        predictions
    )


    # ---------------------------------------------------------
    # Latency
    # ---------------------------------------------------------

    sorted_latencies = sorted(
        latencies
    )

    n = len(sorted_latencies)

    p95_index = min(
        int(0.95 * n),
        n - 1,
    )

    average_latency = (
        sum(latencies)
        / len(latencies)
    )

    median_latency = (
        sorted_latencies[n // 2]
    )

    p95_latency = (
        sorted_latencies[p95_index]
    )

    min_latency = min(latencies)
    max_latency = max(latencies)


    print("\n")
    print("=" * 60)
    print("LATENCY")
    print("=" * 60)

    print(
        f"Average: {average_latency:.2f} ms"
    )

    print(
        f"Median:  {median_latency:.2f} ms"
    )

    print(
        f"P95:     {p95_latency:.2f} ms"
    )

    print(
        f"Minimum: {min_latency:.2f} ms"
    )

    print(
        f"Maximum: {max_latency:.2f} ms"
    )


    # ---------------------------------------------------------
    # GPU memory
    # ---------------------------------------------------------

    if device == "cuda":

        memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print("\n")
        print("=" * 60)
        print("GPU MEMORY")
        print("=" * 60)

        print(
            f"Peak allocated: "
            f"{memory:.3f} GB"
        )


if __name__ == "__main__":
    main()