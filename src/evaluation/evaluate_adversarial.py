import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.policy import apply_policy_overrides
from evaluation.metrics import wilson_interval


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

ADAPTER_PATH = "experiments/smollm2-lora"

TEST_FILE = "data/odd/adversarial_test.jsonl"


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


def load_test_data(path=TEST_FILE):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    return examples


def extract_prediction(text):

    text = text.upper()

    for label in [
        "RESTRICTED",
        "CONFIDENTIAL",
        "INTERNAL",
        "PUBLIC",
    ]:

        if label in text:
            return label

    return "UNKNOWN"


def print_confusion_matrix(actuals, predictions):

    matrix = {
        actual: {
            predicted: 0
            for predicted in LABELS
        }
        for actual in LABELS
    }

    for actual, prediction in zip(actuals, predictions):

        if prediction in LABELS:
            matrix[actual][prediction] += 1

    print("\n")
    print("=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)

    print(
        f"{'Actual':<15}"
        f"{'PUBLIC':<15}"
        f"{'INTERNAL':<15}"
        f"{'CONFIDENTIAL':<15}"
        f"{'RESTRICTED':<15}"
    )

    for actual in LABELS:

        print(
            f"{actual:<15}"
            f"{matrix[actual]['PUBLIC']:<15}"
            f"{matrix[actual]['INTERNAL']:<15}"
            f"{matrix[actual]['CONFIDENTIAL']:<15}"
            f"{matrix[actual]['RESTRICTED']:<15}"
        )


def calculate_metrics(actuals, predictions):

    print("\n")
    print("=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)

    precisions = []
    recalls = []
    f1s = []

    for label in LABELS:

        tp = sum(
            1
            for actual, prediction
            in zip(actuals, predictions)
            if actual == label and prediction == label
        )

        fp = sum(
            1
            for actual, prediction
            in zip(actuals, predictions)
            if actual != label and prediction == label
        )

        fn = sum(
            1
            for actual, prediction
            in zip(actuals, predictions)
            if actual == label and prediction != label
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        print(
            f"{label:<15}"
            f"Precision={precision * 100:6.2f}%  "
            f"Recall={recall * 100:6.2f}%  "
            f"F1={f1 * 100:6.2f}%"
        )

    print()

    print(
        f"Macro Precision: "
        f"{sum(precisions) / len(precisions) * 100:.2f}%"
    )

    print(
        f"Macro Recall: "
        f"{sum(recalls) / len(recalls) * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{sum(f1s) / len(f1s) * 100:.2f}%"
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_file",
        nargs="?",
        default=TEST_FILE,
    )
    parser.add_argument(
        "--adapter",
        default=ADAPTER_PATH,
    )
    parser.add_argument(
        "--include-guidance",
        action="store_true",
        help="Use the policy-aware prompt used by the policy adapter.",
    )
    parser.add_argument(
        "--no-policy",
        action="store_true",
        help="Report the raw model prediction without policy overrides.",
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Loading model on {device}...")

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
        args.adapter,
    )

    model.to(device)
    model.eval()

    examples = load_test_data(args.test_file)

    print(
        f"Loaded {len(examples)} adversarial test examples"
    )

    correct = 0

    actuals = []
    predictions = []

    latencies = []

    errors = []

    print("\nEvaluating...")
    print("-" * 60)

    for i, example in enumerate(
        examples,
        start=1,
    ):

        guidance = (
            "Use these definitions:\n"
            f"{CLASSIFICATION_GUIDANCE}\n\n"
            if args.include_guidance
            else ""
        )

        prompt = (
            "Classify the sensitivity of the "
            "following company document.\n\n"
            f"{guidance}"
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

        model_prediction = extract_prediction(response)
        prediction = (
            model_prediction
            if args.no_policy
            else apply_policy_overrides(
                example["text"],
                model_prediction,
            )
        )

        actual = example["label"]

        actuals.append(actual)
        predictions.append(prediction)

        if prediction == actual:

            correct += 1

        else:

            errors.append(
                {
                    "index": i,
                    "text": example["text"],
                    "actual": actual,
                    "prediction": prediction,
                    "model_response": response,
                }
            )

        print(
            f"{i:03d}. "
            f"Actual={actual:<12} "
            f"Prediction={prediction:<12} "
            f"Latency={latency:.2f} ms"
        )

    total = len(examples)

    accuracy = correct / total

    accuracy_lower, accuracy_upper = wilson_interval(
        correct,
        total,
    )

    sorted_latencies = sorted(latencies)

    median = sorted_latencies[
        len(sorted_latencies) // 2
    ]

    p95_index = int(
        0.95 * len(sorted_latencies)
    ) - 1

    p95 = sorted_latencies[
        max(0, p95_index)
    ]

    average = sum(latencies) / len(latencies)

    print("\n")
    print("=" * 60)
    print("ADVERSARIAL / OOD RESULTS")
    print("=" * 60)

    print(
        f"Correct: {correct}/{total}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"95% Wilson interval: "
        f"[{accuracy_lower * 100:.2f}%, "
        f"{accuracy_upper * 100:.2f}%]"
    )

    print_confusion_matrix(
        actuals,
        predictions,
    )

    calculate_metrics(
        actuals,
        predictions,
    )

    print("\n")
    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    print(
        f"Misclassified examples: "
        f"{len(errors)}"
    )

    for error in errors:

        print("\n")
        print(
            f"Example #{error['index']}"
        )

        print(
            f"Actual:     {error['actual']}"
        )

        print(
            f"Prediction: {error['prediction']}"
        )

        print(
            f"Model response: "
            f"{error['model_response']!r}"
        )

        print(
            f"Text: "
            f"{error['text']}"
        )

    print("\n")
    print("=" * 60)
    print("LATENCY")
    print("=" * 60)

    print(
        f"Average: {average:.2f} ms"
    )

    print(
        f"Median:  {median:.2f} ms"
    )

    print(
        f"P95:     {p95:.2f} ms"
    )

    print(
        f"Minimum: {min(latencies):.2f} ms"
    )

    print(
        f"Maximum: {max(latencies):.2f} ms"
    )


if __name__ == "__main__":
    main()