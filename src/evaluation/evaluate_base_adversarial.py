import argparse
import json
import time

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

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

    with open(
        path,
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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "test_file",
        nargs="?",
        default=TEST_FILE,
    )
    args = parser.parse_args()

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Loading BASE model on {device}..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=(
            torch.float16
            if device == "cuda"
            else torch.float32
        ),
    )

    model.to(device)

    model.eval()

    examples = load_test_data(args.test_file)

    print(
        f"Loaded {len(examples)} adversarial test examples"
    )

    correct = 0
    total_latency = 0.0

    predictions = []

    print("\nEvaluating BASE MODEL...")
    print("-" * 60)

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

        total_latency += latency

        generated_tokens = output[
            0
        ][inputs["input_ids"].shape[1]:]

        response = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        
        print(f"    Raw response: {response!r}")

        prediction = extract_prediction(
            response
        )

        actual = example["label"]

        if prediction == actual:
            correct += 1

        predictions.append(
            {
                "actual": actual,
                "prediction": prediction,
            }
        )

        print(
            f"{i:03d}. "
            f"Actual={actual:<12} "
            f"Prediction={prediction:<12} "
            f"Latency={latency:.2f} ms"
        )

    accuracy = (
        correct / len(examples)
    )

    average_latency = (
        total_latency / len(examples)
    )

    print("\n")
    print("=" * 60)
    print("BASE MODEL — ADVERSARIAL RESULTS")
    print("=" * 60)

    print(
        f"Correct: {correct}/{len(examples)}"
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print(
        f"Average latency: "
        f"{average_latency:.2f} ms"
    )


if __name__ == "__main__":
    main()