import json
import math
import time

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


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


def get_label_score(
    model,
    tokenizer,
    prompt,
    label,
    device,
):
    """
    Calculate normalized log-probability of the
    candidate label given the prompt.

    Logits are explicitly converted to float32
    before softmax for numerical stability.
    """

    prefix = prompt.rstrip()

    prompt_ids = tokenizer(
        prefix,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]

    full_ids = tokenizer(
        prefix + " " + label,
        return_tensors="pt",
        add_special_tokens=True,
    )["input_ids"]

    input_ids = full_ids.to(device)

    prompt_length = prompt_ids.shape[1]
    label_length = input_ids.shape[1] - prompt_length

    if label_length <= 0:
        raise ValueError(
            f"Label {label!r} produced no suffix tokens."
        )

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids
        )

        # IMPORTANT:
        # Convert FP16 logits to FP32 before
        # calculating log probabilities.
        logits = outputs.logits.float()

    # The logits at position t predict token t+1.
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = torch.log_softmax(
        shift_logits,
        dim=-1,
    )

    # First label token corresponds to the
    # prediction immediately after the prompt.
    label_start = prompt_length - 1

    label_log_probs = []

    for i in range(label_length):

        token_position = label_start + i

        token_id = shift_labels[
            0,
            token_position,
        ]

        token_log_prob = log_probs[
            0,
            token_position,
            token_id,
        ]

        label_log_probs.append(
            token_log_prob.item()
        )

    # Normalize by number of tokens so labels
    # with different token lengths are comparable.
    return sum(label_log_probs) / len(
        label_log_probs
    )


def evaluate_model(
    model,
    tokenizer,
    examples,
    device,
    model_name,
):

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    correct = 0

    results = []

    total_latency = 0.0

    for i, example in enumerate(
        examples,
        start=1,
    ):

        prompt = (
            "Classify the sensitivity of the following "
            "company document.\n\n"
            "Use these definitions:\n"
            f"{CLASSIFICATION_GUIDANCE}\n\n"
            f"Document:\n{example['text']}\n\n"
            "Classification:"
        )

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        scores = {}

        for label in LABELS:

            score = get_label_score(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                label=label,
                device=device,
            )

            scores[label] = score

        if device == "cuda":
            torch.cuda.synchronize()

        latency = (
            time.perf_counter() - start
        ) * 1000

        total_latency += latency

        prediction = max(
            scores,
            key=scores.get,
        )

        actual = example["label"]

        if prediction == actual:
            correct += 1

        results.append(
            {
                "actual": actual,
                "prediction": prediction,
                "scores": scores,
            }
        )

        print(
            f"{i:03d}. "
            f"Actual={actual:<12} "
            f"Prediction={prediction:<12} "
            f"Latency={latency:.2f} ms"
        )

        print(
            "     "
            + " | ".join(
                f"{label}={scores[label]:.8f}"
                for label in LABELS
            )
        )

    accuracy = (
        correct / len(examples)
    )

    average_latency = (
        total_latency / len(examples)
    )

    print()
    print("-" * 70)
    print(f"Correct: {correct}/{len(examples)}")
    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )
    print(
        f"Average latency: "
        f"{average_latency:.2f} ms"
    )
    print("-" * 70)

    return {
        "correct": correct,
        "total": len(examples),
        "accuracy": accuracy,
        "average_latency": average_latency,
        "results": results,
    }


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Loading models on {device}..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    print()
    print("Loading BASE model...")

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=dtype,
        )
    )

    base_model.to(device)
    base_model.eval()

    examples = load_test_data()

    print(
        f"Loaded {len(examples)} adversarial examples"
    )

    # --------------------------------------------------
    # BASE MODEL
    # --------------------------------------------------

    base_results = evaluate_model(
        model=base_model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        model_name="BASE MODEL — LABEL LIKELIHOOD",
    )

    # Free base model before loading LoRA.
    del base_model

    if device == "cuda":
        torch.cuda.empty_cache()

    # --------------------------------------------------
    # LoRA MODEL
    # --------------------------------------------------

    print()
    print("Loading LoRA model...")

    lora_base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=dtype,
        )
    )

    lora_model = PeftModel.from_pretrained(
        lora_base_model,
        ADAPTER_PATH,
    )

    lora_model.to(device)
    lora_model.eval()

    lora_results = evaluate_model(
        model=lora_model,
        tokenizer=tokenizer,
        examples=examples,
        device=device,
        model_name="LoRA MODEL — LABEL LIKELIHOOD",
    )

    # --------------------------------------------------
    # FINAL COMPARISON
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("BASE vs LoRA — LABEL LIKELIHOOD")
    print("=" * 70)

    print(
        f"Base accuracy: "
        f"{base_results['accuracy'] * 100:.2f}%"
    )

    print(
        f"LoRA accuracy: "
        f"{lora_results['accuracy'] * 100:.2f}%"
    )

    improvement = (
        lora_results["accuracy"]
        - base_results["accuracy"]
    )

    print(
        f"Absolute improvement: "
        f"{improvement * 100:.2f} percentage points"
    )

    print(
        f"Base average latency: "
        f"{base_results['average_latency']:.2f} ms"
    )

    print(
        f"LoRA average latency: "
        f"{lora_results['average_latency']:.2f} ms"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()