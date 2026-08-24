import json
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

LABELS = [
    "PUBLIC",
    "INTERNAL",
    "CONFIDENTIAL",
    "RESTRICTED",
]

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Load model
# -------------------------

print(f"Loading model on {device}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

model = model.to(device)
model.eval()


# -------------------------
# Load dataset
# -------------------------

rows = []

with open(
    "data/raw/documents.jsonl",
    "r",
    encoding="utf-8",
) as f:

    for line in f:
        if line.strip():
            rows.append(json.loads(line))


print(f"Loaded {len(rows)} examples")


# -------------------------
# Classification function
# -------------------------

def classify(text):

    prompt = f"""Classify the following company document into exactly one
of these categories:

PUBLIC
INTERNAL
CONFIDENTIAL
RESTRICTED

Document:
{text}

Classification:"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
        )

    generated_tokens = output[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    # Extract the first valid classification
    response_upper = response.upper()

    for label in LABELS:
        if label in response_upper:
            return label

    return "UNKNOWN"


# -------------------------
# Run evaluation
# -------------------------

predictions = []

for i, row in enumerate(rows):

    start = time.perf_counter()

    prediction = classify(row["text"])

    latency = (
        time.perf_counter() - start
    ) * 1000

    predictions.append(
        {
            "text": row["text"],
            "actual": row["label"],
            "prediction": prediction,
            "latency_ms": latency,
        }
    )

    print(
        f"{i + 1:02d}. "
        f"Actual={row['label']:<12} "
        f"Prediction={prediction:<20} "
        f"Latency={latency:.2f} ms"
    )


# -------------------------
# Basic accuracy
# -------------------------

correct = 0

for result in predictions:

    if result["prediction"].upper() == result["actual"]:
        correct += 1


accuracy = correct / len(predictions)

print("\n-------------------------")
print("Zero-Shot Results")
print("-------------------------")

print(f"Correct: {correct}/{len(predictions)}")
print(f"Accuracy: {accuracy:.2%}")