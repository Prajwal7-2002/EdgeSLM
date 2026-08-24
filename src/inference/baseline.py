import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

model = model.to(device)
model.eval()

prompt = "Explain why protecting sensitive company documents is important."

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

print("Input shape:", inputs["input_ids"].shape)

# -------------------------
# Warm-up
# -------------------------

print("\nRunning warm-up...")

with torch.no_grad():
    for _ in range(3):
        model.generate(
            **inputs,
            max_new_tokens=50,
        )

torch.cuda.synchronize()

# -------------------------
# Benchmark
# -------------------------

latencies = []

for i in range(10):

    torch.cuda.synchronize()

    start = time.perf_counter()

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
        )

    torch.cuda.synchronize()

    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

    print(f"Run {i + 1}: {latency_ms:.2f} ms")


# -------------------------
# Results
# -------------------------

avg_latency = sum(latencies) / len(latencies)

print("\n-------------------------")
print("Benchmark Results")
print("-------------------------")

print(f"Average latency: {avg_latency:.2f} ms")
print(f"Minimum latency: {min(latencies):.2f} ms")
print(f"Maximum latency: {max(latencies):.2f} ms")

print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB",
)

response = tokenizer.decode(
    output[0],
    skip_special_tokens=True,
)

print("\nExample response:")
print(response)