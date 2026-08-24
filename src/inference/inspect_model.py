import torch
from transformers import AutoModelForCausalLM


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)

model = model.to(device)
model.eval()


# -------------------------
# Parameter information
# -------------------------

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("Model:", MODEL_NAME)

print("Total parameters:", total_params)
print("Total parameters (M):", total_params / 1e6)

print("Trainable parameters:", trainable_params)


# -------------------------
# Architecture
# -------------------------

config = model.config

print("\nModel configuration")

print("Hidden size:", config.hidden_size)
print("Number of layers:", config.num_hidden_layers)
print("Attention heads:", config.num_attention_heads)

if hasattr(config, "num_key_value_heads"):
    print("KV heads:", config.num_key_value_heads)

print("Vocabulary size:", config.vocab_size)

if hasattr(config, "max_position_embeddings"):
    print("Max position embeddings:", config.max_position_embeddings)


# -------------------------
# Memory
# -------------------------

print(
    "\nGPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 3),
    "GB",
)