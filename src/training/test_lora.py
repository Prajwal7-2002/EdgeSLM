import torch

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


def count_parameters(model):

    total = 0
    trainable = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:
            trainable += parameter.numel()

    return total, trainable


def main():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Loading model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=(
            torch.float16
            if device == "cuda"
            else torch.float32
        ),
    )

    total_before, trainable_before = count_parameters(model)

    print("\nBefore LoRA")
    print("-------------------------")
    print(
        f"Total parameters: "
        f"{total_before:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_before:,}"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    total_after, trainable_after = count_parameters(
        model
    )

    percentage = (
        trainable_after
        / total_after
        * 100
    )

    print("\nAfter LoRA")
    print("-------------------------")

    print(
        f"Total parameters: "
        f"{total_after:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_after:,}"
    )

    print(
        f"Trainable percentage: "
        f"{percentage:.4f}%"
    )

    print("\nLoRA configuration")
    print("-------------------------")
    print("Rank (r): 8")
    print("Alpha: 16")
    print("Dropout: 0.05")
    print("Targets: q_proj, v_proj")

    print("\nTrainable parameter summary:")
    model.print_trainable_parameters()


if __name__ == "__main__":
    main()