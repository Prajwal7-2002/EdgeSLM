import torch
from transformers import AutoModelForCausalLM


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    print("\nModel loaded.")

    print("\nLinear layers:")

    found = set()

    for name, module in model.named_modules():

        if isinstance(module, torch.nn.Linear):

            # Get the final component of the module name.
            component = name.split(".")[-1]

            found.add(component)

    for component in sorted(found):

        print(" ", component)


if __name__ == "__main__":
    main()
    