import torch

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

TRAIN_FILE = "data/splits/train.jsonl"
BOUNDARY_TRAIN_FILE = "data/raw/boundary_training.jsonl"
VAL_FILE = "data/splits/validation.jsonl"

OUTPUT_DIR = "experiments/smollm2-lora-policy"

CLASSIFICATION_GUIDANCE = (
    "PUBLIC = approved information intended for external sharing.\n"
    "INTERNAL = routine company-only procedures or workflows.\n"
    "CONFIDENTIAL = private business, employee, customer, or contract information.\n"
    "RESTRICTED = credentials, keys, tokens, passwords, or access-bypass instructions."
)


def format_example(example):

    return (
        "Classify the sensitivity of the following company document.\n\n"
        "Use these definitions:\n"
        f"{CLASSIFICATION_GUIDANCE}\n\n"
        f"Document:\n{example['text']}\n\n"
        f"Classification: {example['label']}"
    )


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Training device: {device}")

    if device == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print("\nLoading datasets...")

    dataset = load_dataset(
        "json",
        data_files={
            "train": [TRAIN_FILE, BOUNDARY_TRAIN_FILE],
            "validation": VAL_FILE,
        },
    )

    print(
        f"Train examples: {len(dataset['train'])}"
    )

    print(
        f"Validation examples: {len(dataset['validation'])}"
    )

    # --------------------------------------------------
    # Tokenizer
    # --------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Training configuration
    # --------------------------------------------------

    training_args = SFTConfig(

        output_dir=OUTPUT_DIR,

        # Training
        num_train_epochs=3,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,

        gradient_accumulation_steps=4,

        learning_rate=2e-4,

        # Sequence length
        max_length=256,

        # Evaluation
        eval_strategy="steps",
        eval_steps=50,

        # Saving
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,

        # Logging
        logging_steps=10,

        # GPU
        fp16=(device == "cuda"),

        gradient_checkpointing=True,

        # Disable external logging
        report_to="none",

        # Dataset handling
        dataset_text_field="text",

        # Don't remove columns before formatting
        remove_unused_columns=False,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = SFTTrainer(

        model=MODEL_NAME,

        args=training_args,

        train_dataset=dataset["train"],

        eval_dataset=dataset["validation"],

        processing_class=tokenizer,

        peft_config=lora_config,

        formatting_func=format_example,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print("\nStarting LoRA training...")
    print("--------------------------------")

    trainer.train()

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    print("\nSaving LoRA adapter...")

    trainer.save_model(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print("\nTraining complete.")

    print(
        "Adapter saved to:",
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()