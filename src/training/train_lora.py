"""LoRA fine-tuning for the document classifier.

    python src/training/train_lora.py                       # v2: label-only loss, v1+v2 data
    python src/training/train_lora.py --data v1 --output experiments/smollm2-v1data-labelloss
    python src/training/train_lora.py --objective full-sequence --data v1   # the original v1 recipe

Two things changed between v1 and v2, and both are flags so their effects can be
measured separately:

--objective  full-sequence trains on every token of prompt + document + label,
             which is what the v1 adapter did. Only the one or two label tokens
             carry the classification signal; the rest teaches the model to
             reproduce template wording. label-only puts the loss on the label
             alone.
--data       v1 is the template generator plus the boundary set. v2 adds
             data/v2, which teaches the concepts behind each label (see
             data/generate_dataset_v2.py).
"""

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.prompt import build_prompt, completion


DATA = {
    "v1": {
        "train": ["data/splits/train.jsonl", "data/raw/boundary_training.jsonl"],
        "validation": ["data/splits/validation.jsonl"],
    },
    "v2": {
        "train": [
            "data/splits/train.jsonl",
            "data/raw/boundary_training.jsonl",
            "data/v2/train.jsonl",
        ],
        "validation": ["data/splits/validation.jsonl", "data/v2/validation.jsonl"],
    },
}

ALL_LINEAR = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def to_prompt_completion(example):
    return {"prompt": build_prompt(example["text"]), "completion": completion(example["label"])}


def to_full_text(example):
    return {"text": build_prompt(example["text"]) + completion(example["label"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--data", choices=sorted(DATA), default="v2")
    parser.add_argument("--objective", choices=["label-only", "full-sequence"], default="label-only")
    parser.add_argument("--output", type=Path, default=Path("experiments/smollm2-v2"))
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--attention-only", action="store_true",
                        help="adapt only q_proj and v_proj, as v1 did")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model} data={args.data} objective={args.objective}")

    files = DATA[args.data]
    dataset = load_dataset("json", data_files=files)
    convert = to_prompt_completion if args.objective == "label-only" else to_full_text
    dataset = dataset.map(convert, remove_columns=["text", "label"])
    print(f"train={len(dataset['train'])} validation={len(dataset['validation'])}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["q_proj", "v_proj"] if args.attention_only else ALL_LINEAR,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_length=384,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        fp16=(device == "cuda"),
        gradient_checkpointing=True,
        report_to="none",
        seed=args.seed,
        # device_map="auto" (TRL's default) pre-allocates the whole model on the
        # GPU in one block, which Windows refuses when the system commit charge
        # is high even with the GPU empty. Loading on CPU and moving tensor by
        # tensor avoids that and gives the same weights.
        model_init_kwargs={"device_map": None},
        **({"dataset_text_field": "text"} if args.objective == "full-sequence" else {}),
    )

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print(f"Adapter saved to {args.output}")


if __name__ == "__main__":
    main()
