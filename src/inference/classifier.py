import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.policy import apply_policy_overrides


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
DEFAULT_ADAPTER = Path("experiments/smollm2-lora-policy")
LABELS = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")

CLASSIFICATION_GUIDANCE = (
    "PUBLIC = approved information intended for external sharing.\n"
    "INTERNAL = routine company-only procedures or workflows.\n"
    "CONFIDENTIAL = private business, employee, customer, or contract information.\n"
    "RESTRICTED = credentials, keys, tokens, passwords, or access-bypass instructions."
)


class DocumentClassifier:
    def __init__(self, adapter_path=DEFAULT_ADAPTER, device=None):
        self.adapter_path = str(adapter_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.label_token_ids = {
            label: self.tokenizer(
                " " + label,
                add_special_tokens=False,
            )["input_ids"][0]
            for label in LABELS
        }

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=dtype,
        )
        self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
        self.model.to(self.device)
        self.model.eval()

    def _prompt(self, text):
        return (
            "Classify the sensitivity of the following company document.\n\n"
            f"Use these definitions:\n{CLASSIFICATION_GUIDANCE}\n\n"
            f"Document:\n{text}\n\n"
            "Classification:"
        )

    def _extract_label(self, response):
        response = response.upper().strip()
        for label in ("RESTRICTED", "CONFIDENTIAL", "INTERNAL", "PUBLIC"):
            if response.startswith(label) or label in response:
                return label
        return "UNKNOWN"

    def _relative_confidence(self, first_token_logits):
        scores = torch.stack(
            [
                first_token_logits[token_id]
                for token_id in self.label_token_ids.values()
            ]
        )
        probabilities = torch.softmax(scores.float(), dim=0)
        return round(float(probabilities.max().item()), 4)

    def classify(self, text, use_policy=True):
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        policy_prediction = (
            apply_policy_overrides(text, None)
            if use_policy
            else None
        )
        if policy_prediction is not None:
            return {
                "label": policy_prediction,
                "confidence": 1.0,
                "policy_override": True,
                "model_prediction": None,
                "model_version": Path(self.adapter_path).name,
                "device": self.device,
                "latency_ms": 0.0,
            }

        prompt = self._prompt(text)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        if self.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()

        response = self.tokenizer.decode(
            output.sequences[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        model_prediction = self._extract_label(response)
        prediction = (
            apply_policy_overrides(text, model_prediction)
            if use_policy
            else model_prediction
        )
        confidence = self._relative_confidence(output.scores[0][0])

        return {
            "label": prediction,
            "confidence": confidence,
            "policy_override": prediction != model_prediction,
            "model_prediction": model_prediction,
            "model_version": Path(self.adapter_path).name,
            "device": self.device,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
