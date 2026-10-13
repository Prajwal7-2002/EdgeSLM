import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.chunking import document_token_budget, most_sensitive, split_into_chunks
from inference.policy import apply_policy_overrides
from inference.prompt import LABELS, build_prompt, label_token_ids


DEFAULT_ADAPTER = Path("experiments/qwen0.5b-v2")


class DocumentClassifier:
    """Base model + LoRA adapter, scored over the four labels in one forward pass.

    Documents longer than one prompt are split into chunks (see
    inference.chunking) and take the label of their most sensitive chunk.
    """

    def __init__(self, adapter_path=DEFAULT_ADAPTER, device=None):
        self.adapter_path = Path(adapter_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        config = json.loads((self.adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
        base_model = config["base_model_name_or_path"]

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.label_ids = torch.tensor(list(label_token_ids(self.tokenizer).values()))
        self.chunk_budget = document_token_budget(self.tokenizer)

        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
        self.model = PeftModel.from_pretrained(model, str(self.adapter_path))
        self.model.to(self.device).eval()

    def scores(self, text):
        """Label probabilities for text that already fits in one prompt."""
        inputs = self.tokenizer(build_prompt(text), return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        probabilities = torch.softmax(logits.float().cpu()[self.label_ids], dim=0)
        return {label: round(float(p), 4) for label, p in zip(LABELS, probabilities)}

    def classify(self, text, use_policy=True):
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        start = time.perf_counter()
        policy_label = apply_policy_overrides(text, None) if use_policy else None
        if policy_label is not None:
            return {
                "label": policy_label,
                "confidence": 1.0,
                "policy_override": True,
                "model_prediction": None,
                "scores": None,
                "chunks": 0,
                "chunk_labels": [],
                "model_version": self.adapter_path.name,
                "device": self.device,
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }

        chunk_results = []
        for chunk in split_into_chunks(self.tokenizer, text, self.chunk_budget):
            scores = self.scores(chunk)
            chunk_results.append({"label": max(scores, key=scores.get), "scores": scores})

        label = most_sensitive([c["label"] for c in chunk_results])
        deciding = next(c for c in chunk_results if c["label"] == label)
        return {
            "label": label,
            "confidence": deciding["scores"][label],
            "policy_override": False,
            "model_prediction": label,
            "scores": deciding["scores"],
            "chunks": len(chunk_results),
            "chunk_labels": [c["label"] for c in chunk_results],
            "model_version": self.adapter_path.name,
            "device": self.device,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        }
