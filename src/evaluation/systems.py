"""The systems compared in the ablation.

Every system exposes ``predict(text) -> (label, confidence)`` where ``label`` is
one of the four classes, ``"UNKNOWN"`` (model produced no parseable label), or
``"ABSTAIN"`` (rule-based system had no rule fire). Both non-answers are scored
as incorrect; the harness reports their rate separately.
"""

import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.chunking import document_token_budget, most_sensitive, split_into_chunks
from inference.policy import apply_policy_overrides
from inference.prompt import LABELS, build_prompt, label_token_ids


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"


def adapter_base_model(adapter):
    """The base model an adapter was trained on, from its own config."""
    config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
    return config["base_model_name_or_path"]


def extract_label(response):
    response = response.upper().strip()
    # Longest-first so CONFIDENTIAL is not shadowed by a substring match.
    for label in ("RESTRICTED", "CONFIDENTIAL", "INTERNAL", "PUBLIC"):
        if label in response:
            return label
    return "UNKNOWN"


class GenerativeSystem:
    """Shared decode paths for the PyTorch and ONNX builds.

    ``decode="generate"`` is the v1 path: greedy-decode a few tokens and look
    for a label in the text, which can fail to produce one (UNKNOWN).
    ``decode="score"`` runs a single forward pass and picks whichever of the
    four labels' first tokens has the highest logit, so it always answers and
    needs no decode loop. Subclasses set ``self.model``, ``self.tokenizer`` and
    ``self.device``.
    """

    guidance = True
    use_policy = False
    device = "cpu"
    decode = "score"

    def _init_labels(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_token_ids = label_token_ids(self.tokenizer)
        self.chunk_budget = document_token_budget(self.tokenizer, guidance=self.guidance)
        self._label_index = torch.tensor(list(self.label_token_ids.values()))

    def _label_probabilities(self, next_token_logits):
        scores = next_token_logits.float().cpu()[self._label_index]
        return torch.softmax(scores, dim=0)

    def _inputs(self, text):
        # Only the v1 generate path uses this; it keeps v1's truncation so the
        # recorded v1 numbers stay reproducible.
        return self.tokenizer(
            build_prompt(text, self.guidance),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

    def _score_chunk(self, text):
        inputs = self.tokenizer(build_prompt(text, self.guidance), return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        probabilities = self._label_probabilities(logits)
        index = int(probabilities.argmax())
        return LABELS[index], round(float(probabilities[index]), 4)

    def _score(self, text):
        results = [
            self._score_chunk(chunk)
            for chunk in split_into_chunks(self.tokenizer, text, self.chunk_budget)
        ]
        label = most_sensitive([label for label, _ in results])
        return next(result for result in results if result[0] == label)

    def _generate(self, text):
        inputs = self._inputs(text)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        response = self.tokenizer.decode(
            output.sequences[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        confidence = round(float(self._label_probabilities(output.scores[0][0]).max()), 4)
        return extract_label(response), confidence

    def predict(self, text):
        # Matches DocumentClassifier: when the policy layer is on, a firing rule
        # short-circuits the model entirely.
        if self.use_policy:
            forced = apply_policy_overrides(text, None)
            if forced is not None:
                return forced, 1.0

        if self.decode == "score":
            return self._score(text)
        return self._generate(text)

    def close(self):
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ModelSystem(GenerativeSystem):
    """SmolLM2 under PyTorch, optionally with a LoRA adapter and policy layer."""

    def __init__(
        self,
        adapter=None,
        guidance=True,
        use_policy=False,
        device=None,
        dtype=None,
        decode="score",
        base_model=None,
    ):
        self.adapter = adapter
        self.guidance = guidance
        self.use_policy = use_policy
        self.decode = decode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        resolved_dtype = dtype or (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        base_model = base_model or (adapter_base_model(adapter) if adapter else BASE_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._init_labels()

        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=resolved_dtype)
        if adapter:
            model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model.to(self.device).eval()


def _graph_has_kv_cache(path):
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    return any(i.name.startswith("past_key_values") for i in graph.input)


class OnnxSystem(GenerativeSystem):
    """ONNX Runtime build of the merged adapter, fp32 or quantised.

    Produced by ``src/edge/export_onnx.py``. ONNX Runtime here targets CPU,
    which is the deployment this project is named for.
    """

    def __init__(self, model_dir, guidance=True, use_policy=False, decode="score"):
        import pathlib

        from optimum.onnxruntime import ORTModelForCausalLM

        self.model_dir = str(model_dir)
        self.guidance = guidance
        self.use_policy = use_policy
        self.decode = decode
        self.device = "cpu"

        # The quantiser writes model_quantized.onnx, which is not a name
        # from_pretrained looks for. Without an explicit file_name it warns and
        # falls back to an untrained graph that scores at chance, so the file is
        # resolved here rather than left to the default lookup.
        candidates = sorted(pathlib.Path(self.model_dir).glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(f"no .onnx file in {self.model_dir}")
        if len(candidates) > 1:
            raise ValueError(
                f"expected one .onnx file in {self.model_dir}, "
                f"found {[c.name for c in candidates]}"
            )

        self.model_file = candidates[0].name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._init_labels()
        self.model = ORTModelForCausalLM.from_pretrained(
            self.model_dir,
            file_name=self.model_file,
            use_cache=_graph_has_kv_cache(candidates[0]),
        )


class PolicyOnlySystem:
    """The regex layer with no model behind it.

    Answers ABSTAIN when no rule fires, which makes the rule layer's true
    coverage visible instead of hiding it behind a model fallback.
    """

    def predict(self, text):
        label = apply_policy_overrides(text, None)
        if label is None:
            return "ABSTAIN", 0.0
        return label, 1.0

    def close(self):
        pass


class TfidfSystem:
    """Classical baseline trained on exactly the data the LoRA adapter saw."""

    def __init__(self, train_rows):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        self.pipeline = make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                strip_accents="unicode",
            ),
            LogisticRegression(max_iter=2000, C=4.0),
        )
        self.pipeline.fit(
            [row["text"] for row in train_rows],
            [row["label"] for row in train_rows],
        )

    def predict(self, text):
        probabilities = self.pipeline.predict_proba([text])[0]
        index = int(probabilities.argmax())
        return str(self.pipeline.classes_[index]), round(float(probabilities[index]), 4)

    def close(self):
        pass
