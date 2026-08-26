"""The systems compared in the ablation.

Every system exposes ``predict(text) -> (label, confidence)`` where ``label`` is
one of the four classes, ``"UNKNOWN"`` (model produced no parseable label), or
``"ABSTAIN"`` (rule-based system had no rule fire). Both non-answers are scored
as incorrect; the harness reports their rate separately.
"""

import gc

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference.policy import apply_policy_overrides


BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

LABELS = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")

CLASSIFICATION_GUIDANCE = (
    "PUBLIC = approved information intended for external sharing.\n"
    "INTERNAL = routine company-only procedures or workflows.\n"
    "CONFIDENTIAL = private business, employee, customer, or contract information.\n"
    "RESTRICTED = credentials, keys, tokens, passwords, or access-bypass instructions."
)


def build_prompt(text, guidance=True):
    """Prompt used for training and inference; must stay in sync with training."""
    if guidance:
        return (
            "Classify the sensitivity of the following company document.\n\n"
            f"Use these definitions:\n{CLASSIFICATION_GUIDANCE}\n\n"
            f"Document:\n{text}\n\n"
            "Classification:"
        )
    return (
        "Classify the sensitivity of the following company document.\n\n"
        f"Document:\n{text}\n\n"
        "Classification:"
    )


def extract_label(response):
    response = response.upper().strip()
    # Longest-first so CONFIDENTIAL is not shadowed by a substring match.
    for label in ("RESTRICTED", "CONFIDENTIAL", "INTERNAL", "PUBLIC"):
        if label in response:
            return label
    return "UNKNOWN"


class GenerativeSystem:
    """Shared greedy-decode path.

    Subclasses set ``self.model``, ``self.tokenizer`` and ``self.device``. The
    decode path mirrors ``inference.classifier.DocumentClassifier`` so the
    ablation measures the configuration the API actually serves.
    """

    guidance = True
    use_policy = False
    device = "cpu"

    def _init_labels(self):
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.label_token_ids = {
            label: self.tokenizer(" " + label, add_special_tokens=False)["input_ids"][0]
            for label in LABELS
        }

    def _confidence(self, first_token_logits):
        scores = torch.stack(
            [first_token_logits[i] for i in self.label_token_ids.values()]
        )
        return round(float(torch.softmax(scores.float(), dim=0).max()), 4)

    def predict(self, text):
        # Matches DocumentClassifier: when the policy layer is on, a firing rule
        # short-circuits the model entirely.
        if self.use_policy:
            forced = apply_policy_overrides(text, None)
            if forced is not None:
                return forced, 1.0

        inputs = self.tokenizer(
            build_prompt(text, self.guidance),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

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
        label = extract_label(response)
        confidence = self._confidence(output.scores[0][0])

        if self.use_policy:
            label = apply_policy_overrides(text, label)
        return label, confidence

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
    ):
        self.adapter = adapter
        self.guidance = guidance
        self.use_policy = use_policy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        resolved_dtype = dtype or (
            torch.float16 if self.device == "cuda" else torch.float32
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self._init_labels()

        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=resolved_dtype)
        if adapter:
            model = PeftModel.from_pretrained(model, str(adapter))
        self.model = model.to(self.device).eval()


class OnnxSystem(GenerativeSystem):
    """ONNX Runtime build of the merged adapter, fp32 or dynamic INT8.

    Produced by ``src/edge/export_onnx.py``. ONNX Runtime here targets CPU,
    which is the deployment this project is named for.
    """

    def __init__(self, model_dir, guidance=True, use_policy=False):
        import pathlib

        from optimum.onnxruntime import ORTModelForCausalLM

        self.model_dir = str(model_dir)
        self.guidance = guidance
        self.use_policy = use_policy
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
            self.model_dir, file_name=self.model_file
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
