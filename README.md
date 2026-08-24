# EdgeSLM Content Security Classifier

A local small-language-model prototype for classifying company documents as `PUBLIC`, `INTERNAL`, `CONFIDENTIAL`, or `RESTRICTED`.

The system combines a LoRA-adapted SmolLM2 model with deterministic safeguards for high-risk credentials, keys, tokens, passwords, and access-bypass instructions. The model runs locally through a FastAPI service so document text does not need to be sent to a hosted inference API.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The trained adapter is intentionally excluded from Git because model binaries are large. Place the local adapter at `experiments/smollm2-lora-policy` before running inference.

## Run The API

```powershell
python -m uvicorn api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health`
- `GET /version`
- `POST /classify`
- `GET /docs`

## Evaluate

```powershell
python src/evaluation/audit_dataset.py
python src/evaluation/evaluate_adversarial.py data/realistic_test.jsonl --adapter experiments/smollm2-lora-policy --include-guidance
```

Use `--no-policy` to measure the raw model without deterministic overrides.

## Current Status

This is a research prototype. The generated benchmark is not an independent production evaluation. Current work focuses on local serving, safety policy, evaluation rigor, and edge optimization.
