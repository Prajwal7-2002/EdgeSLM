# Quickstart ⚡

Get EdgeSLM running locally on your workstation or server in under 2 minutes.

---

## 1. Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and pinned in `uv.lock`. On Windows, it automatically installs PyTorch with CUDA 12.4 support; on Linux, it installs the lightweight CPU build.

```bash
git clone https://github.com/Prajwal7-2002/EdgeSLM.git
cd EdgeSLM
uv sync
```

### Run Unit Tests
Run the complete test suite to verify extraction, chunking, policy scanning, and model abstractions:

```bash
uv run pytest
# Output: 51 passed in ~8s (Zero GPU or model weights required)
```

---

## 2. Start the Local Privacy API

Launch the local zero-data-leakage inference server:

```powershell
uv run uvicorn src.api.app:app --port 8000
```

Once running:
- **Interactive UI**: Open **`http://127.0.0.1:8000/docs`** in your browser. Drag and drop any `.pdf`, `.docx`, `.eml`, `.txt`, `.md`, or `.env` file directly into `/classify-file` for an immediate security verdict.
- **Single Prompt Classification**:
  ```bash
  curl -X POST http://127.0.0.1:8000/classify \
    -H "Content-Type: application/json" \
    -d '{"text": "Use svc_ingest / Copper!2291 for the ledger service until Monday."}'
  ```

### Example JSON Response
```json
{
  "label": "RESTRICTED",
  "confidence": 1.0,
  "policy_override": true,
  "override_reason": "high-entropy credential assignment",
  "scores": {
    "PUBLIC": 0.0,
    "INTERNAL": 0.0,
    "CONFIDENTIAL": 0.0,
    "RESTRICTED": 1.0
  },
  "chunks": 1,
  "latency_ms": 0.85
}
```

---

## 3. Scanning Folders & Directories

Scan an entire directory of mixed documents recursively and export audit results to CSV:

```powershell
uv run python src/scan.py path/to/folder --csv results/scan.csv
```

`/classify-file` and `src/scan.py` support:
- PDF documents (`.pdf` with extractable text layers)
- Word documents (`.docx`)
- Email archives (`.eml` with RFC 822 header parsing)
- Configuration & Code (`.env`, `.yaml`, `.json`, `.py`, `.sql`)

---

## 4. Evaluating on Real Documents (`real-bench-100`)

To run the balanced 100-document real benchmark (SEC 8-K filings, Enron emails, GitLab SOPs, and real cryptographic keys) without manual spreadsheet labeling:

```powershell
uv run python src/data/build_real_bench_100.py
```
