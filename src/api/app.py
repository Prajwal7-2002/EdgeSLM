import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from inference.classifier import DEFAULT_ADAPTER, DocumentClassifier


app = FastAPI(title="EdgeSLM Content Security Classifier", version="0.1.0")
classifier = None


class ClassificationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    use_policy: bool = True


@app.on_event("startup")
def load_classifier():
    global classifier
    adapter = os.getenv("EDGESLM_ADAPTER", str(DEFAULT_ADAPTER))
    classifier = DocumentClassifier(adapter_path=adapter)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.get("/version")
def version():
    adapter = os.getenv("EDGESLM_ADAPTER", str(DEFAULT_ADAPTER))
    return {"service_version": app.version, "adapter": Path(adapter).name}


@app.post("/classify")
def classify(request: ClassificationRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    try:
        return classifier.classify(
            request.text,
            use_policy=request.use_policy,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
