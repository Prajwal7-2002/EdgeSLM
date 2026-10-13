import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from inference.classifier import DEFAULT_ADAPTER, DocumentClassifier
from inference.extract import UnsupportedFile, extract_text


MAX_UPLOAD_BYTES = 20 * 1024 * 1024


classifier = None


def adapter_path():
    return os.getenv("EDGESLM_ADAPTER", str(DEFAULT_ADAPTER))


@asynccontextmanager
async def lifespan(_app):
    global classifier
    classifier = DocumentClassifier(adapter_path=adapter_path())
    yield
    classifier = None


app = FastAPI(title="EdgeSLM Content Security Classifier", version="0.3.0", lifespan=lifespan)


class ClassificationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=100_000)
    use_policy: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.get("/version")
def version():
    return {"service_version": app.version, "adapter": Path(adapter_path()).name}


@app.post("/classify")
def classify(request: ClassificationRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    try:
        return classifier.classify(request.text, use_policy=request.use_policy)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# A plain def, not async: classification is seconds of CPU work, and FastAPI
# runs sync endpoints in a worker thread so the event loop stays free.
@app.post("/classify-file")
def classify_file(file: UploadFile = File(...), use_policy: bool = True):
    if classifier is None:
        raise HTTPException(status_code=503, detail="model is not loaded")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file is larger than 20 MB")
    try:
        text = extract_text(data, file.filename or "")
    except UnsupportedFile as error:
        raise HTTPException(status_code=415, detail=str(error)) from error
    result = classifier.classify(text, use_policy=use_policy)
    return {"filename": file.filename, "characters": len(text), **result}
