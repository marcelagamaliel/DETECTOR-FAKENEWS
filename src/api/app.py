from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import os
from pydantic import BaseModel
from pathlib import Path

# Caminhos base
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "linear_svm.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"
WEB_PATH = BASE_DIR / "web"

# Cria app
app = FastAPI(title="Fake News Detector API")

# Monta pasta estática
app.mount("/static", StaticFiles(directory=WEB_PATH), name="static")

# Modelo de entrada
class TextRequest(BaseModel):
    text: str

# Carrega modelo e vetor
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

@app.get("/", response_class=HTMLResponse)
def serve_home():
    return FileResponse(WEB_PATH / "index.html")

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = "FAKE" if pred == 1 else "REAL"
    return JSONResponse({"text": text, "prediction": label})
