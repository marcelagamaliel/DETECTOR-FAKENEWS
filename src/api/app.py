from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import joblib

# =======================================================
# Configuração básica da API
# =======================================================
app = FastAPI(title="Fake News Detector API (TF-IDF + Random Forest)")

# Permitir chamadas do frontend (HTML local)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # em produção, restrinja ao domínio real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir os arquivos estáticos da pasta /static
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# =======================================================
# Carregar modelo e vetorizador TF-IDF
# =======================================================
MODEL_PATH = "models/random_forest.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

print("Carregando modelo e vetorizador...")
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
print("Modelo Random Forest e TF-IDF carregados com sucesso!")

# =======================================================
# Estrutura do input
# =======================================================
class NewsItem(BaseModel):
    text: str

# =======================================================
# Rota principal de predição
# =======================================================
@app.post("/predict")
def predict(item: NewsItem):
    # Transforma texto em vetor TF-IDF
    X = vectorizer.transform([item.text])

    # Faz a previsão
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    label = "FAKE" if pred == 1 else "REAL"
    confidence = round(float(max(prob)) * 100, 2)

    return {"prediction": label, "confidence": f"{confidence}%"}

# =======================================================
# Rota raiz (opcional: serve o HTML direto)
# =======================================================
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")
