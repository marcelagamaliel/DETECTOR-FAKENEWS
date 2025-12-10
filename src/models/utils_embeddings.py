"""
Funções utilitárias para geração e avaliação de embeddings usando SentenceTransformers.
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)

def load_split(split_path: str) -> pd.DataFrame:
    """Carrega um split salvo em CSV (train.csv, val.csv ou test.csv)."""
    return pd.read_csv(split_path)

def encode_texts(texts, model_name: str = "all-MiniLM-L6-v2"):
    """
    Gera embeddings densos a partir de uma lista de textos.
    Retorna (np.ndarray[float32], SentenceTransformer).
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
    list(map(str, texts)),
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=False # manter bruto p/ SVM linear
    ).astype(np.float32)
    return embeddings, model

def evaluate_model(model, X, y, split_name: str = "VAL"):
    """
    Avalia o modelo e imprime métricas principais.
    Retorna dict com accuracy, f1, auc.
    - AUC calculada com scores contínuos (decision_function ou predict_proba).
    """
    preds = model.predict(X)
    print(f"\n=== Resultados em {split_name} ===")
    print(classification_report(y, preds, digits=4))
    print("Matriz de Confusão:")
    print(confusion_matrix(y, preds))

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    # AUC com pontuações contínuas
    auc = np.nan
    try:
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            auc = roc_auc_score(y, scores)
        elif hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, proba)
    except Exception:
        pass

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    if not np.isnan(auc):
        print(f"AUC: {auc:.4f}")

    return {"accuracy": acc, "f1": f1, "auc": auc}
