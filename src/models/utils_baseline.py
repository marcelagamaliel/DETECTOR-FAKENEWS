import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
import numpy as np


def load_split(split_path: str) -> pd.DataFrame:
    """Carrega um split salvo em CSV (train.csv, val.csv ou test.csv)."""
    return pd.read_csv(split_path)


def vectorize(train_texts, val_texts, max_features=5000):
    """Transforma textos em vetores TF-IDF com n-gramas até 2."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    return X_train, X_val, vectorizer


def evaluate_model(model, X, y, split_name="VAL"):
    """Avalia o modelo e imprime métricas principais. Retorna métricas numéricas."""
    preds = model.predict(X)

    # Impressão detalhada
    print(f"\n=== Resultados em {split_name} ===")
    print(classification_report(y, preds, digits=4))
    print("Matriz de Confusão:")
    print(confusion_matrix(y, preds))

    # Métricas principais
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)

    # AUC opcional (apenas se o modelo suportar)
    try:
        if hasattr(model, "decision_function"):
            auc = roc_auc_score(y, model.decision_function(X))
        elif hasattr(model, "predict_proba"):
            auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    if not np.isnan(auc):
        print(f"AUC: {auc:.4f}")

    # Retorno estruturado
    return {"accuracy": acc, "f1": f1, "auc": auc}
