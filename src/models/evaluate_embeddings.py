# evaluate_embeddings.py (versão final)
from pathlib import Path
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from utils_embeddings import load_split

# 1. Carregar TEST
test_df = load_split("data/processed/test.csv")
X_text = test_df["text"].astype(str).tolist()
y_test = test_df["label"]

# 2. Carregar encoder
encoder_path = Path("models/sentence_transformer_encoder.pkl")
encoder_name = "all-MiniLM-L6-v2"
encoder = joblib.load(encoder_path) if encoder_path.exists() else SentenceTransformer(encoder_name)

# 3. Gerar embeddings do TEST
X_test = encoder.encode(X_text, show_progress_bar=True)

# 4. Carregar modelo
model = joblib.load("models/linear_svm_embeddings.pkl")

# 5. Avaliar
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
try:
    auc = roc_auc_score(y_test, preds)
except:
    auc = None

print("=== Relatório (TEST) ===")
print(classification_report(y_test, preds, digits=4))
print("Matriz de Confusão:")
cm = confusion_matrix(y_test, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão - SVM (Embeddings)")
plt.xlabel("Predito")
plt.ylabel("Real")
plt.show()

print(f"Acurácia: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
if auc:
    print(f"AUC: {auc:.4f}")

# 6. Salvar resultados
pd.DataFrame([{
    "Modelo": "SVM (Embeddings)",
    "Acurácia": acc,
    "F1-Score": f1,
    "AUC": auc
}]).to_csv("models/results_embeddings.csv", index=False)

print("\nResultados salvos em: models/results_embeddings.csv")
