# train_embeddings.py (versão final)
import os
import joblib
from sklearn.svm import LinearSVC
from pathlib import Path
from utils_embeddings import load_split, encode_texts, evaluate_model

os.makedirs("models", exist_ok=True)

# 1. Carregar dados
train_df = load_split("data/processed/train.csv")
val_df = load_split("data/processed/val.csv")

X_train, encoder_model = encode_texts(train_df["text"])
X_val, _ = encode_texts(val_df["text"])
y_train = train_df["label"]
y_val = val_df["label"]

# 2. Definir modelo
models = {
    "Linear SVM (Embeddings)": LinearSVC()
}

# 3. Treinar e avaliar
for name, model in models.items():
    print(f"\n### Treinando {name} ###")
    model.fit(X_train, y_train)
    evaluate_model(model, X_val, y_val, split_name="VAL")
    joblib.dump(model, f"models/{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl")

# 4. Salvar encoder
joblib.dump(encoder_model, "models/sentence_transformer_encoder.pkl")
Path("models/sentence_transformer_name.txt").write_text("all-MiniLM-L6-v2")

print("Modelo SVM + Embeddings e encoder salvos em /models")
