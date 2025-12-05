# train_baseline.py (versão final)
import os
import csv
import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from utils_baseline import load_split, vectorize, evaluate_model

os.makedirs("models", exist_ok=True)

# 1. Carregar dados
train_df = load_split("data/processed/train.csv")
val_df = load_split("data/processed/val.csv")

# 2. Vetorização TF-IDF
X_train, X_val, vectorizer = vectorize(train_df["text"], val_df["text"])
y_train = train_df["label"]
y_val = val_df["label"]

# 3. Modelos a comparar
models = {
    "Linear SVM": LinearSVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

# 4. Treinar, avaliar e salvar modelos
results = []
for name, model in models.items():
    print(f"\n### Treinando {name} ###")
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_val, y_val, split_name="VAL")
    joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")
    results.append([name, metrics["accuracy"], metrics["f1"], metrics["auc"]])

# Salvar o vetor TF-IDF
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# 5. Salvar resultados em CSV
with open("models/results_baseline_val.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Modelo", "Acurácia", "F1-Score", "AUC"])
    writer.writerows(results)

print("\nResultados de validação salvos em: models/results_baseline_val.csv")
