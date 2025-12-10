# evaluate_baseline.py (versão final)
import csv
import joblib
from utils_baseline import load_split, evaluate_model

# 1. Carregar conjunto de teste
test_df = load_split("data/processed/test.csv")
X_test = test_df["text"]
y_test = test_df["label"]

# 2. Carregar o vetor TF-IDF
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
X_test_tfidf = vectorizer.transform(X_test)

# 3. Modelos a avaliar
model_paths = {
    "Linear SVM": "models/linear_svm.pkl",
    "Random Forest": "models/random_forest.pkl",
}

# 4. Avaliar e salvar resultados
results = []
for name, path in model_paths.items():
    print(f"\n### Avaliando {name} no conjunto de TESTE ###")
    model = joblib.load(path)
    metrics = evaluate_model(model, X_test_tfidf, y_test, split_name="TEST")
    results.append([name, metrics["accuracy"], metrics["f1"], metrics["auc"]])

# 5. Salvar CSV de resultados
with open("models/results_baseline_test.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Modelo", "Acurácia", "F1-Score", "AUC"])
    writer.writerows(results)

print("\nResultados de teste salvos em: models/results_baseline_test.csv")
