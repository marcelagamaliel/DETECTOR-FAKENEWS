import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from utils_baseline import load_split, vectorize, evaluate_model

# garantir que a pasta models exista
os.makedirs("models", exist_ok=True)

# 1. Carregar dados
train_df = load_split("data/processed/train.csv")
val_df = load_split("data/processed/val.csv")

# 2. Vetorização TF-IDF
X_train, X_val, vectorizer = vectorize(train_df["text"], val_df["text"])

y_train = train_df["label"]
y_val = val_df["label"]

# 3. Definir modelos para comparar
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Linear SVM": LinearSVC(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
}

# 4. Treinar e avaliar
for name, model in models.items():
    print(f"\n### Treinando {name} ###")
    model.fit(X_train, y_train)
    evaluate_model(model, X_val, y_val, split_name="VAL")
    joblib.dump(model, f"models/{name.replace(' ', '_').lower()}.pkl")

# Salvar também o vetor TF-IDF
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
