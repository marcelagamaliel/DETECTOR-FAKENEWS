import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from src.features.text_preprocess import clean_text

def load_splits():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/val.csv")
    test = pd.read_csv("data/processed/test.csv")
    return train, val, test

if __name__ == "__main__":
    # 1) Carregar os splits
    train, val, test = load_splits()

    # 2) Limpar texto
    train["text_clean"] = train["preprocessed_news"].map(clean_text)
    val["text_clean"] = val["preprocessed_news"].map(clean_text)
    test["text_clean"] = test["preprocessed_news"].map(clean_text)

    # 3) Vetorização TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train["text_clean"])
    X_val = vectorizer.transform(val["text_clean"])
    X_test = vectorizer.transform(test["text_clean"])

    y_train, y_val, y_test = train["label"], val["label"], test["label"]

    # 4) Modelo baseline
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    # 5) Avaliar
    preds_val = model.predict(X_val)
    preds_test = model.predict(X_test)

    print("\n🔹 Val Metrics:")
    print(classification_report(y_val, preds_val))
    print("Acc Val:", accuracy_score(y_val, preds_val))

    print("\n🔹 Test Metrics:")
    print(classification_report(y_test, preds_test))
    print("Acc Test:", accuracy_score(y_test, preds_test))

    print("\nMatriz de confusão (Test):")
    print(confusion_matrix(y_test, preds_test))
