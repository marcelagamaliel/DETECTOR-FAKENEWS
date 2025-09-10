import pandas as pd
from sklearn.model_selection import train_test_split

def load_fakebr_csv(path="data/Fake.br-Corpus-master/preprocessed/pre-processed.csv"):
   df = pd.read_csv(path)
   print("Colunas disponíveis:", df.columns)
   print("Exemplo de dados:", df.head())
   return df

def stratified_splits(df, seed=42):
    train, temp = train_test_split(
        df, test_size=0.2, stratify=df.label, random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.5, stratify=temp.label, random_state=seed
    )
    return train, val, test

if __name__ == "__main__":
    df = load_fakebr_csv()

    # Garantir que não haja duplicados
    df = df.drop_duplicates(subset=["preprocessed_news"]).reset_index(drop=True)

    print("Shape inicial:", df.shape)

    train, val, test = stratified_splits(df)

    train.to_csv("data/processed/train.csv", index=False)
    val.to_csv("data/processed/val.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("Train:", train.shape, "Val:", val.shape, "Test:", test.shape)
