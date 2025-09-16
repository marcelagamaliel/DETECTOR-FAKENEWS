#!/usr/bin/env python3
"""
make_dataset.py
- Lê o pre-processed CSV do Fake.br Corpus
- Normaliza labels para 0 (REAL) e 1 (FAKE)
- Remove duplicatas/na
- Faz splits estratificados train/val/test = 70/15/15
- Salva em data/processed/
"""
import argparse
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_PATH = "data/Fake.br-Corpus-master/preprocessed/pre-processed.csv"
OUT_DIR = Path("data/processed")

def normalize_label_series(s):
    """
    Recebe uma Series e tenta mapear para 0 = real, 1 = fake.
    Regras:
     - se já for numérico com {0,1} retorna como está
     - senão converte string para lowercase e busca 'fake' -> 1, 'real'/'true' -> 0
    """
    if pd.api.types.is_numeric_dtype(s):
        uniq = sorted(s.dropna().unique().tolist())
        if set(uniq).issubset({0, 1}):
            return s.astype(int)
        # tentar mapear valores 1/0 em outras bases
    # tratar strings
    def map_label(x):
        if pd.isna(x):
            return x
        xs = str(x).strip().lower()
        if "fake" in xs or "falso" in xs:
            return 1
        if "real" in xs or "true" in xs or "verd" in xs:  # 'verdadeiro' -> 'verd'
            return 0
        # tentativas adicionais
        if xs in {"0", "1"}:
            return int(xs)
        # fallback: raise para inspeção
        raise ValueError(f"Rótulo desconhecido: {x}")
    return s.map(map_label).astype(int)

def load_and_prepare(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    df = pd.read_csv(path)
    print("CSV carregado. Colunas:", df.columns.tolist())
    # Esperamos coluna 'preprocessed_news' e 'label' (conforme informado)
    expected_cols = {"preprocessed_news", "label"}
    if not expected_cols.issubset(set(df.columns)):
        raise KeyError(f"O CSV precisa conter as colunas {expected_cols}. Colunas encontradas: {df.columns.tolist()}")

    # Renomear coluna de texto para 'text' internamente
    df = df.rename(columns={"preprocessed_news": "text"}).copy()

    # eliminar linhas vazias e NaN
    n0 = len(df)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    print(f"Removidas {n0 - len(df)} linhas com NaN em text/label.")

    # deduplicar texto (precaução)
    n_before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Removidas {n_before - len(df)} duplicatas exatas de texto.")

    # normalizar rótulos
    try:
        df["label"] = normalize_label_series(df["label"])
    except ValueError as e:
        print("Erro ao normalizar rótulos:", e)
        print("Exemplos únicos encontrados:", df["label"].unique()[:20])
        raise

    return df

def stratified_save(df, out_dir=OUT_DIR, seed=42):
    out_dir.mkdir(parents=True, exist_ok=True)
    # train 70%, val 15%, test 15%
    train, temp = train_test_split(df, test_size=0.30, stratify=df.label, random_state=seed)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp.label, random_state=seed)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print("Splits salvos em:", out_dir)
    print("Train:", train.shape, "Val:", val.shape, "Test:", test.shape)
    # salvar metadados simples
    meta = {
        "total": len(df),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "label_counts": df["label"].value_counts().to_dict()
    }
    (out_dir / "meta.txt").write_text(str(meta))
    print("Meta salvo em meta.txt")

def main(args):
    df = load_and_prepare(args.input)
    stratified_save(df, out_dir=Path(args.outdir), seed=args.seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=DEFAULT_PATH,
                        help="Caminho para pre-processed CSV")
    parser.add_argument("--outdir", type=str, default="data/processed",
                        help="Diretório de saída para train/val/test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
