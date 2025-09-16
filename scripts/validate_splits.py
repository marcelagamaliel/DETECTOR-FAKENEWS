#!/usr/bin/env python3
"""
validate_splits.py
Valida existência e distribuição das splits geradas.
"""
import pandas as pd
from pathlib import Path

PROC_DIR = Path("data/processed")

def load_csv(name):
    p = PROC_DIR / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} não encontrado. Rode make_dataset.py primeiro.")
    return pd.read_csv(p)

if __name__ == "__main__":
    for name in ["train", "val", "test"]:
        df = load_csv(name)
        print(f"\n=== {name.upper()} ({len(df)}) ===")
        print(df["label"].value_counts(normalize=False))
        print(df["label"].value_counts(normalize=True).round(3))
        print("Exemplo (first 3):")
        print(df[["text", "label"]].head(3).to_string(index=False))
