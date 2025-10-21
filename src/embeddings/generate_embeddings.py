#!/usr/bin/env python3
# src/embeddings/generate_embeddings.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Paths
ROOT = Path.cwd()
PROC = ROOT / "data" / "processed"
OUT = PROC  # salvar embeddings ao lado dos csvs

MODEL_NAME = "all-MiniLM-L6-v2"  # bom tradeoff velocidade/qualidade

def load_split(name):
    p = PROC / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} não encontrado. Rode make_dataset.py primeiro.")
    return pd.read_csv(p)

def generate_and_save(name, model, batch_size=64):
    df = load_split(name)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    # Gera embeddings em batches
    emb_list = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"encode {name}"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        emb_list.append(emb)
    embeddings = np.vstack(emb_list)

    out_file = OUT / f"embeddings_{name}.npz"
    np.savez_compressed(out_file, embeddings=embeddings, labels=labels)
    print(f"Salvo: {out_file}  shape={embeddings.shape}")

def main():
    print("Carregando modelo:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    for split in ["train", "val", "test"]:
        generate_and_save(split, model, batch_size=64)

if __name__ == "__main__":
    main()
