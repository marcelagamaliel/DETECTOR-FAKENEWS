#!/usr/bin/env python3
# Gera e salva embeddings para train/val/test (opcional, útil p/ análises rápidas).

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

ROOT = Path.cwd()
PROC = ROOT / "data" / "processed"
OUT = PROC  # salva ao lado dos CSVs
MODEL_NAME = "all-MiniLM-L6-v2"  # mesmo do train_embeddings

def load_split(name: str) -> pd.DataFrame:
    p = PROC / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} não encontrado. Rode make_dataset.py primeiro.")
    return pd.read_csv(p)

def generate_and_save(name: str, model: SentenceTransformer, batch_size: int = 128):
    df = load_split(name)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(int).to_numpy()

    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"encode {name}"):
        batch = texts[i:i+batch_size]
        emb = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype(np.float32)
        embs.append(emb)

    embeddings = np.vstack(embs)
    out_file = OUT / f"embeddings_{name}.npz"
    np.savez_compressed(out_file, embeddings=embeddings, labels=labels)
    print(f"✅ Salvo: {out_file}  shape={embeddings.shape}")

def main():
    print("Carregando modelo:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    for split in ["train", "val", "test"]:
        generate_and_save(split, model, batch_size=128)

    # meta simples
    meta = {
        "model_name": MODEL_NAME,
        "files": [f"embeddings_{s}.npz" for s in ["train", "val", "test"]],
        "dtype": "float32",
    }
    (OUT / "embeddings_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"📝 Meta salvo em: {OUT / 'embeddings_meta.json'}")

if __name__ == "__main__":
    main()
