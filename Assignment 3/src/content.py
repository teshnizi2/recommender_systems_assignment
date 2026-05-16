"""Sentence-T5 content embeddings for every item.

Runs the encoder once per dataset dump, caches the (n_items+1, 768) tensor
to EMB_DIR/content_sentence-t5-base.npy. Subsequent calls just memory-map it.

On a T4 this is a few minutes; on M1 MPS about 10-15 minutes; on CPU ~1h.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from . import config


def _cache_path(model_name: str) -> Path:
    safe = model_name.replace("/", "_")
    return config.EMB_DIR / f"content_{safe}.npy"


def encode(item_text: list[str], model_name: str | None = None,
           batch_size: int = config.CONTENT_BATCH,
           device: torch.device | None = None) -> np.ndarray:
    """Return a (n_items+1, dim) float32 array. Row 0 is zeros (PAD).

    Lazy-loaded from cache if it exists; the cache key is only the model name,
    so the item list must be in a stable order (we always sort by asin in
    dataset.build()).
    """
    config.ensure_dirs()
    model_name = model_name or config.SENTENCE_ENCODER
    cache = _cache_path(model_name)
    if cache.exists():
        emb = np.load(cache, mmap_mode="r")
        if emb.shape[0] == len(item_text):
            print(f"[content] loaded cached embeddings  shape={emb.shape}  from {cache.name}")
            return np.asarray(emb)
        print(f"[content] cached shape {emb.shape} != items {len(item_text)} — recomputing")

    # Lazy import — sentence-transformers is a slow import.
    from sentence_transformers import SentenceTransformer

    device = device or config.get_device()
    print(f"[content] encoding {len(item_text) - 1:,} items with {model_name} on {device}")
    model = SentenceTransformer(model_name, device=str(device))

    # Skip index 0 (PAD); encode the rest in one shot.
    real = item_text[1:]
    embs = model.encode(
        real,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embs = embs.astype(np.float32, copy=False)   # MPS sometimes returns fp16
    out = np.zeros((len(item_text), embs.shape[1]), dtype=np.float32)
    out[1:] = embs
    np.save(cache, out)
    print(f"[content] cached -> {cache}")
    return out


if __name__ == "__main__":
    from . import dataset
    d = dataset.build()
    emb = encode(d.item_text)
    print(f"shape = {emb.shape}")
    print(f"row 0 (PAD) norm = {np.linalg.norm(emb[0]):.4f}  (expect 0)")
    print(f"row 1 norm       = {np.linalg.norm(emb[1]):.4f}")
    print(f"mean cos sim row1 vs row2-5 = "
          f"{(emb[1] @ emb[2:6].T / (np.linalg.norm(emb[1]) * np.linalg.norm(emb[2:6], axis=1))).mean():.4f}")
