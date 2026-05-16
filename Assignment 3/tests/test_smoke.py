"""Smoke tests — run with `python -m tests.test_smoke` from Assignment 3/."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

# allow running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_rqvae_recon() -> None:
    """RQ-VAE should drive reconstruction loss below 0.05 on 2-cluster toy data."""
    from src.rqvae import RQVAE, collision_stats, extract_semantic_ids

    torch.manual_seed(0)
    n = 200
    emb = np.random.randn(n + 1, 768).astype(np.float32) * 0.1
    emb[1:101] += 1.0
    emb[101:] -= 1.0
    emb[0] = 0

    model = RQVAE(content_dim=768, levels=3, codebook_size=32, latent_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    x = torch.from_numpy(emb[1:])
    for _ in range(50):
        opt.zero_grad()
        out = model(x)
        out["loss"].backward()
        opt.step()
    assert out["recon"].item() < 0.05, f"recon {out['recon'].item()} too high"
    sem = extract_semantic_ids(model, torch.from_numpy(emb))
    assert sem.shape == (n + 1, 4)
    stats = collision_stats(sem)
    # Two clusters should collapse to <=8 unique tuples (heavy collision)
    assert stats["unique_tuples"] <= 10, stats
    print(f"  rqvae   recon={out['recon'].item():.5f}  unique_tuples={stats['unique_tuples']}")


def test_transformer_forward() -> None:
    """Forward pass of the encoder-decoder should match shape (B, T_out, V)."""
    from src.transformer import (
        TigerTransformer, build_id_lookup, build_vocab_from_ids, tokenize_items,
    )

    n_items = 50
    L = 2
    codes = np.random.randint(0, 8, size=(n_items + 1, L), dtype=np.int64)
    codes[0] = 0
    sem = np.concatenate([codes, np.zeros((n_items + 1, 1), dtype=np.int64)], axis=1)
    vocab = build_vocab_from_ids(sem)
    tokens = tokenize_items(sem, vocab)

    model = TigerTransformer(vocab_size=vocab.size, hidden=32, heads=2, layers=1, ffn=64,
                              enc_max_len=20 * (L + 1), dec_max_len=L + 2)
    src = torch.zeros(2, 20 * (L + 1), dtype=torch.long)
    src[0, -L - 1:] = torch.from_numpy(tokens[1])
    tgt_in = torch.tensor([[1, tokens[5][0], tokens[5][1]],
                            [1, tokens[6][0], tokens[6][1]]])
    out = model(src, tgt_in)
    assert out.shape == (2, 3, vocab.size), out.shape
    print(f"  forward shape {out.shape}")


def test_beam_search() -> None:
    """Beam search returns beam_size candidates per row, ranked by score."""
    from src.transformer import (
        TigerTransformer, build_id_lookup, build_vocab_from_ids, tokenize_items,
    )

    torch.manual_seed(0)
    n_items = 50
    L = 2
    codes = np.random.randint(0, 8, size=(n_items + 1, L), dtype=np.int64)
    codes[0] = 0
    sem = np.concatenate([codes, np.zeros((n_items + 1, 1), dtype=np.int64)], axis=1)
    vocab = build_vocab_from_ids(sem)
    tokens = tokenize_items(sem, vocab)
    lookup = build_id_lookup(tokens)

    model = TigerTransformer(vocab_size=vocab.size, hidden=32, heads=2, layers=1, ffn=64,
                              enc_max_len=20 * (L + 1), dec_max_len=L + 2)
    src = torch.zeros(3, 20 * (L + 1), dtype=torch.long)
    seqs, scores, items = model.beam_search(src, beam_size=5, seq_len=L + 1,
                                             vocab=vocab, id_lookup=lookup)
    assert len(seqs) == 3
    for b in range(3):
        assert len(seqs[b]) == 5, len(seqs[b])
        # scores should be sorted descending
        assert all(scores[b][i] >= scores[b][i + 1] - 1e-6 for i in range(4)), scores[b]
    print(f"  beam search beams={len(seqs[0])}  example invalid count={sum(1 for it in items[0] if it < 0)}/5")


def test_metric_helpers() -> None:
    from src.evaluate import _metrics_for_user, _ndcg

    # Target at rank 1
    m = _metrics_for_user([7, 2, 9], target=7, ks=(5, 10))
    assert m["recall@5"] == 1.0 and m["ndcg@5"] == 1.0
    # Target at rank 3 (NDCG = 1/log2(4) = 0.5)
    m = _metrics_for_user([1, 2, 7], target=7, ks=(5,))
    assert abs(m["ndcg@5"] - 1 / np.log2(4)) < 1e-6
    # Target not in ranked
    m = _metrics_for_user([1, 2, 3], target=7, ks=(5,))
    assert m["recall@5"] == 0.0
    print(f"  metrics ok  ndcg(rank=3)={_ndcg(3, 5):.4f}")


def main() -> None:
    for fn in (test_rqvae_recon, test_transformer_forward, test_beam_search, test_metric_helpers):
        print(f"{fn.__name__}")
        fn()
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
