"""TIGER full-item-ranking evaluation.

Beam-search decode a list of candidate Semantic IDs for each user. Each
candidate maps to an item via the (L+1)-token tuple lookup; invalid candidates
(tuples that don't match any item) are counted and skipped.

We rank candidates by beam score, drop already-seen items from the user's
history (TIGER protocol: don't recommend re-purchases), then compute
Recall@K and NDCG@K against the held-out target.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import config


# --------------------------------------------------------------------------- #
# Dataset used at eval time                                                    #
# --------------------------------------------------------------------------- #
class _EvalDataset(Dataset):
    def __init__(self, histories, item_tokens, max_seq_len: int):
        self.histories = histories
        self.item_tokens = item_tokens
        self.tokens_per_item = item_tokens.shape[1]
        self.src_len = max_seq_len * self.tokens_per_item

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, i: int) -> torch.Tensor:
        hist = self.histories[i]
        if len(hist) == 0:
            tokens = np.zeros(self.src_len, dtype=np.int64)
        else:
            rows = self.item_tokens[np.array(hist, dtype=np.int64)]
            tokens = rows.reshape(-1)
            if tokens.shape[0] < self.src_len:
                pad = np.zeros(self.src_len - tokens.shape[0], dtype=np.int64)
                tokens = np.concatenate([pad, tokens], axis=0)
            else:
                tokens = tokens[-self.src_len:]
        return torch.from_numpy(tokens).long()


# --------------------------------------------------------------------------- #
# Metrics                                                                      #
# --------------------------------------------------------------------------- #
def _ndcg(rank: int, k: int) -> float:
    """1-indexed rank. Returns DCG/IDCG @k for a single relevant item."""
    if rank > k or rank <= 0:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def _metrics_for_user(ranked_items: list[int], target: int, ks: Iterable[int]) -> dict[str, float]:
    out = {}
    rank = -1
    for r, item in enumerate(ranked_items, start=1):
        if item == target:
            rank = r
            break
    for k in ks:
        if rank == -1 or rank > k:
            out[f"recall@{k}"] = 0.0
            out[f"ndcg@{k}"] = 0.0
        else:
            out[f"recall@{k}"] = 1.0
            out[f"ndcg@{k}"] = _ndcg(rank, k)
    return out


# --------------------------------------------------------------------------- #
# Top-level eval                                                               #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_transformer(model, histories, targets, user_full_seqs, item_tokens,
                          vocab, id_lookup, beam_size: int = config.BEAM_SIZE,
                          ks=(5, 10), batch_size: int = 64,
                          device=None, max_users: int | None = None) -> dict:
    """Full-ranking eval (TIGER protocol).

    For each user, beam-search the top-`beam_size` candidate Semantic IDs.
    Map candidates back to items (invalid ones counted). Drop items the user
    already interacted with (besides the target), then compute Recall@K /
    NDCG@K.
    """
    device = device or next(model.parameters()).device
    model.eval()
    if max_users is not None and max_users < len(histories):
        rng = np.random.default_rng(config.RANDOM_SEED)
        keep = rng.choice(len(histories), size=max_users, replace=False)
        keep = sorted(keep.tolist())
        histories = [histories[i] for i in keep]
        targets = [targets[i] for i in keep]
        user_full_seqs = [user_full_seqs[i] for i in keep]
    ds = _EvalDataset(histories, item_tokens, max_seq_len=config.MAX_LEN)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    seq_len = vocab.levels + 1
    metrics_sum = {f"recall@{k}": 0.0 for k in ks}
    metrics_sum.update({f"ndcg@{k}": 0.0 for k in ks})
    n = 0
    invalid_total = 0
    invalid_per_user = 0.0
    cursor = 0

    for src in loader:
        src = src.to(device)
        seqs, scores, items = model.beam_search(
            src, beam_size=beam_size, seq_len=seq_len, vocab=vocab,
            id_lookup=id_lookup,
        )
        bsz = src.size(0)
        for b in range(bsz):
            user_idx = cursor + b
            target = targets[user_idx]
            full_history = set(user_full_seqs[user_idx])
            # Order beams by score descending (beam_search already returns in that order).
            ranked: list[int] = []
            seen_items: set[int] = set()
            invalid_here = 0
            for it in items[b]:
                if it < 0:
                    invalid_here += 1
                    continue
                if it == target:
                    if it not in seen_items:
                        ranked.append(it)
                        seen_items.add(it)
                    continue
                if it in full_history:
                    continue   # already seen by this user
                if it in seen_items:
                    continue
                ranked.append(it)
                seen_items.add(it)
            invalid_total += invalid_here
            invalid_per_user += invalid_here / max(len(items[b]), 1)

            user_metrics = _metrics_for_user(ranked, target, ks)
            for k_name, v in user_metrics.items():
                metrics_sum[k_name] += v
            n += 1
        cursor += bsz

    out = {k: v / max(n, 1) for k, v in metrics_sum.items()}
    out["invalid_rate"] = invalid_per_user / max(n, 1)
    out["invalid_total"] = float(invalid_total)
    out["n_users"] = n
    out["beam_size"] = beam_size
    return out
