"""Amazon Toys & Games (2014 McAuley 5-core dump) preprocessing.

Pipeline:
  1. Stream the gzipped reviews + metadata files.
  2. Inner-join on `asin` (every review must have metadata; for the 5-core
     dump this drops nothing — see `scripts/inspect_data.py`).
  3. Iterative 5-core filtering on (users, items).
  4. Build per-user chronological sequences (sorted by unixReviewTime).
  5. Leave-one-out split: train = all but last 2, val target = -2, test = -1.
  6. Cap history at MAX_LEN (truncate from the left if longer).

Outputs cached to `data/cache.pkl` so we only run this once per dump.
"""
from __future__ import annotations

import ast
import gzip
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# Streaming readers                                                            #
# --------------------------------------------------------------------------- #
def _iter_reviews(path: Path) -> Iterator[dict]:
    """Reviews file is one valid JSON object per line."""
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def _iter_metadata(path: Path) -> Iterator[dict]:
    """Metadata file uses Python dict-repr (single quotes) — needs literal_eval."""
    with gzip.open(path, "rt") as f:
        for line in f:
            yield ast.literal_eval(line)


# --------------------------------------------------------------------------- #
# Core dataset object                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class TigerData:
    """Everything downstream stages need from preprocessing.

    Item IDs are int in [1, n_items]; id 0 is reserved for PAD downstream.
    User IDs are int in [0, n_users); we never embed users in TIGER but they
    index the sequence list.
    """
    asin2id: dict[str, int]
    id2asin: list[str]              # length n_items + 1; index 0 unused (PAD)
    item_text: list[str]            # length n_items + 1; index 0 unused
    train_seqs: list[list[int]]     # per user, history (already <= MAX_LEN, last 2 stripped)
    val_targets: list[int]
    test_targets: list[int]
    # raw per-user full chronological sequence (for evaluation: masking known positives)
    user_full_seqs: list[list[int]]

    @property
    def n_items(self) -> int:
        return len(self.id2asin) - 1

    @property
    def n_users(self) -> int:
        return len(self.train_seqs)


# --------------------------------------------------------------------------- #
# Loading + filtering                                                          #
# --------------------------------------------------------------------------- #
def _content_sentence(meta: dict) -> str:
    """Concatenate metadata fields into one sentence for the sentence encoder.

    Following TIGER paper: title + categories + brand + description.
    """
    parts: list[str] = []
    title = (meta.get("title") or "").strip()
    if title:
        parts.append(f"Title: {title}.")

    cats = meta.get("categories") or []
    flat: list[str] = []
    for branch in cats:
        for c in branch:
            if c and c not in flat:
                flat.append(c)
    if flat:
        parts.append("Categories: " + ", ".join(flat) + ".")

    brand = (meta.get("brand") or "").strip()
    if brand:
        parts.append(f"Brand: {brand}.")

    desc = (meta.get("description") or "").strip()
    if desc:
        # cap description to avoid Sentence-T5 truncating uselessly
        if len(desc) > 1500:
            desc = desc[:1500].rsplit(" ", 1)[0] + "..."
        parts.append(f"Description: {desc}")

    return " ".join(parts) if parts else "Unknown toy."


def _iterative_kcore(
    interactions: list[tuple[str, str, int]],
    k: int = config.CORE,
) -> list[tuple[str, str, int]]:
    """Drop users / items with <k interactions iteratively until stable."""
    rows = interactions
    while True:
        u_count: dict[str, int] = defaultdict(int)
        i_count: dict[str, int] = defaultdict(int)
        for u, i, _ in rows:
            u_count[u] += 1
            i_count[i] += 1
        before = len(rows)
        rows = [
            (u, i, t)
            for (u, i, t) in rows
            if u_count[u] >= k and i_count[i] >= k
        ]
        if len(rows) == before:
            return rows


def build() -> TigerData:
    """End-to-end preprocessing. ~30s on a laptop."""
    config.ensure_dirs()
    cache_path = config.DATA_DIR / "cache.pkl"
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------- load reviews
    reviews_path = config.DATA_DIR / config.REVIEWS_FILE
    raw: list[tuple[str, str, int]] = []   # (user, item, unix_time)
    for r in _iter_reviews(reviews_path):
        raw.append((r["reviewerID"], r["asin"], int(r["unixReviewTime"])))
    print(f"[dataset] loaded {len(raw):,} reviews")

    # ------------------------------------------------------------------- 5-core
    rows = _iterative_kcore(raw, k=config.CORE)
    print(f"[dataset] after iterative {config.CORE}-core: {len(rows):,} interactions")

    # ------------------------------------------------------------------- collect ids
    kept_asins: set[str] = {i for _, i, _ in rows}
    kept_users: set[str] = {u for u, _, _ in rows}
    print(f"[dataset]   users: {len(kept_users):,}   items: {len(kept_asins):,}")

    # ------------------------------------------------------------------- metadata
    meta_path = config.DATA_DIR / config.META_FILE
    meta_for_kept: dict[str, dict] = {}
    for m in _iter_metadata(meta_path):
        a = m.get("asin")
        if a in kept_asins:
            meta_for_kept[a] = m
    print(f"[dataset] metadata coverage: {len(meta_for_kept):,} / {len(kept_asins):,}")
    if len(meta_for_kept) < len(kept_asins):
        missing = kept_asins - meta_for_kept.keys()
        # Substitute a stub for any items missing metadata so we keep the same ids
        for a in missing:
            meta_for_kept[a] = {"asin": a, "title": "Toy"}

    # ------------------------------------------------------------------- assign integer ids
    # 1-indexed items so 0 can be PAD throughout the stack.
    asin2id: dict[str, int] = {}
    id2asin: list[str] = ["<PAD>"]
    item_text: list[str] = ["<PAD>"]
    for asin in sorted(kept_asins):
        new_id = len(id2asin)
        asin2id[asin] = new_id
        id2asin.append(asin)
        item_text.append(_content_sentence(meta_for_kept[asin]))

    # ------------------------------------------------------------------- user sequences
    by_user: dict[str, list[tuple[int, int]]] = defaultdict(list)  # user -> [(time, item_id)]
    for u, i, t in rows:
        by_user[u].append((t, asin2id[i]))

    train_seqs: list[list[int]] = []
    val_targets: list[int] = []
    test_targets: list[int] = []
    user_full_seqs: list[list[int]] = []
    skipped = 0
    for u in sorted(by_user.keys()):
        events = sorted(by_user[u])
        seq = [iid for _, iid in events]
        if len(seq) < 3:                       # need at least train/val/test
            skipped += 1
            continue
        test_targets.append(seq[-1])
        val_targets.append(seq[-2])
        history = seq[:-2]                    # everything before val
        if len(history) > config.MAX_LEN:
            history = history[-config.MAX_LEN:]   # truncate from the left
        train_seqs.append(history)
        user_full_seqs.append(seq)
    if skipped:
        print(f"[dataset] dropped {skipped} users with < 3 interactions (post k-core)")
    print(
        f"[dataset] final  users: {len(train_seqs):,}   items: {len(id2asin) - 1:,}   "
        f"avg hist len: {np.mean([len(s) for s in train_seqs]):.1f}"
    )

    data = TigerData(
        asin2id=asin2id,
        id2asin=id2asin,
        item_text=item_text,
        train_seqs=train_seqs,
        val_targets=val_targets,
        test_targets=test_targets,
        user_full_seqs=user_full_seqs,
    )
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[dataset] cached -> {cache_path}")
    return data


if __name__ == "__main__":
    d = build()
    print()
    print(f"users           {d.n_users:>10,}")
    print(f"items           {d.n_items:>10,}")
    lens = np.array([len(s) for s in d.train_seqs])
    print(f"hist len (train) p50/p90/max  {np.percentile(lens, 50):.0f}  "
          f"{np.percentile(lens, 90):.0f}  {lens.max()}")
    print()
    print("example item text:")
    print(f"  id 1 ({d.id2asin[1]}): {d.item_text[1][:200]}...")
