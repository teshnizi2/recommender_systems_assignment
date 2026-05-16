"""Plot figures for the TIGER report.

Reads results/*.json (produced by main.py) and writes report/*.pdf and *.png.

Three figure groups:
  1. RQ-VAE training curves + codebook usage per level (for the default run).
  2. Transformer training dynamics (loss + val NDCG@10).
  3. Sorted bar chart of NDCG@10 across all runs (default + ablations).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPORT = HERE / "report"


def _load_all() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(RESULTS.glob("*.json")):
        with open(p) as f:
            out[p.stem] = json.load(f)
    return out


def _save(fig, name: str) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(REPORT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(REPORT / f"{name}.png", dpi=160, bbox_inches="tight")
    print(f"wrote {REPORT / name}.{{pdf,png}}")


def rqvae_panel(runs: dict) -> None:
    if "default" not in runs:
        return
    hist = runs["default"].get("rqvae_history")
    if not hist:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    epochs = np.arange(1, len(hist["recon"]) + 1)
    axes[0].plot(epochs, hist["recon"], label="recon")
    axes[0].plot(epochs, hist["codebook"], label="codebook")
    axes[0].plot(epochs, hist["commit"], label="commit")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].set_yscale("log")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_title("RQ-VAE training")
    usage = np.array(hist["usage"])    # (epochs, L)
    for ell in range(usage.shape[1]):
        axes[1].plot(epochs, usage[:, ell], label=f"level {ell}")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("codebook usage")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(loc="best", fontsize=8)
    axes[1].set_title("RQ-VAE codebook usage")
    _save(fig, "rqvae_panel")
    plt.close(fig)


def tiger_panel(runs: dict) -> None:
    if "default" not in runs:
        return
    hist = runs["default"].get("tiger_history")
    if not hist:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6))
    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    axes[0].plot(epochs, hist["train_loss"])
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss (CE)")
    axes[0].set_title("Transformer training loss")
    if hist.get("val_ndcg10"):
        ve = np.arange(1, len(hist["val_ndcg10"]) + 1)
        axes[1].plot(ve, hist["val_ndcg10"], label="NDCG@10")
        axes[1].plot(ve, hist["val_recall10"], label="Recall@10")
        axes[1].set_xlabel("epoch")
        axes[1].set_ylabel("validation")
        axes[1].set_title("Validation metrics")
        axes[1].legend(loc="best", fontsize=8)
    _save(fig, "tiger_panel")
    plt.close(fig)


def overview_bar(runs: dict) -> None:
    rows = []
    for tag, payload in runs.items():
        test = payload.get("test")
        if not test:
            continue
        rows.append((tag, test["ndcg@10"], test["recall@10"]))
    if not rows:
        return
    rows.sort(key=lambda r: r[1])
    tags = [r[0] for r in rows]
    ndcg = [r[1] for r in rows]
    recall = [r[2] for r in rows]
    fig, ax = plt.subplots(figsize=(6.8, max(2.2, 0.32 * len(rows))))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, ndcg,   0.4, label="NDCG@10")
    ax.barh(y + 0.2, recall, 0.4, label="Recall@10")
    ax.set_yticks(y)
    ax.set_yticklabels(tags)
    ax.set_xlabel("test metric")
    ax.set_title("All configurations")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, "overview_bar")
    plt.close(fig)


def main() -> None:
    runs = _load_all()
    if not runs:
        print("no results/*.json yet")
        return
    print(f"found {len(runs)} runs: {sorted(runs.keys())}")
    rqvae_panel(runs)
    tiger_panel(runs)
    overview_bar(runs)


if __name__ == "__main__":
    main()
