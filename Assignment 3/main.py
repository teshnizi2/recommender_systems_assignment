"""TIGER on Amazon Toys & Games — one CLI to drive every stage.

Stages:
  embed   — sentence-T5 over item content; cached to EMB_DIR
  rqvae   — train RQ-VAE on cached content; extract Semantic IDs
  tiger   — train the generative Transformer + final eval
  all     — runs the three stages in order

Each run writes results/<tag>.json with the history + final metrics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src import config
from src.dataset import build as build_dataset
from src.content import encode as encode_content
from src.rqvae import RQVAE, collision_stats, extract_semantic_ids
from src.train import train_rqvae, train_transformer, set_seed
from src.transformer import build_id_lookup, build_vocab_from_ids, tokenize_items
from src.evaluate import evaluate_transformer


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _save_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # numpy types -> python for JSON
    def _to_py(obj):
        if isinstance(obj, dict):
            return {k: _to_py(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_py(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    with open(path, "w") as f:
        json.dump(_to_py(payload), f, indent=2)


def _load_or_train_rqvae(tag: str, content: np.ndarray, args) -> tuple[RQVAE, dict]:
    ckpt = config.CKPT_DIR / f"{tag}.best.pt"
    if ckpt.exists() and not args.retrain_rqvae:
        device = config.get_device()
        state = torch.load(ckpt, map_location=device, weights_only=False)
        cfg = state["config"]
        print(f"[rqvae] reusing {ckpt}  (epoch {state['epoch']}, "
              f"L={cfg['levels']}, K={cfg['K']})")
        model = RQVAE(
            content_dim=content.shape[1],
            latent_dim=cfg["latent_dim"],
            levels=cfg["levels"],
            codebook_size=cfg["K"],
        ).to(device)
        model.load_state_dict(state["model"])
        return model, {"loaded_from": str(ckpt), "epoch": state["epoch"]}
    return train_rqvae(
        content,
        tag=tag,
        epochs=args.rq_epochs,
        levels=args.rq_levels,
        codebook_size=args.rq_codebook_size,
        latent_dim=args.rq_latent_dim,
    )


# --------------------------------------------------------------------------- #
# Stages                                                                       #
# --------------------------------------------------------------------------- #
def stage_embed(args) -> np.ndarray:
    data = build_dataset()
    return encode_content(data.item_text)


def stage_rqvae(args) -> tuple[np.ndarray, dict]:
    data = build_dataset()
    content = encode_content(data.item_text)
    content_t = torch.from_numpy(content).float()

    model, history = _load_or_train_rqvae(args.rqvae_tag, content, args)
    sem_ids = extract_semantic_ids(model, content_t)
    stats = collision_stats(sem_ids)
    print(f"[rqvae] semantic-ID stats: {stats}")
    np.save(config.DATA_DIR / "semantic_ids.npy", sem_ids)
    return sem_ids, {"rqvae_history": history, "collision_stats": stats}


def stage_tiger(args) -> dict:
    data = build_dataset()
    sem_path = config.DATA_DIR / "semantic_ids.npy"
    if not sem_path.exists():
        raise FileNotFoundError(
            "data/semantic_ids.npy not found — run stage `rqvae` (or `all`) first."
        )
    sem_ids = np.load(sem_path)
    vocab = build_vocab_from_ids(sem_ids)
    item_tokens = tokenize_items(sem_ids, vocab)
    print(f"[tiger] vocab size = {vocab.size}  L={vocab.levels}  K={vocab.codebook_size}  "
          f"suffix_max={vocab.max_suffix}")

    model, history = train_transformer(
        train_seqs=data.train_seqs,
        val_targets=data.val_targets,
        user_full_seqs=data.user_full_seqs,
        item_tokens=item_tokens,
        vocab=vocab,
        tag=args.tiger_tag,
        epochs=args.t_epochs,
        batch_size=args.t_batch_size,
        lr=args.t_lr,
        hidden=args.t_hidden,
        heads=args.t_heads,
        layers=args.t_layers,
        ffn=args.t_ffn,
        dropout=args.t_dropout,
        beam_size=args.beam_size,
        max_train_examples=args.max_train_examples,
        eval_every=args.eval_every,
        eval_max_users=args.eval_max_users,
    )

    id_lookup = build_id_lookup(item_tokens)
    test = evaluate_transformer(
        model=model,
        histories=[seq + [data.val_targets[u]] for u, seq in enumerate(data.train_seqs)],
        targets=data.test_targets,
        user_full_seqs=data.user_full_seqs,
        item_tokens=item_tokens,
        vocab=vocab,
        id_lookup=id_lookup,
        beam_size=args.beam_size,
        ks=(5, 10),
    )
    print(f"[tiger] TEST  recall@5={test['recall@5']:.4f}  ndcg@5={test['ndcg@5']:.4f}  "
          f"recall@10={test['recall@10']:.4f}  ndcg@10={test['ndcg@10']:.4f}  "
          f"invalid={test['invalid_rate']:.3f}")
    return {"tiger_history": history, "test": test}


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="TIGER on Amazon Toys & Games")
    parser.add_argument("--stage", choices=("embed", "rqvae", "tiger", "all"),
                         default="all")
    parser.add_argument("--tag", default="default",
                         help="tag for results/<tag>.json")
    parser.add_argument("--rqvae_tag", default="rqvae_default")
    parser.add_argument("--tiger_tag", default="tiger_default")
    parser.add_argument("--retrain_rqvae", action="store_true",
                         help="ignore existing rqvae checkpoint")

    # RQ-VAE
    parser.add_argument("--rq_levels", type=int, default=config.RQ_LEVELS)
    parser.add_argument("--rq_codebook_size", type=int, default=config.RQ_CODEBOOK_SIZE)
    parser.add_argument("--rq_latent_dim", type=int, default=config.RQ_LATENT_DIM)
    parser.add_argument("--rq_epochs", type=int, default=config.RQ_EPOCHS)

    # Transformer
    parser.add_argument("--t_hidden", type=int, default=config.T_HIDDEN)
    parser.add_argument("--t_heads", type=int, default=config.T_HEADS)
    parser.add_argument("--t_layers", type=int, default=config.T_LAYERS)
    parser.add_argument("--t_ffn", type=int, default=config.T_FFN)
    parser.add_argument("--t_dropout", type=float, default=config.T_DROPOUT)
    parser.add_argument("--t_lr", type=float, default=config.T_LR)
    parser.add_argument("--t_batch_size", type=int, default=config.T_BATCH_SIZE)
    parser.add_argument("--t_epochs", type=int, default=config.T_EPOCHS)
    parser.add_argument("--beam_size", type=int, default=config.BEAM_SIZE)
    parser.add_argument("--max_train_examples", type=int, default=None,
                         help="(debug) cap training examples")
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_max_users", type=int, default=None,
                         help="(debug) cap eval users so beam search runs fast on CPU/MPS")

    args = parser.parse_args()
    set_seed()
    config.ensure_dirs()
    print(f"[main] device = {config.get_device()}")
    t0 = time.perf_counter()

    payload: dict = {"args": vars(args)}
    if args.stage in ("embed", "all"):
        emb = stage_embed(args)
        payload["embed"] = {"shape": list(emb.shape)}
    if args.stage in ("rqvae", "all"):
        sem_ids, rq_payload = stage_rqvae(args)
        payload.update(rq_payload)
    if args.stage in ("tiger", "all"):
        tig_payload = stage_tiger(args)
        payload.update(tig_payload)
    payload["wall_time_sec"] = time.perf_counter() - t0

    out = config.RESULTS_DIR / f"{args.tag}.json"
    _save_results(out, payload)
    print(f"[main] wrote {out}")
    print(f"[main] total wall time: {payload['wall_time_sec']:.1f}s")


if __name__ == "__main__":
    main()
