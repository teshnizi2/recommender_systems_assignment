"""Training loops for both TIGER stages.

Stage 1 (RQ-VAE): MSE recon + commit + codebook on content embeddings.
Stage 2 (Transformer): standard seq2seq cross-entropy over Semantic ID tokens.

Both stages checkpoint after every epoch so a Colab disconnect doesn't kill us.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from . import config
from .rqvae import RQVAE, collision_stats, extract_semantic_ids
from .transformer import (
    TigerTransformer, Vocab, build_id_lookup, build_vocab_from_ids,
    tokenize_items,
)


# --------------------------------------------------------------------------- #
# Seed                                                                         #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = config.RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# RQ-VAE training                                                              #
# --------------------------------------------------------------------------- #
class _ContentDataset(Dataset):
    def __init__(self, content: np.ndarray) -> None:
        # skip index 0 (PAD)
        self.x = torch.from_numpy(content[1:]).float()

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> torch.Tensor:
        return self.x[i]


def train_rqvae(content: np.ndarray, tag: str = "rqvae",
                 epochs: int = config.RQ_EPOCHS,
                 batch_size: int = config.RQ_BATCH_SIZE,
                 lr: float = config.RQ_LR,
                 weight_decay: float = config.RQ_WEIGHT_DECAY,
                 patience: int = config.RQ_PATIENCE,
                 levels: int = config.RQ_LEVELS,
                 codebook_size: int = config.RQ_CODEBOOK_SIZE,
                 latent_dim: int = config.RQ_LATENT_DIM,
                 warmup_epochs: int = 20,
                 verbose: bool = True,
                 ) -> tuple[RQVAE, dict]:
    """Train and return (model, history).

    Three-phase training:
      1. Encoder + decoder warmup (no quantization) for `warmup_epochs` to get
         diverse encoder outputs.
      2. K-means initialization of every codebook level on warmed-up encoder outputs.
      3. Joint training with the residual quantizer.
    """
    set_seed()
    config.ensure_dirs()
    device = config.get_device()
    if verbose:
        print(f"[rqvae] device={device}  items={content.shape[0]-1:,}  L={levels}  K={codebook_size}")

    ds = _ContentDataset(content)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)

    model = RQVAE(
        content_dim=content.shape[1],
        latent_dim=latent_dim,
        levels=levels,
        codebook_size=codebook_size,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict = {"recon": [], "codebook": [], "commit": [], "usage": [],
                      "epoch_time": [], "warmup_recon": []}

    # -- Phase 0: initialize content_mean buffer ------------------------------
    full = torch.from_numpy(content[1:]).float().to(device)
    model.initialize_content_stats(full)
    if verbose:
        cm = model.content_mean.detach()
        centered_std = (full - cm).std(dim=0).mean().item()
        print(f"[rqvae] content_mean ||={cm.norm().item():.4f}  "
              f"centered_var={((full - cm) ** 2).sum(-1).mean().item():.4f}  "
              f"centered per-dim std={centered_std:.4f}")

    # Denoising sigma -- noise on (x - mean) of magnitude comparable to the
    # per-dim std prevents trivial encoder collapse: the decoder needs the
    # encoder output to know which item is being reconstructed.
    NOISE_SIGMA = 0.5 * centered_std

    # -- Phase 1: denoising encoder/decoder warmup ----------------------------
    if verbose:
        print(f"[rqvae] phase 1: denoising warmup ({warmup_epochs} epochs, sigma={NOISE_SIGMA:.4f})")
    import torch.nn.functional as F
    for ep in range(1, warmup_epochs + 1):
        model.train()
        sum_loss = 0.0
        for x in loader:
            x = x.to(device)
            opt.zero_grad()
            # noise is added to the centered input -- encoder sees noisy, but
            # the loss compares the decoder output to the clean x.
            noisy = x + torch.randn_like(x) * NOISE_SIGMA
            z = model.encode_latent(noisy)
            x_hat = model.decode(z)
            loss = F.mse_loss(x_hat, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sum_loss += loss.item() * x.size(0)
        avg = sum_loss / len(ds)
        history["warmup_recon"].append(avg)
        if verbose and (ep == 1 or ep % 5 == 0 or ep == warmup_epochs):
            # also probe encoder spread on a fixed slice
            with torch.no_grad():
                z_probe = model.encode_latent(full[:2048])
                z_std = z_probe.std(dim=0).mean().item()
            print(f"[rqvae] warmup ep {ep:3d}  recon={avg:.5f}  z_std={z_std:.4f}")

    # -- Phase 2: k-means init codebooks on warmed-up encoder outputs ---------
    if verbose:
        print(f"[rqvae] phase 2: k-means init of {levels} codebooks (K={codebook_size})")
    model.initialize_codebooks(full)
    if verbose:
        # quick check: how diverse are the level-0 choices on real data?
        with torch.no_grad():
            z = model.encode_latent(full[:2048])
            _, codes, _, _ = model.quantize(z)
            unique_lvl0 = int(codes[:, 0].unique().numel())
            print(f"[rqvae]   level-0 unique codes on 2048-sample probe: {unique_lvl0}/{codebook_size}")

    # -- Phase 3: joint training with quantization + EMA codebooks ------------
    if verbose:
        print(f"[rqvae] phase 3: joint training ({epochs} epochs)")
    best_recon = float("inf")
    no_improve = 0
    ckpt_best = config.CKPT_DIR / f"{tag}.best.pt"

    for ep in range(1, epochs + 1):
        model.train()
        model.reset_usage()
        t0 = time.perf_counter()
        recon_sum = cb_sum = co_sum = 0.0
        for x in loader:
            x = x.to(device)
            opt.zero_grad()
            # keep denoising during joint training to maintain encoder spread
            noisy = x + torch.randn_like(x) * NOISE_SIGMA
            out = model(noisy)
            # but the recon target is the clean x -- override out["recon"]/loss
            x_hat = out["x_hat"]
            recon = F.mse_loss(x_hat, x)
            loss = recon + model.commitment_beta * out["commit"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            n = x.size(0)
            recon_sum += recon.item() * n
            cb_sum += out["codebook"].item() * n
            co_sum += out["commit"].item() * n
        n_total = len(ds)
        recon = recon_sum / n_total
        cb = cb_sum / n_total
        co = co_sum / n_total
        usage = model.codebook_usage()
        # Dead-code reset every 5 epochs to keep all codebook entries useful
        dead_reset = 0
        if ep % 5 == 0:
            with torch.no_grad():
                z_samples = model.encode_latent(full)
                residual = z_samples
                for cb_mod in model.codebooks:
                    dead_reset += cb_mod.reset_dead_codes(residual)
                    # recompute residual through fresh codebook for next level
                    r2 = (residual ** 2).sum(-1, keepdim=True)
                    e2 = (cb_mod.embedding ** 2).sum(-1)
                    re = residual @ cb_mod.embedding.t()
                    dist = r2 - 2 * re + e2.unsqueeze(0)
                    idx = dist.argmin(dim=-1)
                    residual = residual - cb_mod.embedding[idx]
        elapsed = time.perf_counter() - t0
        history["recon"].append(recon)
        history["codebook"].append(cb)
        history["commit"].append(co)
        history["usage"].append(usage)
        history["epoch_time"].append(elapsed)
        if verbose:
            extra = f"  dead_reset={dead_reset}" if dead_reset else ""
            print(f"[rqvae] ep {ep:3d}  recon={recon:.5f}  commit={co:.5f}  "
                  f"usage={['%.2f'%u for u in usage]}{extra}  {elapsed:.1f}s")
        # early stop on recon plateau
        if recon < best_recon - 1e-5:
            best_recon = recon
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "config": {"levels": levels, "K": codebook_size,
                                   "latent_dim": latent_dim}}, ckpt_best)
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"[rqvae] early stop at ep {ep} (best recon {best_recon:.5f})")
                break

    # restore best
    state = torch.load(ckpt_best, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    history["best_recon"] = best_recon
    return model, history


# --------------------------------------------------------------------------- #
# Transformer training                                                         #
# --------------------------------------------------------------------------- #
class _SeqDataset(Dataset):
    """Builds training instances by sliding next-item targets across each user's history.

    For a user with chronological items [i_1, ..., i_H] (history = train_seqs[u]) we
    produce H-1 training instances: (history[:k], history[k]) for k=1..H-1.
    Each instance becomes (src tokens, target tokens).
    """
    def __init__(self, train_seqs: list[list[int]], item_tokens: np.ndarray,
                  vocab: Vocab, max_seq_len: int = config.MAX_LEN) -> None:
        self.item_tokens = item_tokens                          # (n_items+1, L+1)
        self.vocab = vocab
        self.tokens_per_item = item_tokens.shape[1]             # L+1
        self.src_len = max_seq_len * self.tokens_per_item
        self.examples: list[tuple[np.ndarray, int]] = []
        for seq in train_seqs:
            for k in range(1, len(seq)):
                history = seq[max(0, k - max_seq_len):k]
                target = seq[k]
                self.examples.append((np.array(history, dtype=np.int64), target))

    def __len__(self) -> int:
        return len(self.examples)

    def _flatten(self, items: np.ndarray) -> np.ndarray:
        # items: (h,) item ids -> (h*(L+1),) token ids
        rows = self.item_tokens[items]                          # (h, L+1)
        return rows.reshape(-1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history, target = self.examples[idx]
        src_tokens = self._flatten(history)
        # left-pad src to fixed length
        if src_tokens.shape[0] < self.src_len:
            pad = np.zeros(self.src_len - src_tokens.shape[0], dtype=np.int64)
            src_tokens = np.concatenate([pad, src_tokens], axis=0)
        else:
            src_tokens = src_tokens[-self.src_len:]
        tgt_full = self.item_tokens[target]                     # (L+1,)
        tgt_in = np.concatenate([[config.BOS], tgt_full[:-1]], axis=0)
        tgt_out = tgt_full
        return (
            torch.from_numpy(src_tokens).long(),
            torch.from_numpy(tgt_in).long(),
            torch.from_numpy(tgt_out).long(),
        )


def train_transformer(train_seqs: list[list[int]],
                       val_targets: list[int],
                       user_full_seqs: list[list[int]],
                       item_tokens: np.ndarray,
                       vocab: Vocab,
                       tag: str = "tiger",
                       epochs: int = config.T_EPOCHS,
                       batch_size: int = config.T_BATCH_SIZE,
                       lr: float = config.T_LR,
                       weight_decay: float = config.T_WEIGHT_DECAY,
                       patience: int = config.T_PATIENCE,
                       warmup_steps: int = config.T_WARMUP_STEPS,
                       hidden: int = config.T_HIDDEN,
                       heads: int = config.T_HEADS,
                       layers: int = config.T_LAYERS,
                       ffn: int = config.T_FFN,
                       dropout: float = config.T_DROPOUT,
                       beam_size: int = config.BEAM_SIZE,
                       max_train_examples: int | None = None,
                       eval_every: int = 1,
                       eval_max_users: int | None = None,
                       verbose: bool = True,
                       ) -> tuple[TigerTransformer, dict]:
    from .evaluate import evaluate_transformer

    set_seed()
    config.ensure_dirs()
    device = config.get_device()
    if verbose:
        print(f"[tiger] device={device}  vocab={vocab.size}  L={vocab.levels}  K={vocab.codebook_size}  "
              f"suffix_max={vocab.max_suffix}  hidden={hidden}  layers={layers}  heads={heads}")

    ds = _SeqDataset(train_seqs, item_tokens, vocab, max_seq_len=config.MAX_LEN)
    if max_train_examples is not None and max_train_examples < len(ds):
        # used for quick smoke tests; takes a random subset
        rng = np.random.default_rng(config.RANDOM_SEED)
        keep = rng.choice(len(ds), size=max_train_examples, replace=False)
        ds.examples = [ds.examples[i] for i in keep]
    if verbose:
        print(f"[tiger] training examples: {len(ds):,}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = TigerTransformer(
        vocab_size=vocab.size,
        hidden=hidden, heads=heads, layers=layers, ffn=ffn, dropout=dropout,
        enc_max_len=config.MAX_LEN * (vocab.levels + 1),
        dec_max_len=vocab.levels + 2,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                             betas=(0.9, 0.98), eps=1e-9)

    # Inverse-sqrt schedule with linear warmup
    def lr_at(step: int) -> float:
        step = max(step, 1)
        if step < warmup_steps:
            return step / warmup_steps
        return (warmup_steps / step) ** 0.5

    loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD, label_smoothing=0.0)
    id_lookup = build_id_lookup(item_tokens)

    history: dict = {"train_loss": [], "val_ndcg10": [], "val_recall10": [], "lr": [],
                      "epoch_time": []}
    best_ndcg = -1.0
    no_improve = 0
    ckpt_best = config.CKPT_DIR / f"{tag}.best.pt"
    step = 0
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum = 0.0
        n_batches = 0
        for src, tgt_in, tgt_out in loader:
            src = src.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)

            step += 1
            for g in opt.param_groups:
                g["lr"] = lr * lr_at(step)

            opt.zero_grad()
            logits = model(src, tgt_in)
            loss = loss_fn(logits.reshape(-1, vocab.size), tgt_out.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            n_batches += 1

        train_loss = loss_sum / max(n_batches, 1)
        history["train_loss"].append(train_loss)
        history["lr"].append(opt.param_groups[0]["lr"])

        do_eval = (ep % eval_every == 0) or (ep == epochs)
        if do_eval:
            val = evaluate_transformer(
                model=model,
                histories=train_seqs,
                targets=val_targets,
                user_full_seqs=user_full_seqs,
                item_tokens=item_tokens,
                vocab=vocab,
                id_lookup=id_lookup,
                beam_size=beam_size,
                ks=(5, 10),
                device=device,
                max_users=eval_max_users,
            )
            history["val_ndcg10"].append(val["ndcg@10"])
            history["val_recall10"].append(val["recall@10"])
        else:
            val = {"ndcg@10": float("nan"), "recall@10": float("nan"),
                   "invalid_rate": float("nan")}
        elapsed = time.perf_counter() - t0
        history["epoch_time"].append(elapsed)
        if verbose:
            print(f"[tiger] ep {ep:3d}  loss={train_loss:.4f}  "
                  f"val_ndcg@10={val['ndcg@10']:.4f}  val_recall@10={val['recall@10']:.4f}  "
                  f"invalid={val['invalid_rate']:.3f}  lr={history['lr'][-1]:.2e}  "
                  f"{elapsed:.1f}s")

        if do_eval and val["ndcg@10"] > best_ndcg + 1e-6:
            best_ndcg = val["ndcg@10"]
            no_improve = 0
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "best_ndcg10": best_ndcg,
                        "vocab": vocab.__dict__,
                        "config": {"hidden": hidden, "heads": heads,
                                   "layers": layers, "ffn": ffn, "dropout": dropout}},
                       ckpt_best)
        elif do_eval:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"[tiger] early stop at ep {ep} (best val NDCG@10 {best_ndcg:.4f})")
                break

    state = torch.load(ckpt_best, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    history["best_val_ndcg10"] = best_ndcg
    return model, history
