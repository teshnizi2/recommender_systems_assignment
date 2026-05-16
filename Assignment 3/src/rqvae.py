"""Residual-Quantized VAE for TIGER's Semantic IDs.

Architecture (paper defaults):
  Encoder MLP: content_dim (768) -> 512 -> 256 -> 128 -> latent_dim (32)
  Residual quantizer: L=3 levels, K=256 entries per codebook, latent_dim (32)
  Decoder MLP: latent_dim (32) -> 128 -> 256 -> 512 -> content_dim (768)

Each codebook is a learned (K, latent_dim) embedding. The residual quantizer
operates on the latent z:
  r_0 = z
  for ell in 1..L:
      c_ell = argmin_k || r_{ell-1} - codebook_ell[k] ||^2
      e_ell = codebook_ell[c_ell]
      r_ell = r_{ell-1} - e_ell
  z_q   = sum_ell e_ell

Losses (paper / VQ-VAE):
  recon       = || x - decode(z_q) ||^2
  codebook    = sum_ell || sg(r_{ell-1}) - e_ell ||^2
  commitment  = sum_ell || r_{ell-1} - sg(e_ell) ||^2
  total       = recon + codebook + beta * commitment
Straight-through estimator: forward uses z_q, backward copies gradients to z.

After training, each item gets an L-tuple of codebook indices (the Semantic ID).
Items that collide to the same tuple get a disambiguation suffix 0,1,2,... so
every item ends up with a unique (L+1)-token Semantic ID.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


# --------------------------------------------------------------------------- #
# Building blocks                                                              #
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    def __init__(self, dims: list[int], dropout: float = 0.0,
                 final_activation: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == len(dims) - 2
            if (not is_last) or final_activation:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Codebook(nn.Module):
    """A single residual-quantizer level with EMA codebook updates.

    The codebook entries are buffers (not parameters): they are updated as the
    EMA of the encoder outputs that selected them, following the original
    VQ-VAE paper. This is more stable than gradient-based codebook updates and
    avoids the "winner-take-all" collapse seen with random initialization.

    Dead entries (low cluster size) are periodically replaced with random
    encoder-output samples that have been buffered.
    """
    def __init__(self, codebook_size: int, dim: int,
                 decay: float = 0.99, eps: float = 1e-5,
                 dead_code_threshold: float = 1.0) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.decay = decay
        self.eps = eps
        self.dead_code_threshold = dead_code_threshold
        # Codebook is a BUFFER, not a parameter — updated by EMA, not autograd.
        self.register_buffer("embedding", torch.randn(codebook_size, dim) * 0.02)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("cluster_sum", torch.zeros(codebook_size, dim))
        # Diagnostic: tracks how many times each entry has been picked since
        # last reset_usage().
        self.register_buffer("usage_count", torch.zeros(codebook_size, dtype=torch.long))

    def forward(self, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """residual: (B, D). Returns (chosen_vec: (B, D), indices: (B,))."""
        # ||r - e||^2 = ||r||^2 - 2 r . e + ||e||^2
        r2 = (residual ** 2).sum(dim=-1, keepdim=True)
        e2 = (self.embedding ** 2).sum(dim=-1)
        re = residual @ self.embedding.t()
        dist = r2 - 2 * re + e2.unsqueeze(0)
        idx = dist.argmin(dim=-1)
        chosen = self.embedding[idx]

        if self.training:
            with torch.no_grad():
                # EMA codebook update -------------------------------------------------
                one_hot = torch.zeros(idx.size(0), self.codebook_size,
                                       device=idx.device, dtype=residual.dtype)
                one_hot.scatter_(1, idx.unsqueeze(1), 1.0)
                batch_count = one_hot.sum(dim=0)                         # (K,)
                batch_sum = one_hot.t() @ residual                       # (K, D)
                self.cluster_size.mul_(self.decay).add_(
                    batch_count * (1 - self.decay))
                self.cluster_sum.mul_(self.decay).add_(
                    batch_sum * (1 - self.decay))
                # Laplace smoothing to avoid zero-division on cold entries
                n = self.cluster_size.sum()
                smoothed = (
                    (self.cluster_size + self.eps)
                    / (n + self.codebook_size * self.eps) * n
                )
                self.embedding.copy_(self.cluster_sum / smoothed.unsqueeze(-1))
                # diagnostic counter (cumulative usage)
                ones = torch.ones_like(idx, dtype=torch.long)
                self.usage_count.scatter_add_(0, idx, ones)
        return chosen, idx

    @torch.no_grad()
    def reset_dead_codes(self, samples: torch.Tensor) -> int:
        """Replace entries with cluster_size < threshold by random samples.

        Returns the number of entries reset.
        """
        dead = self.cluster_size < self.dead_code_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return 0
        # pick random samples from `samples` (B, D) and copy into dead slots
        idx = torch.randint(samples.size(0), (n_dead,), device=samples.device)
        self.embedding[dead] = samples[idx]
        # mark them as alive with a baseline cluster size
        self.cluster_size[dead] = self.dead_code_threshold
        self.cluster_sum[dead] = self.embedding[dead] * self.dead_code_threshold
        return n_dead


# --------------------------------------------------------------------------- #
# RQ-VAE                                                                       #
# --------------------------------------------------------------------------- #
class RQVAE(nn.Module):
    def __init__(self,
                 content_dim: int = config.CONTENT_DIM,
                 hidden_dims: tuple[int, ...] = config.RQ_HIDDEN_DIMS,
                 latent_dim: int = config.RQ_LATENT_DIM,
                 levels: int = config.RQ_LEVELS,
                 codebook_size: int = config.RQ_CODEBOOK_SIZE,
                 commitment_beta: float = config.RQ_COMMITMENT_BETA) -> None:
        super().__init__()
        self.levels = levels
        self.codebook_size = codebook_size
        self.commitment_beta = commitment_beta

        enc_dims = [content_dim, *hidden_dims, latent_dim]
        dec_dims = [latent_dim, *reversed(hidden_dims), content_dim]
        self.encoder = _MLP(enc_dims)
        self.decoder = _MLP(dec_dims)
        self.codebooks = nn.ModuleList(
            [_Codebook(codebook_size, latent_dim) for _ in range(levels)]
        )
        # Sentence-T5 outputs are anisotropic: most of the norm sits along a
        # shared mean direction and only ~0.4 of the norm is per-item variation.
        # Without centering, the encoder collapses to that mean direction.
        # We center inside encode_latent and re-add the mean in decode (set in
        # initialize_content_stats before training).
        self.register_buffer("content_mean", torch.zeros(content_dim))
        self.register_buffer("_stats_initialized", torch.zeros((), dtype=torch.bool))

    # --------------------------------------------------------------------- API
    @torch.no_grad()
    def initialize_content_stats(self, content: torch.Tensor) -> None:
        """Compute and store the per-dim mean of the content data."""
        self.content_mean.copy_(content.mean(dim=0))
        self._stats_initialized.fill_(True)

    def encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x - self.content_mean)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_q) + self.content_mean

    def quantize(self, z: torch.Tensor):
        """Forward pass through the residual quantizer.

        With EMA codebook updates, the codebook is updated as a buffer inside
        each _Codebook.forward; only the commitment loss is in the autograd
        graph (pushing encoder outputs toward their chosen codebook entry).

        Returns:
          z_q       (B, D) sum of selected codebook vectors (straight-through)
          codes     (B, L) selected codebook indices per level
          codebook_loss   scalar zero tensor (kept for backward-compat callers)
          commit_loss     scalar   pushes encoder -> chosen codebook
        """
        residual = z
        chosen_per_level: list[torch.Tensor] = []
        codes: list[torch.Tensor] = []
        codebook_loss = torch.zeros((), device=z.device)
        commit_loss = torch.zeros((), device=z.device)
        for cb in self.codebooks:
            e, idx = cb(residual)
            # Only commitment loss flows gradients; codebook is updated via EMA
            # inside _Codebook.forward.
            commit_loss = commit_loss + F.mse_loss(residual, e.detach())
            chosen_per_level.append(e)
            codes.append(idx)
            residual = residual - e

        z_q_hard = torch.stack(chosen_per_level, dim=0).sum(dim=0)
        # Straight-through estimator: forward uses z_q_hard; backward copies
        # the gradient onto z.
        z_q = z + (z_q_hard - z).detach()
        codes_t = torch.stack(codes, dim=-1)
        return z_q, codes_t, codebook_loss, commit_loss

    def forward(self, x: torch.Tensor):
        z = self.encode_latent(x)
        z_q, codes, cb_loss, commit_loss = self.quantize(z)
        x_hat = self.decode(z_q)
        recon = F.mse_loss(x_hat, x)
        # codebook is updated by EMA -- cb_loss is always zero in this path.
        total = recon + self.commitment_beta * commit_loss
        return {
            "loss": total,
            "recon": recon,
            "codebook": cb_loss,
            "commit": commit_loss,
            "codes": codes,
            "x_hat": x_hat,
        }

    # --------------------------------------------------------------------- diagnostics
    def codebook_usage(self) -> list[float]:
        """Fraction of each codebook's entries that have ever been used in training."""
        out: list[float] = []
        for cb in self.codebooks:
            used = (cb.usage_count > 0).float().mean().item()
            out.append(used)
        return out

    def reset_usage(self) -> None:
        for cb in self.codebooks:
            cb.usage_count.zero_()

    # --------------------------------------------------------------------- init
    @torch.no_grad()
    def initialize_codebooks(self, content: torch.Tensor, kmeans_iters: int = 20) -> None:
        """Data-dependent codebook init via k-means on encoder outputs (residuals).

        Standard VQ-VAE / TIGER trick: random init causes codebook collapse on
        clustered embedding spaces (e.g. Sentence-T5 outputs). K-means on the
        encoder outputs guarantees the level-0 codebook spans the data; each
        subsequent level k-means the residual after the previous level.

        content: (N, D_content) tensor of content embeddings (no PAD row).
        """
        self.eval()
        z = self.encode_latent(content)              # (N, latent_dim) -- centers via content_mean
        residual = z
        for cb in self.codebooks:
            centers = _kmeans_pp(residual, k=cb.codebook_size, iters=kmeans_iters)
            cb.embedding.copy_(centers)
            # Seed the EMA buffers so the first batch doesn't wash out the
            # k-means init.
            cb.cluster_size.fill_(1.0)
            cb.cluster_sum.copy_(centers)
            # quantize using new codebook to get residual for next level
            r2 = (residual ** 2).sum(-1, keepdim=True)
            e2 = (cb.embedding ** 2).sum(-1)
            re = residual @ cb.embedding.t()
            dist = r2 - 2 * re + e2.unsqueeze(0)
            idx = dist.argmin(dim=-1)
            chosen = cb.embedding[idx]
            residual = residual - chosen
        self.train()


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _kmeans_pp(x: torch.Tensor, k: int, iters: int = 20) -> torch.Tensor:
    """K-means with k-means++ seeding. x: (N, D). Returns (k, D)."""
    n, d = x.shape
    device = x.device
    # k-means++ seeding: pick first center uniformly, subsequent centers
    # proportionally to squared distance to nearest chosen center.
    centers = torch.empty(k, d, device=device, dtype=x.dtype)
    first = torch.randint(n, (1,), device=device).item()
    centers[0] = x[first]
    dist2 = ((x - centers[0]) ** 2).sum(-1)             # (n,)
    for j in range(1, k):
        probs = dist2 / dist2.sum().clamp_min(1e-12)
        next_idx = torch.multinomial(probs, 1).item()
        centers[j] = x[next_idx]
        new_dist2 = ((x - centers[j]) ** 2).sum(-1)
        dist2 = torch.minimum(dist2, new_dist2)

    # Lloyd iterations
    for _ in range(iters):
        x2 = (x ** 2).sum(-1, keepdim=True)
        c2 = (centers ** 2).sum(-1)
        xc = x @ centers.t()
        dist = x2 - 2 * xc + c2.unsqueeze(0)
        labels = dist.argmin(dim=-1)
        # Update centers; keep old centers for empty clusters.
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = x[mask].mean(dim=0)
    return centers


# --------------------------------------------------------------------------- #
# Semantic-ID extraction with collision disambiguation                         #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_semantic_ids(model: RQVAE, content: torch.Tensor,
                          batch_size: int = 1024,
                          device: torch.device | None = None) -> np.ndarray:
    """Return a (n_items+1, L+1) int array of Semantic IDs.

    Last column is the disambiguation suffix (0 for the first item that hits a
    given L-tuple, 1 for the second, ...).  Row 0 is all zeros (PAD).
    """
    model.eval()
    device = device or next(model.parameters()).device
    content = content.to(device)

    codes_per_item: list[np.ndarray] = []
    for start in range(1, content.shape[0], batch_size):
        end = min(start + batch_size, content.shape[0])
        z = model.encode_latent(content[start:end])
        _, codes, _, _ = model.quantize(z)
        codes_per_item.append(codes.cpu().numpy())
    codes = np.concatenate(codes_per_item, axis=0)              # (n_items, L)

    # Disambiguation: for items sharing the same L-tuple, append 0,1,2,...
    seen: dict[tuple[int, ...], int] = defaultdict(int)
    suffix = np.zeros(codes.shape[0], dtype=np.int64)
    for i, tup in enumerate(map(tuple, codes.tolist())):
        suffix[i] = seen[tup]
        seen[tup] += 1
    full = np.concatenate([codes, suffix[:, None]], axis=1)     # (n_items, L+1)

    # Prepend a row of zeros (PAD)
    out = np.zeros((content.shape[0], full.shape[1]), dtype=np.int64)
    out[1:] = full
    return out


def collision_stats(semantic_ids: np.ndarray) -> dict:
    """Stats over the L-tuple part (suffix excluded)."""
    L = semantic_ids.shape[1] - 1
    tuples = [tuple(row[:L].tolist()) for row in semantic_ids[1:]]
    counts: dict[tuple, int] = defaultdict(int)
    for t in tuples:
        counts[t] += 1
    n_items = len(tuples)
    unique = len(counts)
    in_collision = sum(c for c in counts.values() if c > 1)
    max_collision = max(counts.values())
    return {
        "n_items": n_items,
        "unique_tuples": unique,
        "collision_rate": (n_items - unique) / n_items,
        "items_in_collision": in_collision,
        "max_collision_group": max_collision,
        "mean_collision_group": float(np.mean(list(counts.values()))),
    }
