"""Encoder-decoder Transformer that generates Semantic IDs (TIGER section 3.2).

Vocabulary layout (shared across encoder + decoder):
    0                     PAD
    1                     BOS
    2                     EOS
    3 .. 3+L*K-1          codebook tokens, contiguous: level ell occupies
                          [3 + ell*K, 3 + (ell+1)*K)
    3+L*K .. 3+L*K+S-1    disambiguation suffix tokens, where S is the largest
                          collision group + 1

A single item is the (L+1)-token sequence (level_0, level_1, ..., level_L-1, suffix).
A user history of H items is flattened into H*(L+1) tokens for the encoder.
The decoder autoregressively generates a single item -- L+1 tokens -- starting
from BOS, with standard causal masking.

Loss is cross-entropy over the L+1 target tokens; we don't mask the per-position
output vocabulary (the model learns the position -> codebook-range mapping).
Beam search decodes candidate item tuples at inference; tuples that don't map
to a real item are counted as invalid generations and reported.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import config


# --------------------------------------------------------------------------- #
# Vocabulary helpers                                                           #
# --------------------------------------------------------------------------- #
@dataclass
class Vocab:
    levels: int                 # L  (number of codebook levels)
    codebook_size: int          # K
    max_suffix: int             # S  (largest collision group across items)

    @property
    def codebook_start(self) -> int:
        return config.NUM_SPECIALS

    @property
    def suffix_start(self) -> int:
        return config.NUM_SPECIALS + self.levels * self.codebook_size

    @property
    def size(self) -> int:
        return self.suffix_start + self.max_suffix

    def codebook_token(self, level: int, k: int) -> int:
        return self.codebook_start + level * self.codebook_size + k

    def suffix_token(self, s: int) -> int:
        return self.suffix_start + s

    def encode_item_tuple(self, codes: np.ndarray) -> np.ndarray:
        """codes shape (L+1,) -> token-id sequence shape (L+1,)."""
        out = np.empty(self.levels + 1, dtype=np.int64)
        for ell in range(self.levels):
            out[ell] = self.codebook_token(ell, int(codes[ell]))
        out[self.levels] = self.suffix_token(int(codes[self.levels]))
        return out

    def decode_token(self, tok: int) -> tuple[str, int]:
        """Inverse: token id -> ('codebook', (level, k)) or ('suffix', s) or ('special', tok)."""
        if tok < config.NUM_SPECIALS:
            return "special", tok
        rel = tok - self.codebook_start
        if rel < self.levels * self.codebook_size:
            return "codebook", (rel // self.codebook_size, rel % self.codebook_size)
        return "suffix", tok - self.suffix_start


def build_vocab_from_ids(semantic_ids: np.ndarray) -> Vocab:
    """semantic_ids: (n_items+1, L+1) array including row 0 PAD. Last col is suffix."""
    levels = semantic_ids.shape[1] - 1
    codebook_size = int(semantic_ids[1:, :levels].max() + 1)
    # Round codebook_size up to the configured K so the layout is stable.
    codebook_size = max(codebook_size, config.RQ_CODEBOOK_SIZE)
    max_suffix = int(semantic_ids[1:, levels].max() + 1)
    return Vocab(levels=levels, codebook_size=codebook_size, max_suffix=max_suffix)


def tokenize_items(item_codes: np.ndarray, vocab: Vocab) -> np.ndarray:
    """(n_items+1, L+1) codes -> (n_items+1, L+1) token IDs. Row 0 stays all PAD."""
    out = np.zeros_like(item_codes)
    for i in range(1, item_codes.shape[0]):
        out[i] = vocab.encode_item_tuple(item_codes[i])
    return out


def build_id_lookup(item_tokens: np.ndarray) -> dict[tuple[int, ...], int]:
    """tuple of (L+1) token ids -> item id (1-indexed); used to validate beam outputs."""
    lookup: dict[tuple[int, ...], int] = {}
    for i in range(1, item_tokens.shape[0]):
        lookup[tuple(item_tokens[i].tolist())] = i
    return lookup


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #
class _PositionalEmbedding(nn.Module):
    """Learned positional embedding -- TIGER doesn't use sinusoidal."""
    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        self.pos = nn.Embedding(max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        pos = torch.arange(x.size(1), device=x.device)
        return x + self.pos(pos)


class TigerTransformer(nn.Module):
    """Encoder + autoregressive decoder over Semantic ID tokens."""
    def __init__(self,
                 vocab_size: int,
                 hidden: int = config.T_HIDDEN,
                 heads: int = config.T_HEADS,
                 layers: int = config.T_LAYERS,
                 ffn: int = config.T_FFN,
                 dropout: float = config.T_DROPOUT,
                 enc_max_len: int = config.T_INPUT_LEN,
                 dec_max_len: int = config.RQ_LEVELS + 2) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.token_embed = nn.Embedding(vocab_size, hidden, padding_idx=config.PAD)
        self.enc_pos = _PositionalEmbedding(enc_max_len, hidden)
        self.dec_pos = _PositionalEmbedding(dec_max_len, hidden)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=ffn,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers,
                                              norm=nn.LayerNorm(hidden))
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=ffn,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=layers,
                                              norm=nn.LayerNorm(hidden))
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    # ----------------------------------------------------------------- inference helpers
    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """src (B, S): encoder input token ids. Returns (memory (B,S,D), pad_mask (B,S))."""
        pad_mask = src == config.PAD
        x = self.token_embed(src) * math.sqrt(self.hidden)
        x = self.enc_pos(x)
        memory = self.encoder(x, src_key_padding_mask=pad_mask)
        return memory, pad_mask

    def _decode_step(self, tgt_ids: torch.Tensor, memory: torch.Tensor,
                     mem_pad_mask: torch.Tensor) -> torch.Tensor:
        """tgt_ids (B, T'). Returns logits (B, T', V)."""
        y = self.token_embed(tgt_ids) * math.sqrt(self.hidden)
        y = self.dec_pos(y)
        T = tgt_ids.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_ids.device)
        out = self.decoder(y, memory,
                           tgt_mask=causal,
                           memory_key_padding_mask=mem_pad_mask)
        return self.lm_head(out)

    # ----------------------------------------------------------------- training
    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, mem_pad_mask = self.encode(src)
        return self._decode_step(tgt_in, memory, mem_pad_mask)

    # ----------------------------------------------------------------- inference
    @torch.no_grad()
    def beam_search(self, src: torch.Tensor, beam_size: int,
                     seq_len: int, vocab: Vocab,
                     id_lookup: dict[tuple[int, ...], int] | None = None,
                     top_n: int | None = None,
                     ) -> tuple[list[list[tuple[int, ...]]],
                                list[list[float]],
                                list[list[int]]]:
        """Beam search per row of src.

        Returns three parallel lists (one entry per row of src):
          - sequences: top-K generated token tuples
          - logprobs:  matching beam log-probs
          - item_ids:  -1 if the tuple is invalid (not in id_lookup), else the item id

        seq_len is the number of tokens to generate (typically L+1, the
        Semantic-ID length including suffix).
        """
        device = src.device
        B = src.size(0)
        memory, mem_pad_mask = self.encode(src)

        # Expand memory to (B*beam, S, D).
        mem = memory.unsqueeze(1).expand(-1, beam_size, -1, -1).reshape(
            B * beam_size, memory.size(1), memory.size(2))
        mem_mask = mem_pad_mask.unsqueeze(1).expand(-1, beam_size, -1).reshape(
            B * beam_size, mem_pad_mask.size(1))

        # Initial beams: BOS, score 0 for the first beam, -inf for the rest (so
        # that step 1 doesn't multi-count the same continuation).
        tokens = torch.full((B * beam_size, 1), config.BOS, dtype=torch.long, device=device)
        scores = torch.full((B, beam_size), float("-inf"), device=device)
        scores[:, 0] = 0.0
        scores = scores.view(-1)                                  # (B*beam,)

        for _ in range(seq_len):
            logits = self._decode_step(tokens, mem, mem_mask)     # (B*beam, t, V)
            logp = torch.log_softmax(logits[:, -1, :], dim=-1)    # (B*beam, V)
            cand = scores.unsqueeze(-1) + logp                    # (B*beam, V)
            V = cand.size(-1)
            cand = cand.view(B, beam_size * V)                    # (B, beam*V)
            top_scores, top_idx = cand.topk(beam_size, dim=-1)    # (B, beam)
            beam_id = top_idx // V                                # (B, beam)
            tok_id = top_idx % V                                  # (B, beam)

            # Gather chosen prefixes and append new token.
            tokens = tokens.view(B, beam_size, -1)
            new_tokens = torch.gather(
                tokens, 1,
                beam_id.unsqueeze(-1).expand(-1, -1, tokens.size(-1)),
            )
            new_tokens = torch.cat([new_tokens, tok_id.unsqueeze(-1)], dim=-1)
            tokens = new_tokens.view(B * beam_size, -1)
            scores = top_scores.view(-1)

        # Drop the leading BOS column.
        tokens = tokens[:, 1:]                                    # (B*beam, seq_len)
        tokens = tokens.view(B, beam_size, seq_len)
        scores = scores.view(B, beam_size)

        out_seqs: list[list[tuple[int, ...]]] = []
        out_scores: list[list[float]] = []
        out_items: list[list[int]] = []
        top_n = top_n or beam_size
        for b in range(B):
            seqs_b: list[tuple[int, ...]] = []
            scores_b: list[float] = []
            items_b: list[int] = []
            for k in range(beam_size):
                tup = tuple(tokens[b, k].tolist())
                seqs_b.append(tup)
                scores_b.append(float(scores[b, k].item()))
                items_b.append(id_lookup.get(tup, -1) if id_lookup is not None else -1)
                if len(seqs_b) >= top_n:
                    break
            out_seqs.append(seqs_b)
            out_scores.append(scores_b)
            out_items.append(items_b)
        return out_seqs, out_scores, out_items
