"""Central hyperparameters and paths for TIGER on Amazon Toys & Games.

Paths can be overridden by env vars so the same code runs locally and on Colab
(where DATA_DIR / EMB_DIR / CKPT_DIR live on mounted Drive).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent.parent  # Assignment 3/

DATA_DIR = Path(os.environ.get("TIGER_DATA_DIR", HERE / "data"))
EMB_DIR = Path(os.environ.get("TIGER_EMB_DIR", HERE / "data" / "embeddings"))
CKPT_DIR = Path(os.environ.get("TIGER_CKPT_DIR", HERE / "data" / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("TIGER_RESULTS_DIR", HERE / "results"))

REVIEWS_FILE = "reviews_Toys_and_Games_5.json.gz"
META_FILE = "meta_Toys_and_Games.json.gz"


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42


# --------------------------------------------------------------------------- #
# Dataset                                                                     #
# --------------------------------------------------------------------------- #
CORE = 5                # iterative k-core filtering
MAX_LEN = 20            # cap user history at 20 items (truncate from the left)


# --------------------------------------------------------------------------- #
# Content embeddings (Sentence-T5)                                            #
# --------------------------------------------------------------------------- #
SENTENCE_ENCODER = "sentence-transformers/sentence-t5-base"   # 768-dim
CONTENT_DIM = 768
CONTENT_BATCH = 64


# --------------------------------------------------------------------------- #
# RQ-VAE                                                                      #
# --------------------------------------------------------------------------- #
RQ_LATENT_DIM = 32              # encoder output dim (paper: 32)
RQ_HIDDEN_DIMS = (512, 256, 128)  # MLP widths between content_dim and latent
RQ_LEVELS = 3                   # L — number of codebook levels (paper: 3)
RQ_CODEBOOK_SIZE = 256          # K — entries per codebook (paper: 256)
RQ_COMMITMENT_BETA = 0.25       # VQ-VAE commitment loss weight

RQ_LR = 1e-3
RQ_WEIGHT_DECAY = 0.0
RQ_BATCH_SIZE = 1024
RQ_EPOCHS = 200
RQ_PATIENCE = 20                # early stop on flat reconstruction loss


# --------------------------------------------------------------------------- #
# Generative Transformer                                                      #
# --------------------------------------------------------------------------- #
T_LAYERS = 4              # encoder layers == decoder layers
T_HEADS = 6
T_HEAD_DIM = 64
T_HIDDEN = T_HEADS * T_HEAD_DIM   # model dim = 384
T_FFN = 1024
T_DROPOUT = 0.1
# Input length = MAX_LEN items * (RQ_LEVELS + 1 disambiguation token)
T_INPUT_LEN = MAX_LEN * (RQ_LEVELS + 1)

T_LR = 1e-3
T_WEIGHT_DECAY = 0.01
T_BATCH_SIZE = 256
T_EPOCHS = 200
T_PATIENCE = 20
T_WARMUP_STEPS = 10_000

BEAM_SIZE = 10            # inference-time beam width (paper: 10)
TOPK = 10                 # max k reported in metrics


# --------------------------------------------------------------------------- #
# Special token ids                                                            #
# --------------------------------------------------------------------------- #
PAD = 0
BOS = 1
EOS = 2
NUM_SPECIALS = 3


# --------------------------------------------------------------------------- #
# Device                                                                      #
# --------------------------------------------------------------------------- #
def get_device() -> torch.device:
    """CUDA > MPS > CPU. Override with env var TIGER_DEVICE."""
    forced = os.environ.get("TIGER_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dirs() -> None:
    for d in (DATA_DIR, EMB_DIR, CKPT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
