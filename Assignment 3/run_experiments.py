"""Drive the default-config run + the ablations we report on.

Each entry is (tag, extra args appended to main.py). All runs share the same
preprocessed dataset and Sentence-T5 cache; RQ-VAE ablations train a new
quantizer per K/L variant (`--rqvae_tag` is unique so each gets its own
checkpoint). Transformer-architecture ablations and beam-size ablations
reuse the default RQ-VAE Semantic IDs.

Order matters: the K/L ablations train RQ-VAEs and overwrite
data/semantic_ids.npy. Plain Transformer ablations need the default RQ-VAE
IDs in place, so we (a) save semantic_ids per K/L run, (b) restore the
default after each, and (c) run Transformer ablations last.

Run with:
    python run_experiments.py            # everything
    python run_experiments.py --only default     # just the default-config run
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEM_PATH = HERE / "data" / "semantic_ids.npy"
SEM_BACKUP_DIR = HERE / "data" / "semantic_ids_backup"


# --------------------------------------------------------------------------- #
# Experiment list                                                              #
# --------------------------------------------------------------------------- #
RQVAE_ABLATIONS = [
    ("rqvae_K64",  ["--rq_codebook_size", "64"]),
    ("rqvae_K128", ["--rq_codebook_size", "128"]),
    ("rqvae_L2",   ["--rq_levels", "2"]),
    ("rqvae_L4",   ["--rq_levels", "4"]),
]

TIGER_ABLATIONS = [
    ("tiger_small",   ["--t_hidden", "192", "--t_heads", "3"]),
    ("tiger_large",   ["--t_hidden", "768", "--t_heads", "12"]),
    ("tiger_deep",    ["--t_layers", "6"]),
    ("tiger_shallow", ["--t_layers", "2"]),
]

BEAM_ABLATIONS = [
    ("tiger_beam5",  ["--beam_size",  "5"]),
    ("tiger_beam20", ["--beam_size", "20"]),
]


# --------------------------------------------------------------------------- #
# Drivers                                                                      #
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> None:
    print("\n" + "=" * 80)
    print(" ".join(cmd))
    print("=" * 80, flush=True)
    t0 = time.perf_counter()
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise SystemExit(f"command failed: {' '.join(cmd)}  -> exit {ret.returncode}")
    print(f"[{cmd[-1]}] wall time: {time.perf_counter() - t0:.1f}s")


def run_default() -> None:
    """Default-config run: RQ-VAE (L=3 K=256) -> Transformer (4 layers, dim 384, beam 10)."""
    _run([sys.executable, "main.py", "--stage", "all", "--tag", "default",
          "--rqvae_tag", "rqvae_default", "--tiger_tag", "tiger_default"])


def run_rqvae_ablations() -> None:
    for tag, extra in RQVAE_ABLATIONS:
        _run([sys.executable, "main.py", "--stage", "rqvae",
              "--tag", tag, "--rqvae_tag", tag, *extra])
        # snapshot the semantic_ids so we can recover later if needed
        SEM_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        if SEM_PATH.exists():
            shutil.copy(SEM_PATH, SEM_BACKUP_DIR / f"{tag}.npy")


def restore_default_semantic_ids() -> None:
    """Re-extract default Semantic IDs (cheap — just runs the cached RQ-VAE)."""
    _run([sys.executable, "main.py", "--stage", "rqvae",
          "--tag", "rqvae_default_restore",
          "--rqvae_tag", "rqvae_default"])


def run_tiger_ablations() -> None:
    for tag, extra in TIGER_ABLATIONS:
        _run([sys.executable, "main.py", "--stage", "tiger",
              "--tag", tag, "--tiger_tag", tag, *extra])


def run_beam_ablations() -> None:
    for tag, extra in BEAM_ABLATIONS:
        _run([sys.executable, "main.py", "--stage", "tiger",
              "--tag", tag, "--tiger_tag", "tiger_default", *extra])


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", default="all",
        choices=("all", "default", "rqvae", "tiger", "beam"),
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    if args.only in ("all", "default"):
        run_default()
    if args.only in ("all", "rqvae"):
        run_rqvae_ablations()
        restore_default_semantic_ids()
    if args.only in ("all", "tiger"):
        run_tiger_ablations()
    if args.only in ("all", "beam"):
        run_beam_ablations()
    print(f"\nALL DONE in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
