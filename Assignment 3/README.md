# Assignment 3 — TIGER on Amazon Toys & Games

**Run from this folder** (so `data/`, `results/`, `report/` resolve correctly):

```bash
cd "Assignment 3"
```

Reza, Lara, Nithin — Group 15, RS course 2025/26.

We're implementing TIGER (Rajput et al., NeurIPS 2023) — a generative sequential recommender that represents each item as a tuple of discrete **Semantic IDs** and trains an encoder-decoder Transformer to autoregressively generate the Semantic ID of the next item, instead of scoring items in an embedding space the way SASRec (A2) did.

Two stages:
1. **RQ-VAE** quantizes each item's Sentence-T5 content embedding into an `L`-tuple of codebook indices (the Semantic ID). Train it on item content, freeze, dump one Semantic ID per item.
2. **Encoder-decoder Transformer** takes the flattened Semantic ID tokens of a user's history and generates the next item's `L`-tuple token by token. Beam search at inference.

Dataset: **Amazon Toys & Games** (interactions + metadata). Implicit positives, iterative 5-core filtering, leave-one-out (last → test, second-to-last → val), max history 20 items. Eval is full-item ranking, Recall@{5,10} and NDCG@{5,10}.

## Status

Scaffold only. `main.py` and the modules under `src/` are stubs. We've laid out the file structure to mirror A1/A2; implementation TBD.

Deadline: **24 May 23:59** (Brightspace).

## Compute target — Colab T4

This one's heavier than A1/A2 (sentence encoder + RQ-VAE + seq2seq Transformer), so we're running it on Colab with a T4 GPU (free tier):

1. Open https://colab.research.google.com → new notebook.
2. Runtime → Change runtime type → GPU → T4.
3. Verify with `!nvidia-smi` (should show NVIDIA T4, ~15 GB).

Local GPU is fine, but the submitted notebook should still run on Colab T4.

## Drive layout (one-time setup)

Colab's `/content/` is wiped at session end. Keep dataset, cached embeddings, and checkpoints on Drive so we don't repeat the expensive bits:

```
MyDrive/
└── tiger_assignment/
    ├── data/          # raw Amazon Toys & Games files
    ├── embeddings/    # cached Sentence-T5 embeddings
    └── checkpoints/   # RQ-VAE + Transformer checkpoints (per epoch)
```

In every Colab session:

```python
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR = '/content/drive/MyDrive/tiger_assignment/data'
EMB_DIR  = '/content/drive/MyDrive/tiger_assignment/embeddings'
CKPT_DIR = '/content/drive/MyDrive/tiger_assignment/checkpoints'
```

## Dataset

Amazon Product Reviews — Toys & Games category. We need both:

- Reviews / interactions: `userId, itemId (asin), rating, timestamp`
- Item metadata: title, description, categories, brand, price

Sources:
- https://amazon-reviews-2023.github.io/ (2023 dump)
- https://jmcauley.ucsd.edu/data/amazon/ (2014 dump)

Preprocessing (in `src/dataset.py`):
- Merge interactions + metadata on item id.
- Treat every review as a positive interaction (TIGER protocol — no rating threshold).
- Iterative **5-core filtering** until convergence.
- Build per-user chronological sequences; cap length at 20 (truncate left).
- Leave-one-out split: all but last two → train, second-to-last → val, last → test.

Expected scale after 5-core: ~19K users, ~11K items (varies by dump).

## Requirements

```bash
pip install -r requirements.txt
```

The big extra deps vs. A1/A2 are `sentence-transformers` (for Sentence-T5 content embeddings) and `transformers` (in case we use HuggingFace utilities). PyTorch as before.

## How to run (planned)

```bash
# Stage 0 — preprocess + cache Sentence-T5 content embeddings (once per dump)
python main.py --stage embed   --no_download

# Stage 1 — train RQ-VAE on cached embeddings, dump one Semantic ID per item
python main.py --stage rqvae   --rq_levels 3 --rq_codebook_size 256

# Stage 2 — train the generative Transformer on Semantic ID sequences
python main.py --stage tiger   --num_layers 4 --num_heads 6 --hidden 384

# Evaluation (full ranking, beam search) is appended to each training run's JSON
```

`run_experiments.py` will wrap the default config + the ablations we report on (codebook size `K`, levels `L`, Transformer depth/width, beam size).

## Planned default config (from the paper)

- RQ-VAE: encoder MLP → 32-dim latent, `L=3` codebook levels of size `K=256`, suffix token for collisions, standard VQ-VAE losses (reconstruction + commitment + codebook, stop-gradient).
- Transformer: 4 enc + 4 dec layers, 6 heads × 64 = model dim 384, FFN 1024, dropout 0.1.
- Input length: 20 items × (L + 1 suffix) = 80 tokens. Beam size 10–20 at inference. Invalid generations are counted and reported.

## Ablations (from grading rubric)

- RQ-VAE: `K ∈ {64, 128, 256}`, `L ∈ {2, 3, 4}` — codebook usage / collision rate.
- Transformer: hidden size, layers, heads.
- Beam size at inference.
- (Optional) cold-start: hold out a fraction of items from training, measure retrieval quality on them.

## Files (planned)

```
Assignment 3/
├── main.py                    # entry point — picks stage and runs it
├── run_experiments.py         # default + ablations
├── requirements.txt
├── README.md                  ← you are here
├── A3.text                    # raw assignment description from Brightspace
├── src/
│   ├── config.py              # paths, hyperparameters
│   ├── dataset.py             # Amazon Toys & Games loader, 5-core, sequences, split
│   ├── content.py             # Sentence-T5 content embeddings (cached)
│   ├── rqvae.py               # RQ-VAE: encoder + residual quantizer + decoder
│   ├── transformer.py         # encoder-decoder Transformer + beam search
│   ├── train.py               # training loops (RQ-VAE and Transformer)
│   └── evaluate.py            # full-ranking Recall@{5,10} / NDCG@{5,10}
├── data/                      # raw dataset (gitignored; lives on Drive in Colab)
├── results/                   # per-run JSON outputs
└── report/                    # LaTeX write-up (ACM sigconf, as for A1/A2)
```

## Notes / things to watch for

- **Don't joint-train.** RQ-VAE to convergence first, then freeze and extract Semantic IDs, then train the Transformer. The paper is explicit about this and it's also what worked in the reference implementations.
- **Cache content embeddings.** Sentence-T5 over ~11K items isn't huge, but it's wasted compute on every Colab restart. Save to `EMB_DIR` once and reload.
- **Checkpoint every epoch.** Save model + optimizer state to `CKPT_DIR` so a 12h session timeout doesn't cost us a whole run.
- **Invalid generations.** Beam search can produce Semantic ID tuples that no real item maps to. We need to track the rate and either filter or count them as misses for Recall/NDCG — both are valid as long as we report which.
- **Sanity check.** Paper-default on Toys & Games should land near NDCG@5 ≈ 0.03–0.04, Recall@5 ≈ 0.05 (full ranking). Orders of magnitude off ⇒ re-check eval protocol and collision handling before tuning.
