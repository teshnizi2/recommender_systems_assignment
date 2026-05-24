# Assignment 3 — TIGER on Amazon Toys & Games

**Run from this folder** (so `data/`, `results/`, `report/` resolve correctly):

```bash
cd "Assignment 3"
```

Reza, Lara, Nithin — Group 15, RS course 2025/26.

We implement TIGER (Rajput et al., NeurIPS 2023) — a generative sequential recommender that represents each item as a tuple of discrete **Semantic IDs** and trains an encoder-decoder Transformer to autoregressively generate the Semantic ID of the next item, instead of scoring items in an embedding space the way SASRec (A2) did.

Three stages:
1. **Sentence-T5** content encoding — concatenate each item's title + categories + brand + description and run through `sentence-t5-base` to get a 768-dim content vector. Cached to disk.
2. **RQ-VAE** quantizes each content vector into an `L`-tuple of codebook indices (the Semantic ID). Train it on item content, freeze, dump one Semantic ID per item.
3. **Encoder-decoder Transformer** takes the flattened Semantic ID tokens of a user's history and generates the next item's `L`-tuple token by token. Beam search at inference (beam = 10).

Dataset: **Amazon Toys & Games (2014 5-core dump)**. Implicit positives, iterative 5-core filtering, leave-one-out split (last → test, second-to-last → val), max history 20 items. Eval is full-item ranking, Recall@{5,10} and NDCG@{5,10}, with already-seen items filtered out before ranking.

## Results

Final default-config test metrics (single seed, beam = 10):

| Metric | Value |
|---|---|
| Recall@5 | 0.0210 |
| NDCG@5   | 0.0150 |
| Recall@10 | 0.0276 |
| NDCG@10  | 0.0172 |
| Invalid generation rate | 0.17 % |

RQ-VAE: 7.7 % collision rate (11 006 / 11 924 unique L-tuples), 99 %+ codebook utilisation across all three levels, reconstruction MSE 2.2×10⁻⁴ on centred Sentence-T5 vectors.

See `report/report.pdf` for the full write-up.

## Compute

We trained on two paths:
- **Kaggle T4 (×2 slot, single GPU used)** — final reported run. ~108 s for Sentence-T5 encoding, ~9 s for RQ-VAE, ~82 min for 30 Transformer epochs.
- **Local Apple-Silicon MPS** — sanity-check run + ablation iteration on the smaller pieces. Sentence-T5 encoding ~15 min, RQ-VAE ~100 s, Transformer too slow to use for the full 30-epoch schedule (~7 h).

Free Colab T4 was hit-or-miss with disconnects mid-session; Kaggle's 30-hour weekly GPU quota was a more reliable home. Kaggle's GPU-P100 slot rejects PyTorch 2.10 (sm_60 dropped), so stick with the T4 slot.

## Requirements

```bash
pip install -r requirements.txt
```

PyTorch ≥ 2, sentence-transformers, transformers, pandas/numpy, matplotlib, tqdm.

## How to run

```bash
# Stage 0 — preprocess + cache Sentence-T5 content embeddings (one-time)
python main.py --stage embed --tag embed

# Stage 1 — train RQ-VAE on cached embeddings, dump one Semantic ID per item
python main.py --stage rqvae --tag rqvae_default --rqvae_tag rqvae_default \
    --rq_levels 3 --rq_codebook_size 256 --rq_latent_dim 32

# Stage 2 — train the generative Transformer + final test eval (beam 10, full ranking)
python main.py --stage tiger --tag tiger_default --tiger_tag tiger_default \
    --t_hidden 384 --t_heads 6 --t_layers 4 --t_ffn 1024 --t_dropout 0.1 \
    --t_lr 1e-3 --t_batch_size 256 --t_epochs 30 --beam_size 10 \
    --eval_every 5 --eval_max_users 1500
```

Each stage writes a JSON to `results/<tag>.json`. The dataset is auto-downloaded into `data/` on first run, or you can drop the two `.json.gz` files (reviews_Toys_and_Games_5 + meta_Toys_and_Games) there yourself.

For Colab/Kaggle, the same commands run unchanged — `notebook.ipynb` wraps the clone + install + dataset download + the three stages. On Colab set Runtime → T4 first; on Kaggle Session options → Accelerator → GPU T4 ×2.

## Default config (paper)

- RQ-VAE: encoder MLP → 32-dim latent, `L=3` codebook levels of `K=256`, suffix token for collisions, VQ-VAE-style commitment loss with EMA codebook updates (the gradient-based variant collapses on Sentence-T5 outputs — see report §5).
- Transformer: 4 encoder + 4 decoder layers, 6 heads × 64 = model dim 384, FFN 1024, dropout 0.1.
- Input length: 20 history items × (L + 1 suffix) = 80 tokens. Beam size 10 at inference.

## Files

```
Assignment 3/
├── main.py                    # entry point — picks stage and runs it
├── run_experiments.py         # default + ablation grid
├── plot_results.py            # generates report/*.pdf figures from results/*.json
├── notebook.ipynb             # Colab/Kaggle-ready notebook
├── requirements.txt
├── README.md                  ← you are here
├── A3.text                    # raw assignment description from Brightspace
├── src/
│   ├── config.py              # paths + hyperparameters
│   ├── dataset.py             # Amazon Toys & Games loader, 5-core, sequences, split
│   ├── content.py             # Sentence-T5 content embeddings (cached)
│   ├── rqvae.py               # RQ-VAE: encoder + residual quantizer + decoder + EMA updates
│   ├── transformer.py         # encoder-decoder Transformer + beam search
│   ├── train.py               # training loops (RQ-VAE: denoising warmup + k-means init + EMA; Transformer: AdamW + warmup-then-inv-sqrt LR)
│   └── evaluate.py            # full-ranking Recall@{5,10} / NDCG@{5,10}
├── tests/
│   └── test_smoke.py          # quick correctness checks (RQ-VAE recon, beam search, NDCG math)
├── data/                      # raw dataset, embeddings cache, checkpoints (gitignored)
├── results/                   # per-run JSON outputs
└── report/
    ├── report.tex             # ACM sigconf, double-column
    └── report.pdf             # 3-page compiled report (tectonic)
```

## Notes / things that bit us

- **Don't joint-train.** RQ-VAE to convergence first, then freeze and extract Semantic IDs, then train the Transformer.
- **Vanilla VQ-VAE collapses on Sentence-T5 outputs.** Sentence-T5 embeddings have a strong global-mean direction (norm ~0.9) that dwarfs per-item variation (~0.4), so a constant encoder + decoder bias hits the variance floor immediately. The fix in `src/rqvae.py` is the combo of (i) input centring, (ii) denoising warmup with Gaussian noise on the centred input, (iii) k-means++ codebook init on warmed-up encoder outputs, (iv) EMA codebook updates with dead-code resetting. With this combo we get 7.7 % collision rate and 99 %+ codebook usage; without it we get a single codeword and 99.9 % collision.
- **Output buffering hides progress.** Long Python training jobs through `tee` or piped to a Colab/Kaggle console block-buffer their stdout. Prefix with `PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u` to get per-epoch lines flushed live. Our first 30-min Colab run silently completed nothing visible because of this.
- **Cache content embeddings.** Sentence-T5 over ~11 K items is ~3 min on T4, ~15 min on M1 MPS. The cached `.npy` is reused by every RQ-VAE/Transformer rerun.
- **Sanity check.** Paper-default on Toys & Games lands near NDCG@5 ≈ 0.03–0.04, Recall@5 ≈ 0.05 (full ranking). Our 30-epoch run with collision rate 7.7 % comes in at NDCG@5 = 0.0150, Recall@5 = 0.0210 — short of the paper but in the right neighbourhood and improving when we stopped (loss was still dropping at epoch 30).
