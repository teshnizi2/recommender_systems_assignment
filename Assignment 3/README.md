# Assignment 3 — TIGER on Amazon Toys & Games

**MohammadReza AhmadiTeshnizi, Lara Ichli, Nithin Raju — Group 15, RS course 2025/26.**

Repo: <https://github.com/teshnizi2/recommender_systems_assignment>

We implement TIGER (Rajput et al., NeurIPS 2023): a generative sequential recommender that represents each item as a tuple of discrete **Semantic IDs** and trains an encoder–decoder Transformer to autoregressively generate the next item's Semantic ID, instead of scoring items by dot-product in an embedding space the way SASRec does.

Three stages:

1. **Sentence-T5** content encoding. Concatenate each item's title + categories + brand + description, run through `sentence-t5-base` for a 768-dim content vector. Cached to disk.
2. **RQ-VAE** quantises each content vector into an `L`-tuple of codebook indices (the Semantic ID). Train it on item content, freeze, dump one Semantic ID per item.
3. **Encoder–decoder Transformer** consumes the flattened Semantic ID tokens of a user's history and generates the next item's `L`-tuple token by token. Beam search at inference (beam = 20).

Dataset: **Amazon Toys & Games 5-core**. Implicit positives, iterative 5-core filtering, leave-one-out (last → test, second-to-last → val), max history 20 items. Eval is full-item ranking, Recall@{5,10} and NDCG@{5,10}.

## Results

Default config ($K{=}256$, $L{=}3$, $d{=}384$, beam 20), single seed (PyTorch seed 42):

| Metric | Value |
|---|---|
| Recall@5  | 0.0342 |
| NDCG@5    | 0.0245 |
| Recall@10 | 0.0488 |
| NDCG@10   | 0.0292 |

These land **just below** the assignment's expected band (R@5 ≈ 0.05, NDCG@5 ≈ 0.03–0.04): R@5 about 30% below and NDCG@5 about 20% below the lower bound. The headline diagnostic finding is that our RQ-VAE **collapsed** on this dataset: only 1, 20, and 16 of the K=256 codebook entries are used at levels 0, 1, 2 (utilisation 0.4% / 7.8% / 6.3%), leaving 115 unique L-tuples for 11,924 items (99.9% collision). The disambiguation suffix carries the bulk of the item-identifying signal, yet only 2.38% of beam outputs are invalid. Full ablation table (codebook size $K$, levels $L$, Transformer width/depth, beam size) and figures are in `report/report.pdf`.

## Files

```
Assignment 3/
├── tiger_assignment.ipynb         # ← the submission code; runs end-to-end
├── report/
│   ├── report.tex                 # ACM sigconf, 2-column
│   ├── report.pdf                 # 4-page compiled report
│   └── figs/*.png                 # training curves, codebook usage, collisions, ablation plot
├── results/
│   ├── default_test_metrics.json  # default-config test metrics
│   └── ablation_results.csv       # all 8 ablation rows
├── data/                          # dataset goes here (gitignored; see below)
├── requirements.txt               # Python deps
├── A3.text                        # raw assignment description from Brightspace
├── README.md                      ← you are here
└── _attic/                        # earlier modular .py implementation; NOT part of the submission
```

## How to run

The notebook is designed for **Kaggle (T4 ×2 slot, one GPU used)** or **Colab T4**. Tested with PyTorch ≥ 2.0 and `sentence-transformers`. On Kaggle/Colab the notebook creates a runtime working directory called `tiger_assignment/` (separate from the repo layout above) — that is where datasets, embeddings, and checkpoints live during a run.

### Setup

1. Clone this repo (or just download the notebook):
   ```
   git clone https://github.com/teshnizi2/recommender_systems_assignment
   cd "recommender_systems_assignment/Assignment 3"
   ```
2. Install dependencies (locally or via the first cell on Kaggle/Colab):
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset (see *Required dataset files* below) and place the unzipped JSONs in `data/` (locally) or `tiger_assignment/data/` (on Kaggle/Colab).

### Kaggle (recommended — what we used for the reported numbers)

1. Sign in at kaggle.com, **Create → New Notebook**.
2. Upload `tiger_assignment.ipynb` (File → Upload notebook).
3. **Session options → Accelerator → GPU T4 ×2** (P100 will fail with `cudaErrorNoKernelImageForDevice` because recent PyTorch wheels dropped Pascal/`sm_60` support).
4. Add the **Amazon Toys and Games 5-core** dataset as a Kaggle Input, or manually upload `Toys_and_Games_5.json` and `meta_Toys_and_Games.json` (unzipped from the `.json.gz` originals) into `tiger_assignment/data/`.
5. **Runtime → Run all**. The default config takes roughly 4–6 GPU-hours end-to-end; the full ablation suite (8 configs) takes ~50 GPU-hours and is gated by `*_done.flag` files in `tiger_assignment/checkpoints/` so re-runs skip completed configs.

### Colab

The same notebook runs on Colab. Uncomment the Drive-mount block in cell 2 (`BASE_DIR = "/content/drive/MyDrive/tiger_assignment"`) and pre-create the matching folder structure (`tiger_assignment/{data,embeddings,checkpoints,plots}/`) on Drive so progress survives the ~12h session limit.

### Required dataset files

Download from <https://nijianmo.github.io/amazon/index.html> — specifically the 5-core **Toys and Games** subset (McAuley et al., *SIGIR 2015*):

- `Toys_and_Games_5.json.gz` — user × item × timestamp (`gunzip` first)
- `meta_Toys_and_Games.json.gz` — title, categories, brand, description (`gunzip` first)

Drop both unzipped files into `data/` (or `tiger_assignment/data/` on Kaggle/Colab).

## Default config

- **RQ-VAE.** 768-dim Sentence-T5 → encoder MLP → 32-dim latent, $L{=}3$ codebook levels of $K{=}256$ entries, disambiguation suffix for collisions, VQ-VAE commitment loss ($\beta{=}0.25$) with EMA codebook updates. AdamW, LR $10^{-3}$, batch 256, up to 200 epochs with early-stopping patience 20.
- **Transformer.** 4 encoder + 4 decoder layers, 6 heads × 64 = model dim 384, FFN 1024, dropout 0.1. AdamW, LR $10^{-4}$, weight decay $10^{-2}$, batch 128, up to 100 epochs with early-stopping patience 2. Linear warmup over the first 10% of steps, then inverse-sqrt LR decay.
- **Input.** 20 history items × (L+1 = 4 tokens per item) = 80 input tokens. Cross-entropy loss with PAD ignored. Beam size 20 at test (beam 5 during validation for speed).

## Ablations included in the report

| Axis | Values | Held-fixed |
|---|---|---|
| Codebook size $K$ | 64, 128, **256** | $L{=}3$ |
| Levels $L$ | 2, **3**, 4 | $K{=}256$ |
| Transformer $(d, \text{heads}, \text{layers})$ | (256,4,4), **(384,6,4)**, (384,6,6) | — |
| Beam size | 10, **20** | default arch |

Bold = default config. See report Table 2 and Figure 3.

## Notes from the implementation

- **Stage the training.** Train the RQ-VAE to convergence first, freeze it, extract one Semantic ID per item, then train the Transformer on those IDs. Joint training does not work; the paper is explicit about this and the notebook follows the same pattern.
- **The codebook collapsed.** We applied the standard mitigations (EMA updates, small uniform init, dead-code resets, denoising warmup) but the RQ-VAE still narrowed to 1 / 20 / 16 live entries at levels 0 / 1 / 2. Sentence-T5's embedding cloud is too narrow on a catalogue this size for vanilla quantisation; the heavier fixes (k-means init, swapping to a less-narrow encoder like E5) didn't fit the 50 GPU-hour budget. See `report/report.pdf` §4.2 for the full diagnostic.
- **Cache content embeddings.** Sentence-T5 over ~12K items is ~3 min on T4. The cached embeddings are reused by every RQ-VAE / Transformer rerun, which is what makes the 8-config ablation suite tractable.
- **Done-flags for restart safety.** Each ablation run drops a `*_done.flag` file in `tiger_assignment/checkpoints/` on completion. Re-running the notebook skips configs that already have a flag, so a mid-sweep Kaggle disconnect is recoverable without re-running everything.
- **Single seed.** All reported numbers use PyTorch seed 42. Some ablation gaps may be within single-seed noise, especially the closely-matched Transformer-architecture rows.

## What we'd do with another week

1. **Fix the codebook collapse.** The single change most likely to lift Recall@5 toward the upper 0.04s reported by the paper. The minimal viable next experiment is k-means initialisation on a sub-sampled batch with cosine-distance quantisation; the larger lever is replacing Sentence-T5 with a less anisotropic encoder (e.g. E5 or a contrastive-pretrained variant).
2. **Multiple seeds.** Some of the gaps in the ablation table are within plausible seed noise; an averaged-over-seeds version would let us claim which differences are significant.
3. **Cosine LR schedule** with a longer warmup, the more standard choice for sequence models of this size; might push Recall@5 closer to the assignment's expected band.
4. **Cold-start eval.** Hold out a fraction of items from training entirely and measure whether the Transformer can still recover them through Semantic-ID prefix similarity — the headline advantage of TIGER over embedding-based retrievers, which we did not have budget to evaluate.
