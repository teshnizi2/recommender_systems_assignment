# Assignment 3 — TIGER on Amazon Toys & Games

**MohammadReza AhmadiTeshnizi, Lara Ichli, Nithin Raju — Group 15, RS course 2025/26.**

We implement TIGER (Rajput et al., NeurIPS 2023) — a generative sequential recommender that represents each item as a tuple of discrete **Semantic IDs** and trains an encoder–decoder Transformer to autoregressively generate the Semantic ID of the next item, instead of scoring items in an embedding space the way SASRec (A2) did.

Three stages:
1. **Sentence-T5** content encoding — concatenate each item's title + categories + brand + description and run through `sentence-t5-base` to get a 768-dim content vector. Cached to disk.
2. **RQ-VAE** quantises each content vector into an `L`-tuple of codebook indices (the Semantic ID). Train it on item content, freeze, dump one Semantic ID per item.
3. **Encoder–decoder Transformer** takes the flattened Semantic ID tokens of a user's history and generates the next item's `L`-tuple token by token. Beam search at inference (beam = 20).

Dataset: **Amazon Toys & Games 5-core**. Implicit positives, iterative 5-core filtering, leave-one-out (last → test, second-to-last → val), max history 20 items. Eval is full-item ranking, Recall@{5,10} and NDCG@{5,10}.

## Results

| Metric | Default ($K{=}256$, $L{=}3$, $d{=}384$, beam 20) |
|---|---|
| Recall@5  | 0.0342 |
| NDCG@5    | 0.0245 |
| Recall@10 | 0.0488 |
| NDCG@10   | 0.0292 |

Inside the assignment's expected ballpark for Toys & Games (NDCG@5 ≈ 0.03–0.04). Full ablation table (codebook size $K$, levels $L$, Transformer width/depth, beam size) and figures are in `report/report.pdf`.

## Files

```
Assignment 3/
├── tiger_assignment.ipynb     # ← the submission code; runs end-to-end
├── report/
│   ├── report.tex             # ACM sigconf, 2-column
│   ├── report.pdf             # 4-page compiled report
│   └── figs/*.png             # training curves, codebook, ablation plot
├── results/                   # default test metrics + ablation CSV
├── requirements.txt
├── A3.text                    # raw assignment description from Brightspace
├── README.md                  ← you are here
└── _attic/                    # an earlier modular .py implementation we
                               # kept for reference; NOT part of the submission
```

## How to run

The notebook is designed for **Kaggle (T4 ×2 slot, one GPU used)** or **Colab T4**. Tested with PyTorch ≥ 2 and `sentence-transformers`.

### Kaggle (recommended — what we used for the reported numbers)

1. Sign in at <https://kaggle.com>, click *Create → New Notebook*.
2. Open `tiger_assignment.ipynb` (File → Upload notebook).
3. Right panel → **Session options → Accelerator → GPU T4 ×2** (P100 will fail with `cudaErrorNoKernelImageForDevice` because PyTorch 2.10 wheels dropped `sm_60`).
4. Add the **Amazon Toys and Games 5-core** dataset as a Kaggle Input (search the data hub), or upload the `Toys_and_Games_5.json` + `meta_Toys_and_Games.json` files manually into `tiger_assignment/data/` (the notebook expects them there).
5. *Runtime → Run all*. The default config takes about an hour; the full ablation suite (8 configs) takes ~50 GPU-hours and is gated by `*_done.flag` files so re-runs skip completed configs.

### Colab

The same notebook runs on Colab. Uncomment the Drive-mount block in cell 2 (`BASE_DIR = "/content/drive/MyDrive/tiger_assignment"`) and pre-create the matching folder structure on Drive so progress survives the ~12h session limit.

### Required dataset files

Download from <https://nijianmo.github.io/amazon/index.html> (Ni et al. 2019 dump) — specifically the 5-core **Toys and Games** subset:

- `Toys_and_Games_5.json` — user × item × timestamp
- `meta_Toys_and_Games.json` — title, categories, brand, description

Drop both files into `tiger_assignment/data/` next to the notebook.

## Default config (matches the TIGER paper)

- **RQ-VAE.** 768-dim Sentence-T5 → encoder MLP → 32-dim latent, $L{=}3$ codebook levels of $K{=}256$ entries, disambiguation suffix for collisions, VQ-VAE commitment loss ($\beta{=}0.25$) with EMA codebook updates.
- **Transformer.** 4 encoder + 4 decoder layers, 6 heads × 64 = model dim 384, FFN 1024, dropout 0.1.
- **Input.** 20 history items × (L+1 = 4 tokens per item) = 80 input tokens. AdamW optimiser, linear warmup + inverse-sqrt LR decay, cross-entropy. Beam size 20 at inference.

## Ablations included in the report

| Axis | Values | Held-fixed |
|---|---|---|
| Codebook size $K$ | 64, 128, **256** | $L{=}3$ |
| Levels $L$ | 2, **3**, 4 | $K{=}256$ |
| Transformer $(d, \text{heads}, \text{layers})$ | (256,4,4), **(384,6,4)**, (384,6,6) | — |
| Beam size | 10, **20** | default arch |

Bold = default config. See report Table 2 and Figure 3.

## Notes from the implementation

- **Stage the training.** Train the RQ-VAE to convergence first, freeze it, extract one Semantic ID per item, then train the Transformer on those IDs. Joint training does not work — the paper is explicit about this and the notebook follows the same pattern.
- **Keep the codebook alive.** Sentence-T5 outputs on this dataset are strongly anisotropic, which makes a gradient-based VQ-VAE prone to collapsing. The `VectorQuantizer` in the notebook uses EMA codebook updates and the VQ-VAE commitment loss ($\beta = 0.25$), with a small uniform codebook initialisation; this keeps all three levels live throughout training (≥ 97% utilisation by the time the loss plateaus).
- **Cache content embeddings.** Sentence-T5 over ~12K items is ~3 min on T4. The cached embeddings are reused by every RQ-VAE/Transformer rerun, which is what makes the 8-config ablation suite tractable.
- **Done-flags for restart safety.** Each ablation run drops a `*_done.flag` file in `tiger_assignment/checkpoints/` on completion. Re-running the notebook skips configs that already have a flag, so a Kaggle session disconnect mid-sweep is recoverable without re-running everything.

## What we'd do with another week

1. **Multiple seeds.** Some of the gaps in the ablation table are within plausible seed noise; an averaged-over-seeds version would let us claim which differences are significant.
2. **A cosine LR schedule.** We use linear warmup then inverse-square-root decay; a cosine schedule with a longer warmup is the more standard choice for sequence models of this size and might push Recall@5 closer to the paper's published numbers.
3. **Cold-start eval.** Hold out a fraction of items from training entirely and measure whether the Transformer can still recover them through Semantic-ID prefix similarity. This is the headline advantage of TIGER over embedding-based retrievers and we did not have time to evaluate it on this dataset.
