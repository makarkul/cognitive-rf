# Method — E00: Sinusoid recovery with a tiny transformer

## Code

- Folder: `experiments/E00_sinusoid_recovery/`
- Original training entry point: `python train.py --epochs 30`
- Original analysis entry point: `python analyze.py --test-freqs 2.0 5.0 10.0 18.0`
- Frequency probe (probes part of the original E00 story):
  `python probes.py --ckpt ts_transformer.pth`
- LM-vs-denoiser side experiment (added W18 2026):
  `python train_lm_vs_denoiser.py --epochs 25 --train-size 5000 --val-size 1000 --noise-mode sweep`
- All four-mode sweep artifacts land under
  `experiments/E00_sinusoid_recovery/lm_vs_denoiser/{iid, ar1_coloured,
  wiener_phase, dc_offset}/` plus a top-level
  `noise_mode_sweep.pdf` and `sweep_summary.json`.

## Data

- Source: synthetic, generated on the fly per `__getitem__`.
- Distribution per sample:
  - `freq ~ U(1, 20) Hz`
  - `amplitude ~ U(0.5, 2.0)`
  - `phase ~ U(0, 2π)`
- Sampling rate: `fs = 100 Hz`. Context length = 128 → 1.28 s per sequence.
- Noise (canonical `train.py` and `iid` mode of the side experiment):
  - amplitude noise `ε_A ~ N(0, σ_A²)`, default `σ_A = 0.2`
  - phase noise `ε_φ ~ N(0, σ_φ²)`, default `σ_φ = 0.1`
- Side-experiment noise modes (`train_lm_vs_denoiser.py --noise-mode ...`):
  - `iid` — as above.
  - `ar1_coloured` — amplitude noise replaced by AR(1) coloured process
    `ε[n] = α·ε[n-1] + √(1-α²)·w[n]`, default `α = 0.9`. Phase noise i.i.d.
  - `wiener_phase` — phase noise integrated:
    `θ[n] = θ[n-1] + ν[n]`, `ν ~ N(0, σ_φ²)`. Amplitude i.i.d.
  - `dc_offset` — per-sequence DC bias `μ ~ N(0, σ_μ²)`, default
    `σ_μ = 0.5`. Amplitude + phase i.i.d.
- Train / val split: deterministic per-index RNG so the LM and denoiser
  regimes see byte-identical streams.

## Model

- Architecture: `TimeSeriesTransformer` — scalar-input GPT variant.
  Causal multi-head self-attention.
- Hyperparameters (the `TS_TRANSFORMER_CONFIG` constant):
  - `context_length = 128`
  - `emb_dim = 32`
  - `n_heads = 2`
  - `n_layers = 2`
  - `drop_rate = 0.05`
  - `qkv_bias = False`
- Total trainable parameters: ~29 K.
- Input projection: `Linear(1, 32)` over each scalar sample (no
  tokeniser); learned positional embedding `Embedding(128, 32)`.
- Output head: `Linear(32, 1)`, scalar regression at every position.

## Training

- Optimizer: AdamW, `lr = 1e-3`, `weight_decay = 0.01`.
- Loss: MSE at every position (parallel-target).
  - **Denoiser regime:** target = `clean[1..N]` (next clean sample).
  - **LM-style regime:** target = `noisy[1..N]` (next noisy sample).
- Original `train.py`: 30 epochs × 10000 samples. ~5 min CPU.
- Side-experiment (`train_lm_vs_denoiser.py`): 25 epochs × 5000 samples,
  per regime, per noise mode. ~6 min CPU per regime; ~25 min CPU for
  the full four-mode sweep.

## Evaluation

Five separate diagnostics:

1. **Validation MSE-vs-clean and MSE-vs-noisy** at every epoch (both
   regimes, both targets — so the LM model is also scored against
   clean even though it never trained on it).
2. **SNR gain over a naive persistence predictor** that uses
   `noisy[t]` as the prediction for `clean[t+1]`. Reported in dB.
3. **Attention lag profile**, last layer, averaged over heads, on a
   clean test sinusoid at four test frequencies (3, 5, 10, 15 Hz).
   AR(2) signature = peaks at lag 1 and lag 2.
4. **Linear ridge frequency probe** on layer-1 and layer-2
   mean-pooled hidden states, fit on 1500 random sinusoids and
   evaluated on 400 held-out ones. R² + RMSE per layer.
5. **Cross-regime drift** (side experiment only): RMS and mean of
   `(LM_pred − denoiser_pred)` on the validation set. Direct measure
   of how much the LM regime's output diverges from the denoiser's.

Baselines:

- **Naive persistence predictor** (above) for SNR gain.
- **Random-init control** for the frequency probe — same architecture,
  no training, R² should sit near 0.
- The LM regime acts as its own baseline for the denoiser, and
  vice versa, in the side experiment.

Number of seeds: **1** (seed = 42 throughout). The hierarchy across
noise modes is not seed-sensitive at the qualitative level, but the
specific deltas in the side-experiment table are not yet error-barred.

## Hardware

Laptop CPU. No GPU required.

## W&B run ID

Not tracked. Original E00 predates W&B adoption; the side experiment is
deterministic from `seed=42` and produces a single `summary.json` per
mode + a top-level `sweep_summary.json`.
