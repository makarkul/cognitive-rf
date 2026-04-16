# Hypothesis — E01: OFDM supervised receiver on LTE-5 EPA

*Written retroactively for the first experiment; future experiments will have hypothesis.md before running.*

## Question

Can an attention-based receiver trained end-to-end on a time-frequency resource grid match or beat a classical pilot-aided receiver (LS + linear interpolation + zero-forcing equalization) on an LTE-5 MHz FDD downlink with 3GPP EPA multipath + AWGN?

## Prediction

Yes. Specifically:
- Match LS+interp at low SNR (both estimator-limited).
- Beat LS+interp at mid SNR by roughly the estimator-residual factor (~2×).
- Approach but not reach the perfect-CSI ZF oracle, because the oracle has `H` for free.

A 834k-param, 4-layer, 4-head transformer with `d_model=128` should be enough. Parameter count follows the sinusoid recipe: `head_dim ≈ 2·log₂(4200) ≈ 24` → round up to 32; `n_heads = 4` (one per natural relational pattern: freq pilot lookup, time pilot lookup, global pilot average, local data neighborhood); `d_model = n_heads × head_dim = 128`; `d_ff = 4 × d_model = 512`; `n_layers = 4` (one each for pilot LS estimation, freq smoothing, time pooling, symbol decision).

## Why

- On-grid attention lets every RE see every other RE, so the model can pool channel evidence across 200 pilots instead of using the 2–4 nearest like LS+linear-interp does.
- BCE on data bits provides clean, dense supervision: 4000 scored bits per subframe.
- Pilot mask as a third input channel gives the model the information it needs to distinguish reference from data without spelling out the operation.
- Factorized 2D positional embedding (14-way + 300-way) gives grid-aware indexing at 40k params instead of 537k dense.

## Would change our plan if

- If result is **worse than LS+interp**: the architecture is fundamentally wrong; re-examine input representation, loss, or pilot conditioning before scaling.
- If result **matches LS+interp but doesn't beat**: capacity-limited; try d_model=192 or n_layers=6.
- If result **beats LS+interp but flat vs oracle**: the model is learning estimation but not exploiting grid structure; the follow-up is an attention visualization.
- If result **beats the oracle**: deeply interesting — the model is doing something beyond per-cell ZF. Follow-up: interpret (phase 2).

## Dependencies

- LTE-5 signal generator (`ofdm_signal.py`).
- EPA channel model (`channel.py`).
- CRS-style pilot scheme (`pilots.py`).
- Classical baselines for comparison (`baseline_receiver.py`).
- GPU for training (RunPod 4090 used).

## Estimated cost

- Compute: ~3.5 GPU-hours on a 4090 for 30k steps (~$1.50 at RunPod rates).
- Engineer time: ~1 week from scaffold to published `results.md`.
