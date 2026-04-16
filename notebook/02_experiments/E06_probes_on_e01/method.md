# Method — E06: Interpretability probes on the E01 learned receiver

Four independent probes, each in its own script. All probes load the
**same frozen E01 checkpoint** (`best.pt` at step 26000, val BER
4.33e-2); the model is never fine-tuned here — we only read out its
internal state.

## Code

- Folder: `experiments/E06_probes_on_e01/`
- Reproducer (all four): `python run_all_probes.py --ckpt <path>`
- Per-probe entry points:
  - `probe_01_H_linear.py` — linear probe for H[t, k] from residual
    stream, per layer.
  - `probe_02_pilot_output_vs_ls.py` — model H-estimate MSE vs raw
    LS MSE at pilot cells, per SNR.
  - `probe_03_per_cell_ablation.py` — re-run BER sweep with
    attention masked to diagonal.
  - `probe_04_perturbation_kernel.py` — Dirac-perturbation response
    in freq and time.
- Each script writes its own figure(s) to `figures/` and appends a
  block to `results.md` on completion.
- All probes share a small `_common.py` for checkpoint loading,
  hidden-state extraction, and batch generation.

## Checkpoint + data

- **Checkpoint**: `experiments/E01_ofdm_supervised_rx/checkpoints/best.pt`
  (also on HF Hub `makarkul/cognitive-rf-E01`). Model placed in
  `.eval()` mode with gradients disabled throughout.
- **Data source**: `dataset.generate_batch(...)` — same on-the-fly
  synthesis used for E01 training. We regenerate fresh subframes
  per probe (no saved dataset).
- **Known ground truth**: because data is synthetic, every subframe
  ships with:
  - `H_true[t, k]` — true complex channel at every RE.
  - `tx_bits[t, k]` — the bits actually transmitted.
  - `x_tx[t, k]` — the QPSK symbol before channel + noise.
  - `is_pilot[t, k]` — pilot mask.

## Probe-specific protocols

### Probe 1 — Linear probe for H[k]

- **Samples**: 500 subframes at each of three SNRs: 5, 15, 25 dB.
- **Feature extraction**: register forward hooks on the residual
  stream after every transformer block (4 hooks total, plus layer 0
  = post-embedding). Capture `hidden[B, 14, 300, 128]`.
- **Target**: `H_true` as two real channels `(Re(H), Im(H))`.
- **Probe model**: `nn.Linear(128, 2)` per-layer, per-position-type
  (pilot vs data), fit with ridge regression (λ=1e-3) on an 80/20
  train/test split of the 500 subframes.
- **Metric**: R² on the 20 %-held-out subframes, reported for
  `H_real` and `H_imag` separately, averaged.
- **Output**: `figures/probe_01_R2_per_layer.pdf` — R² vs layer,
  one line per SNR × (pilot/data).

### Probe 2 — Pilot-cell output denoising

- **Samples**: 500 subframes at each of four SNRs: 5, 10, 15, 25 dB.
- **LS estimate**: at pilot cells, compute `Ĥ_LS = y_pilot / x_pilot`.
  Measure `MSE(Ĥ_LS, H_true)`.
- **Model estimate**: at pilot cells, apply the layer-3 linear
  probe head from Probe 1 to `hidden[pilot cells]`. This gives a
  model-internal `Ĥ_model`. Measure `MSE(Ĥ_model, H_true)`.
- **Metric**: `10 · log10(MSE_LS / MSE_model)` — the denoising gain
  in dB. Report per SNR.
- **Output**: `figures/probe_02_pilot_denoising_gain.pdf` —
  denoising gain (dB) vs SNR.

### Probe 3 — Per-cell ablation

- **Samples**: 500 subframes per SNR, 11 SNR points (same as the
  500-subframe E01 eval) → 5,500 subframes total.
- **Implementation**: monkey-patch `nn.MultiheadAttention` to use a
  **diagonal mask** (only self-attention). Attention to all other
  positions is set to `-inf` pre-softmax.
- **Sanity check first**: verify that the unablated model on this
  same probe harness reproduces the published E01 BER curve (with
  ~1 % tolerance); only then enable the ablation.
- **Metric**: BER vs SNR under ablation vs un-ablated (the existing
  E01 numbers).
- **Output**: `figures/probe_03_per_cell_ablation_ber.pdf` —
  overlaid BER curves (learned unablated, learned ablated, oracle,
  LS+interp).

### Probe 4 — Perturbation kernel

- **Clean subframe**: one representative subframe at SNR = 20 dB.
- **Perturbation**: at one reference RE `(t₀, k₀) = (7, 150)` (middle
  of the grid), add `δ = (0.1 + 0.1j)` to `y[t₀, k₀]`.
- **Measurement**: compute the model's output difference
  `Δŷ[t, k] = model(y + δ_RE)[t, k] - model(y)[t, k]` at every RE.
- **Plot 1 — frequency kernel**: `|Δŷ[t₀, k₀ + d]|` vs `d ∈ [-50, 50]`.
- **Plot 2 — time kernel**: `|Δŷ[t₀ + τ, k₀]|` vs `τ ∈ [-7, 7]`.
- **Plot 3 — 2D heatmap**: `|Δŷ[t, k]|` as a 14×300 image,
  color-mapped.
- **Repeat** for 5 different `(t₀, k₀)` positions (center, edges,
  near-pilot, far-from-pilot) and average the kernels; report the
  average + per-position overlay.
- **Output**: `figures/probe_04_perturbation_*.pdf`.

## Evaluation discipline

- **Hypotheses are committed before any probe runs.** Git log should
  show `hypothesis.md` committed strictly before `results.md`
  entries.
- **Each probe commits independently.** After Probe 1 finishes,
  commit `results.md` and the Probe-1 figure; then start Probe 2.
  Prevents combining insights post-hoc.
- **Negative results count.** If a prediction is refuted, say so in
  `results.md` and preserve the original hypothesis text.

## Hardware

- Laptop CPU (Windows, no GPU).
- PyTorch model inference on 500 subframes at 834k params: ~15
  seconds per SNR point. All probes finish in under 2 hours
  wall-clock combined.

## Dependencies

- `torch >= 2.1`, `numpy`, `matplotlib`, `scipy` (ridge regression).
- Existing: `experiments/E01_ofdm_supervised_rx/{model.py, dataset.py,
  baseline_receiver.py, channel.py, pilots.py}`.
- Checkpoint: `experiments/E01_ofdm_supervised_rx/checkpoints/best.pt`.

## W&B run ID

Not tracked — probes are deterministic read-outs, not training runs.
Per-probe console logs captured in `experiments/E06_probes_on_e01/logs/`.
