# Results — E00: Sinusoid recovery with a tiny transformer

## Headline

A 29 K-parameter transformer trained on next-sample regression over noisy
sinusoids implicitly discovers AR(2) structure (attention peaks at lags 1
and 2) and encodes the sinusoid frequency in its hidden state (linear-probe
R² > 0.9). Under i.i.d. zero-mean noise, supervised denoiser training and
strict LM-style next-sample prediction are empirically indistinguishable;
under structured noise (DC offset, Wiener phase walk, AR(1) coloured) the
LM regime degrades in proportion to how predictable the noise is.

The full numerical writeup lives in
[`experiments/E00_sinusoid_recovery/RESULTS.md`](../../../experiments/E00_sinusoid_recovery/RESULTS.md);
this file is the notebook-side index.

## Numbers

### Original E00 — AR(2) signature + frequency probe

- Attention lag profile: clear peaks at lag 1 and lag 2 across test
  frequencies (3, 5, 10, 15 Hz). Random-init control: flat.
- Frequency probe R² > 0.9 at layer 2 (linear ridge, 1500 train / 400
  test). Random-init control: ≈ 0.
- Denoising at default noise (`σ_A = 0.2, σ_φ = 0.1`):
  +11.8 dB SNR gain over naive persistence baseline.

See [`experiments/E00_sinusoid_recovery/RESULTS.md`](../../../experiments/E00_sinusoid_recovery/RESULTS.md)
sections 1–5 for the SNR sweep, the FFT-embedding contrast, and the
classical-DSP perspective.

### W18 2026 side experiment — LM-style vs denoiser, four noise modes

`python train_lm_vs_denoiser.py --epochs 25 --train-size 5000 --val-size 1000 --noise-mode sweep`

Headline scalars from the 25-epoch sweep (`seed=42`):

| Noise mode | Δ MSE-vs-clean (LM − den) | Δ SNR gain (dB) | Drift RMSE | LM layer-2 R² | Verdict |
|---|---:|---:|---:|---:|---|
| `iid`                | −0.003 | **+0.26** | 0.063 | 0.969 | equivalence holds |
| `ar1_coloured` (α=0.9) | +0.001 | **−0.12** | 0.090 | 0.938 | small LM penalty |
| `wiener_phase`       | +0.072 | **−0.70** | 0.318 | 0.980 | clean LM penalty (both modes hard) |
| `dc_offset` (σ_μ=0.5) | +0.240 | **−6.91** | 0.483 | 0.967 | equivalence broken |

The drift-RMSE ordering matches the prediction exactly:
`iid (0.063) < ar1 (0.090) < wiener_phase (0.318) < dc_offset (0.483)`.
The `dc_offset` Δ SNR-gain of −6.9 dB is the headline of the sweep —
when the noise has a deterministic component, the LM's Bayes-optimal
output literally contains that component as bias, and the denoiser
gets to subtract it.

Figures:

- [`lm_vs_denoiser/noise_mode_sweep.pdf`](../../../experiments/E00_sinusoid_recovery/lm_vs_denoiser/noise_mode_sweep.pdf)
  — three-panel cross-mode summary.
- Per-mode figures under
  `experiments/E00_sinusoid_recovery/lm_vs_denoiser/<mode>/`.

## What the numbers say

### AR-discovery + frequency representation are robust

Both pre-program findings hold up to re-evaluation. The architecture is not
doing pattern-matching at the input layer — the attention lag profile is a
post-training emergent feature (random init shows flat lag profiles), and
the frequency representation is linearly readable starting at layer 2. This
is why E00 was the right warm-up for E01 and onward.

### LM-style ≡ denoiser only under i.i.d. zero-mean noise

The mathematical argument:

```
E[noisy[n+1] | history] = E[clean[n+1] | history] + E[ε[n+1] | history]
```

The two regimes share an optimum iff the second term is zero. AWGN
satisfies this; structured impairments do not. The empirical sweep
confirms the prediction across four noise modes — see the discussion file
for the full story:

[`01_discussions/009_lm_vs_denoiser_equivalence.md`](../../01_discussions/009_lm_vs_denoiser_equivalence.md)

## Did the hypothesis hold?

- **(1) AR(2) attention signature.** Confirmed — peaks at lag 1 and 2,
  consistent across test frequencies.
- **(2) Linear frequency probe.** Confirmed — R² > 0.9 at layer 2 vs
  ≈ 0 at random init.
- **(3) Equivalence holds under i.i.d. and breaks under structured
  noise.** Confirmed by the 25-epoch sweep. Drift-RMSE ordering
  matches the prediction (0.063 → 0.090 → 0.318 → 0.483), and
  Δ SNR-gain matches the predicted *magnitude* hierarchy: small for
  `iid` and `ar1_coloured` (within seed-noise), larger for
  `wiener_phase` (−0.70 dB), catastrophic for `dc_offset` (−6.91 dB).
  One sub-result was unexpected — under `wiener_phase` the LM has a
  *higher* freq-probe R² than the denoiser; explained in §7d as the
  two regimes solving different effective tasks under a non-stationary
  signal model.

## Caveats

- Single seed (42), single architecture, single noise-magnitude setting
  per mode. The hierarchy across modes is robust at the qualitative
  level; the specific deltas in the table are not error-barred yet.
- The LM regime's irreducible loss floor is `≈ Var(ε)`, not zero. We
  evaluate it on `MSE-vs-clean` post-training (which it never optimised
  for directly) — that is the fair comparison for *denoising quality*,
  not the per-task loss surface.
- The side-experiment dataset is the same paired generator for both
  regimes (deterministic per index). Different RNG seeds for the two
  regimes might shift individual numbers but should not change the
  hierarchy.

## Follow-ups

- **At OFDM-grid scale (E07 prep):** repeat the LM-vs-supervised gap
  measurement with masked-RE pretraining instead of next-sample
  prediction, on the LTE-5 EPA distribution. If the hierarchy
  carries over, the masked-RE recipe has a known failure mode under
  structured impairments and we plan E07's auxiliary supervision
  accordingly. Tracked as Q19 in
  [`04_open_questions.md`](../../04_open_questions.md).
- **Multi-seed error bars** on the noise-mode sweep — cheap (~20 min
  CPU per extra seed), would let us state the hierarchy with a
  confidence interval rather than as a single ordering.
- **Magnitude sweeps within a mode** (e.g. AR(1) `α ∈ {0.0, 0.5,
  0.9}`, DC offset `σ_μ ∈ {0.0, 0.25, 0.5}`) — would map the *shape*
  of the equivalence boundary, not just confirm it exists.

## Artifacts

- **Original training:**
  `experiments/E00_sinusoid_recovery/ts_transformer.pth` (committed).
- **Original analysis figures:** `experiments/E00_sinusoid_recovery/{
  attention_analysis.pdf, attention_lag_profile.pdf, denoising_*Hz.pdf,
  autoreg_*Hz.pdf, probes_summary.pdf, training_results.pdf}`.
- **Side experiment (W18 2026):**
  `experiments/E00_sinusoid_recovery/lm_vs_denoiser/{<mode>/{denoiser.pth,
  lm.pth, loss_curves.pdf, attention_lag_profile.pdf, freq_probe.pdf,
  summary.json}, noise_mode_sweep.pdf, sweep_summary.json}`.
- W&B: not tracked (deterministic CPU runs, single-seed).
