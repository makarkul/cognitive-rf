# Results index

One row per experiment. Keep this up to date as experiments complete.

## Phase 1 — Architecture & validation (months 1–3)

| ID | Title | Status | Headline result | Link |
|---|---|---|---|---|
| E01 | OFDM supervised receiver (LTE-5, EPA, CRS pilots) | complete | Learned RX matches perfect-CSI ZF oracle within ±14% across 0–25 dB and beats LS+interp by 1.4–4× across the full range. At 25 dB: 1.23e-3 (learned) vs 1.78e-3 (oracle, 1.45×) vs 5.07e-3 (LS+interp, 4.1×). 500-subframe eval; original "30× below oracle" was Poisson noise on light sampling. | [results](02_experiments/E01_ofdm_supervised_rx/results.md) |
| E02 | Depth/heads/LayerNorm ablation | planned | — | — |
| E03 | Multi-head receiver (bits + channel MSE) | planned | — | — |
| E04 | Statistical tightening (500 subframes/SNR) | folded into E01 | Re-eval of E01 best.pt at 500 subframes/SNR; headline corrected. See E01 errata. | [E01 results](02_experiments/E01_ofdm_supervised_rx/results.md) |
| E05 | Impairment generalization (CFO, IQ imb, PA, alt-TDL) | planned | — | — |
| E06 | Interpretability probes on E01 (linear H-probe, pilot denoising, per-cell ablation, perturbation kernel) | scaffolded; probes not yet run | Hypotheses committed 2026-04-16 (R²≥0.95 at pilots by layer 2; model MSE 3–6 dB below LS; diagonal-attention ablation drives 25 dB BER back to ≥1.78e-3; freq kernel half-width ~6 SCs, flat in time). Attention-visualization + per-head attribution folded into Probe 3+4 outputs. | [hypothesis](02_experiments/E06_probes_on_e01/hypothesis.md) · [method](02_experiments/E06_probes_on_e01/method.md) |

## Phase 2 — Self-supervised pretraining (months 4–6)

| ID | Title | Status | Headline result | Link |
|---|---|---|---|---|
| E07 | Masked-RE modeling (MREM) pretraining | planned | — | — |
| E08 | Label efficiency probe (1%/10%/100% labels) | planned | — | — |
| E09 | Linear probes: H, SNR, LoS from frozen encoder | planned | — | — |
| E10 | Embedding geometry (intrinsic dim, t-SNE, clusters) | planned | — | — |

## Phase 3 — Blind / pilotless (months 7–9)

| ID | Title | Status | Headline result | Link |
|---|---|---|---|---|
| E11 | IQ-domain conv stem (raw IQ → token grid) | planned | — | — |
| E12 | BRF-WM Phase-1: blind CFO + coarse CIR | planned | — | — |
| E13 | Differentiable signal renderer | planned | — | — |
| E14 | Pilot density sweep (100% → 0%) | planned | — | — |

## Phase 4 — Standard-agnostic (months 10–12)

| ID | Title | Status | Headline result | Link |
|---|---|---|---|---|
| E15 | Spectrogram-tile ViT | planned | — | — |
| E16 | Multi-standard corpus (LTE + 802.11 or 5G NR) | planned | — | — |
| E17 | Modulation classification (held-out modulations) | planned | — | — |
| E18 | Cognitive adaptation demo (RL over pilot density) | planned | — | — |
