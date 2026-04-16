# Results index

One row per experiment. Keep this up to date as experiments complete.

## Phase 1 — Architecture & validation (months 1–3)

| ID | Title | Status | Headline result | Link |
|---|---|---|---|---|
| E01 | OFDM supervised receiver (LTE-5, EPA, CRS pilots) | complete | Learned RX beats LS+interp at SNR ≥ 5 dB; beats perfect-CSI ZF oracle at ≥ 20 dB (30× at 25 dB) | [results](02_experiments/E01_ofdm_supervised_rx/results.md) |
| E02 | Depth/heads/LayerNorm ablation | planned | — | — |
| E03 | Multi-head receiver (bits + channel MSE) | planned | — | — |
| E04 | Statistical tightening (500 subframes/SNR) | planned | — | — |
| E05 | Impairment generalization (CFO, IQ imb, PA, alt-TDL) | planned | — | — |
| E06 | Attention visualization + per-head role attribution | planned | — | — |

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
