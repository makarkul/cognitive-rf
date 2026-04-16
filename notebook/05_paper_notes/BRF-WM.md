# BRF-WM — Blind RF World Model

**Citation:** Concept deck (Notebook LM generated, 12 slides). Not a peer-reviewed paper. Treated as a research direction statement.

## Core claim (proposed, not demonstrated)

A self-supervised, pilot-less encoder trained on raw IQ disentangles a *slow-changing propagation state* (CIR, CFO, STO, Doppler, phase noise) from a *fast-changing content state* (symbols/payload), bootstrapping from cyclic-prefix redundancy, convolutional memory, and CFO phase rotation signatures in the IQ stream.

## Architecture (proposed)

- Encoder maps raw IQ windows to a latent sequence.
- State splitter into:
  - **Content branch** (fast): symbols/payload.
  - **Propagation branch** (slow): CIR taps, CFO, STO, Doppler.
- Propagation head extracts tap delays / CFO / STO.
- Differentiable signal renderer reconstructs the received IQ from both states (physics-aware loss).

## Training regimes (three proposed)

- **(A) Simulator:** fully self-supervised masked reconstruction + future-IQ prediction on synthetic waveforms.
- **(B) Teacher:** self-distillation from a classical blind estimator providing pseudo-labels for coarse CFO / delay spread.
- **(C) Generator:** generative identification with a differentiable channel layer; IQ reconstruction error supervises absolute physical accuracy.

## Phased roadmap

- **Phase 1:** Narrow identifiability sandbox — SISO OFDM, fixed FFT/CP, QPSK only, bounded CFO. Success = recover CFO and coarse CIR better than classical blind baselines.
- **Phase 2:** Explicit latent disentanglement under changing payloads/channels/Doppler/modulation.
- **Phase 3:** Blind equalization — use the latent to equalize without pilots; target: beat blind-classical EVM/BER, stay competitive with pilot-aided at heavily reduced pilot density.

## What's reusable for our program

This is **almost exactly our target operating regime** — SISO OFDM with EPA, fixed FFT/CP, a limited modulation set, and a pilot mask we already control.

### E11 — IQ-domain conv stem
Add a front-end conv to map raw IQ samples → token grid. ~50k extra params. Prerequisite for pilotless.

### E12 — BRF-WM Phase-1 sandbox
- Inject bounded CFO (e.g., ±200 Hz at 15 kHz SCS) at training.
- Disable pilot mask at inference.
- Two heads: regress `(CFO_hat, coarse delay spread)`.
- Baseline: classical CP-autocorrelation CFO estimator; LS delay spread from pilots.
- **Success criterion: CFO MAE ≤ classical blind estimator at SNR 0/5/10 dB.**

### E13 — Differentiable signal renderer
Add a physics-aware reconstruction loss alongside supervised CFO/CIR regression. Compare training stability and final accuracy vs E12.

### E14 — Pilot density sweep
Measure BLER as pilot density drops from 100% → 50% → 25% → 0%. Establishes the axis on which a learned pilotless RX needs to win.

## Caveats

- No reported numbers, no implementation details. Purely directional.
- Phase 1 is within our infrastructure's reach; phase 2–3 require additional engineering (masked SSL at scale, disentanglement probes).
- The "differentiable signal renderer" is the most novel piece and also the most complex to build — budget accordingly.

## Why this is attractive despite being a concept deck

Phase 1 is concrete, falsifiable, and exactly aligned with our infrastructure. The success criterion (beat CP-autocorrelation CFO) is well-defined and low-risk — classical blind estimators have decades of characterization, so we know what "beat" means. A negative result would also be informative.
