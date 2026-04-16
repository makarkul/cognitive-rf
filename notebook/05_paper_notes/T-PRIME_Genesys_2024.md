# T-PRIME — Transformer-based Protocol Identification for ML at the Edge

**Citation:** Northeastern GENESYS Lab. INFOCOM 2024, arXiv 2401.04837.
**Code:** https://github.com/genesys-neu/t-prime

## Core claim

A transformer fed **raw IQ** (no preamble correlation, no handcrafted features) can identify WiFi protocols at ~98% accuracy in simulation and 97% over-the-air for single protocols, 75% under protocol overlap.

## Input representation

- **Raw IQ samples, directly into transformer.** No preamble correlation; no FFT; no classical feature extraction.
- Framed as a sequence-classification task.

## Model architecture

- Transformer encoder + classification head.
- Optimized for **edge deployment** — low-latency, small-footprint. Exact size in the paper.

## Training task

- Supervised multi-class classification over WiFi protocol variants.

## Training data

- **66 GB over-the-air WiFi corpus.** Real captures, not synthetic.
- Plus simulated augmentations.

## Downstream tasks / headline numbers

- >98% protocol classification in simulation.
- 97% single-protocol over-the-air.
- 75% double-protocol (overlap) over-the-air.

## What's reusable for our program

Two specific uses:

### 1. Real OTA WiFi dataset
The 66 GB corpus + code repository is a **ready-made real-world dataset** for when our program needs real captures rather than synthetic EPA. Use cases:
- Phase 4 (E15+) — validate that the spectrogram-tile ViT transfers from synthetic LTE to real WiFi.
- Phase 3 sim-to-real probe — fine-tune our pilotless encoder on a few hours of real IQ and measure degradation.

### 2. Edge-deployment reference
T-PRIME is one of the few transformer-for-RF papers that actually cares about **inference latency and deployment**. If we eventually want our receiver to run on a real baseband processor, T-PRIME's architecture choices (truncation, quantization, specific head/layer counts) are the right reference.

## Differences vs our setup

- T-PRIME is a *classifier* (what protocol is this?), not a *receiver* (give me bits). Different task.
- T-PRIME input is a raw IQ snippet with unknown content. We work with structured OFDM subframes. Our Phase 4 moves closer to T-PRIME's problem but not identical.
- Open source code — that's rare in this literature. Worth considering forking as a starting point for E15.

## URL

- https://arxiv.org/abs/2401.04837
- https://github.com/genesys-neu/t-prime
