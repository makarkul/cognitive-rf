# SoftBank Unified Transformer for Wireless Signal Processing

**Citation:** Kawai, Koodli. arXiv 2508.17960v1. SoftBank Research Institute of Advanced Technology, Tokyo. August 2025.

## Core claim

A single compact resource-element-level transformer backbone, adapted only via a task-specific output head + loss, can serve as an end-to-end receiver, channel-frequency interpolator, or channel estimator — and beats LS+linear, LS+nearest, and the Sionna CNN receiver while meeting sub-ms latency in a real 3GPP OAI+Aerial stack.

## Input representation

- Resource elements as tokens.
- Input tensor: `(subcarriers × OFDM symbols × feature_dim)`.
- End-to-end receiver tile: **12 SC × 14 OFDM symbols** (one RB × one slot) — identical numerology to our LTE-5.
- Real/imag concatenated as per-RE feature. Antenna dimension optionally folded in.

## Model architecture

- Reshape → per-RE Dense → learned positional embedding → 1–4 encoder layers (MHSA + MLP with pre-norm, 1–4 heads) → LayerNorm → MLP → Dense → reshape.
- Default: 4 layers × 4 heads. OTA interpolation task: **1 layer × 1 head**.
- **Notably omits early LayerNorm** to preserve signal magnitude. Explicit deviation from BERT/ViT.

## Training task

Fully supervised:
- (a) End-to-end receiver: per-bit BCE on LLRs.
- (b) Frequency interpolation: MSE on complex channel coefficients (2-channel real/imag regression).
- (c) Channel estimation: MSE on full (12×14×2) channel surface from received y.

## Training data

- Sionna + 3GPP CDL-C at 3.5 GHz, 30 kHz SCS.
- Eb/N0 sweep 0–40 dB.
- 100k samples for E2E, 220k for OTA interpolation.
- OTA interpolation task uses hybrid pipeline: Sionna regenerates `x` from captured transport block (via FAPI logs), `ĥ = y/x` from fronthaul I/Q captures, AWGN augmentation.

## Downstream / evaluation

Three use cases, evaluated in Sionna simulation (BLER vs Eb/N0) and real OTA chamber with programmable-jammer interference on an OAI+Aerial stack (NVIDIA GH200, TensorRT).

## Headline numbers

- 1UE SIMO CDL-C: transformer matches Perfect CSI within ~0.5 dB at BLER=1e-3, beats Sionna CNN receiver.
- 2UE MU-MIMO: 2 layers saturate to near-Perfect CSI; 1-head matches 8-head.
- OTA: 1L1H transformer beats in-house 8-layer ResNet CNN in throughput across interference sweep (-22 to -20.1 dB).
- **1.36× lower PUSCH pipeline latency** (337.7 µs vs 458.8 µs).

## What's reusable for our program

**The closest match to what we've already built — same tile size (12 SC × 14 sym), same CDL class (we use EPA, they use CDL-C), same supervised framing.**

Three direct experiments we can run at our scale:

### E02 — Layer-depth ablation
Sweep 1/2/3/4 encoder layers and 1/2/4/8 heads (param-matched where possible) on our existing 834k model. Plot BLER vs Eb/N0 on the EPA BLER sweep. **Claim to test: 2 layers saturate.**

### E02 — Early LayerNorm ablation
Remove early LN; measure BLER change. Test whether magnitude preservation actually helps in EPA regime.

### E03 — Multi-head receiver (bits + channel MSE)
Swap our receiver's output head to predict per-RE complex channel (MSE), no other architecture change. Then add a second head for bits. Validates the "shared backbone, task-specific head" claim.

## Differences vs our setup

- They use CDL-C (3GPP), we use EPA (also 3GPP, but smaller delay spread).
- They use 12 SC × 14 symbols (single RB); we use 300 SC × 14 symbols (full subframe). Our context is 25× larger.
- They deploy in a real OAI+Aerial stack with hard latency targets; we're pre-deployment.
- Their claim "1 layer × 1 head suffices on CDL-C" is striking; worth verifying on our setup and context size.
