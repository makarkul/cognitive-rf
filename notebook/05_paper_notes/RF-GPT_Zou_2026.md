# RF-GPT — Teaching AI to See the Wireless World

**Citation:** Zou, Tian, Wang, Bariah, Lasaulce, Huang, Debbah. arXiv 2602.14833v1. Khalifa/Zhejiang/Nancy. February 2026.

## Core claim

A vision-language model (Qwen2.5-VL 3B/7B) fine-tuned on STFT magnitude spectrograms, using a tiny linear adapter and synthetic instruction tuning, becomes a "radio-frequency language model" that answers free-form natural-language questions about RF scenes — vastly outperforming off-the-shelf VLMs (and GPT-5) on modulation/overlap/technology/UE-count benchmarks.

## Input representation

- IQ → STFT (Blackman window, FFT 512, hop 512).
- Magnitude in dB, clipped, normalized, resized to 512×512 pseudo-RGB.
- Standard ViT partitioner: 14×14 patches → **1369 RF tokens** per spectrogram.
- Magnitude only; phase discarded.

## Model architecture

Three-stage pipeline:
- Pretrained vision encoder (ViT).
- Linear projection adapter → LLM embedding dim.
- Decoder-only LLM (Qwen2.5-VL backbone: RoPE, RMSNorm, grouped-query attention, SwiGLU MLP).

Autoregressive text generation conditioned on the RF prefix. Supervised instruction fine-tuning, 3 epochs, AdamW lr=2e-4, batch 256, 8× H200.

## Training objective

Standard autoregressive cross-entropy over *answer* tokens only, conditioned on `(q, ϕ_RF(x))`. No SSL/masking; purely instruction-following SFT.

## Training data

Fully synthetic:
- MATLAB wireless toolboxes (5G, LTE, UMTS, WLAN, Satellite, Bluetooth) + TorchSig.
- Rejection-sample valid configs, log metadata, generate dense caption JSON with 5 information levels (summary/global-visual/global-context/signal-visual/signal-context).
- GPT-OSS-120B converts captions → diverse instruction–answer pairs.
- **12k RF scenes → 625k instruction pairs.** Zero human labels, zero OTA.

## Downstream tasks (5 VQA-style benchmarks)

- WBMC — wideband modulation classification, 57 classes, 3 difficulty tiers.
- WBOD — overlap detection in time/frequency.
- WTR — wireless tech + DL/UL recognition across 6 standards.
- WNUC — WLAN user counting (802.11ax vs 11be, MU-MIMO vs OFDMA).
- NRIE — NR information extraction: SCS, SSB pattern, CSI-RS count, SRS count, UE count.

## Headline numbers

- RF-GPT-7B joint WTR accuracy **99.64%** vs ~5% baselines.
- NRIE: SCS 99.1%, SSB pattern 94.1%, UE count 64.2%.
- WBMC-Easy 82.4% vs 7% for GPT-5.
- WBOD-Hard 71.7% vs 12%.
- Baseline VLMs (Qwen2.5-VL, GPT-5) are at chance everywhere.
- **Impairment ablation (most useful finding):** CFO and PA nearly flat degradation; TDL moderate; **IQ imbalance most destructive (6.1 pp drop).**
- Resolution ablation: 224→512 gives +8–10pp across WBMC tiers.

## What's reusable for our program

Full RF-GPT is out of reach (8× H200, no VLM). Three stripped-down ideas transfer:

### E05 — Impairment generalization probe
Sweep CFO, IQ imbalance, PA Rapp nonlinearity, alternate TDL at 5 intensity levels. Train at λ ≤ 0.3, test across full range. Directly copies their protocol.

### Phase 4 caption → instruction pipeline
Our LTE-5 generator already produces metadata (SNR, pilot mask, RB allocation). A small MLP probe on our transformer's CLS-equivalent can be trained to output discrete labels (SNR bucket, Doppler bucket, EPA-vs-AWGN). Less ambitious than NLP Q/A but structurally similar.

### Negative baseline
Run Qwen2.5-VL-3B-Instruct on our spectrograms to establish the bar that any of our learned receivers must clear. "General VLMs have no RF prior" — a useful sanity check to document.

## Differences vs our setup

- Spectrograms vs RE grids. Different input representation; the spectrogram path is our phase 4.
- Text output vs bit output. Different task; we care about receiving, not describing.
- 3B/7B LLM vs 834k transformer. Different scale; our budget is 4–5 orders smaller.
- Fully synthetic (like us). Same sim-to-real caveat applies.
