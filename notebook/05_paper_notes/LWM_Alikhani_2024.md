# LWM — Large Wireless Model

**Citation:** Alikhani, Charan, Alkhateeb. arXiv 2411.08872v2 (extended from IEEE ICMLCN 2025). Arizona State University.

## Core claim

A self-supervised transformer pre-trained with masked channel modeling on DeepMIMO channels yields universal, task-agnostic channel embeddings that beat raw-channel baselines on downstream tasks — especially when labeled data is scarce.

## Input representation

- MIMO-OFDM channel matrix `H ∈ C^{32×32}` (32 antennas × 32 subcarriers).
- Split into real/imag, flattened, diced into P=128 patches of length L=16 (64 real + 64 imag).
- Learnable CLS patch prepended.

## Model architecture

- BERT-style encoder-only transformer.
- 12 layers, 12 heads, embedding D=64, FFN hidden 256, head dim 5.
- **~600k parameters** — essentially our scale.
- No classification head during pretraining; a linear decoder reconstructs masked patches.

## Training task

Masked Channel Modeling (MCM), lifted directly from BERT:
- 15% of patches masked (80% replaced with MASK vector, 10% random, 10% untouched).
- MSE reconstruction loss over masked patches.
- Real and imag of the same index masked together to avoid leakage.

## Training data

- DeepMIMO ray-tracing synthetic channels.
- 15 scenarios (O1, Boston5G, ASU Campus, 12 US cities).
- ~820k train + 200k val channels.
- Pure simulation, no OTA.

## Downstream tasks

- Sub-6 GHz → 28 GHz best-beam prediction on 6 unseen DeepMIMO cities (14,840 samples).
- LoS/NLoS classification on densified Denver (6,639 training samples).

Each compares: frozen CLS embedding / frozen channel-patch embeddings / fine-tuned last-3-layer embeddings vs. raw channel on a 500k-param Residual 1D-CNN.

## Headline numbers

- LWM embeddings reach target F1 with **40–50% of the training data** a raw-channel model needs.
- LoS/NLoS with only **6 labeled samples**: raw F1≈0.55, general-purpose CLS F1≈0.86 (+0.31), fine-tuned F1≈1.0.
- CLS embeddings robust to 5 dB complex-Gaussian corruption.

## What's reusable for our program

The masked-patch SSL recipe translates almost verbatim to our LTE-5 resource grid. We already have an RE grid and a 834k-param receiver — **same parameter budget as LWM**.

**Concrete replication (scheduled as E07):**
- Treat our post-FFT resource grid as the "channel matrix."
- Strip pilots and data labels.
- Train the receiver encoder to reconstruct randomly masked REs using MCM (80/10/10 mask policy, MSE on masked positions only).
- Freeze and probe the CLS embedding with a tiny linear head for SNR regression or EPA-vs-AWGN classification (E09).

This gives us a foundation-model pretraining stage that we currently don't have — E01 used supervised EPA labels.

## Differences vs our setup

- LWM operates on `H` matrices. We operate on *received signal* grids. The model needs to implicitly learn the channel rather than observe it directly. Arguably harder, but more realistic.
- LWM has no pilots; the full `H` is the input. Our grid has pilots flagged. Whether to expose the pilot mask during SSL pretraining is an open question — probably yes, for downstream transferability.
- Our grid is 14×300 vs their 32×32. Larger context, similar parameter budget.
