# WavesFM — Spectrogram Foundation Model with Masked Spectrogram Modeling

**Citation:**
- *Building 6G Radio Foundation Models with Transformer Architectures.* Aboulfotouh, Eshaghbeigi, Abou-Zeid. arXiv 2411.09996 (Nov 2024).
- Follow-up: *6G WavesFM: A Foundation Model for Sensing, Communication, and Localization.* arXiv 2504.14100 (Apr 2025).
- Same Calgary WAVES Lab as IQFM and WirelessJEPA.

## Core claim

A Vision Transformer pretrained with Masked Spectrogram Modeling (MSM) on RF spectrograms produces representations that support multiple downstream tasks (sensing, comms, localization). A pretrained ViT beats a 4× larger from-scratch baseline on downstream segmentation.

## Input representation

- **RF spectrograms** treated as images.
- Standard ViT patchification.

## Model architecture

- Vision Transformer (ViT) — off-the-shelf image transformer, no RF-specific modifications.
- Encoder-only during pretraining with a lightweight decoder for reconstruction.

## Training task

- **Masked Spectrogram Modeling (MSM)** — MAE-style: mask a high fraction of patches, reconstruct from the visible ones.
- Pure SSL; no supervision during pretraining.

## Training data

- RF spectrograms (6G / mmWave provenance; details in the paper).

## Downstream tasks

- CSI-based human activity sensing.
- Spectrogram segmentation.
- (WavesFM extension adds more: localization, communication link adaptation.)

## Headline result

- Pretrained ViT beats a **4× larger** from-scratch ViT on segmentation.
- Demonstrates transfer across task families with frozen + probe setup.

## What's reusable for our program

**Directly relevant to Phase 4 (E15: spectrogram-tile ViT).** WavesFM is effectively the "what we want to build" reference for that experiment, minus the 4× model scale.

### For E15 design
- Start with an off-the-shelf ViT (not a custom RF transformer) to match the reference.
- Use MAE-style high mask ratio (75%), not BERT-style 15%. Spectrograms are locally smooth, so masked regions need to be large enough to force semantic understanding.
- Encoder-only + lightweight reconstruction decoder. Throw the decoder away after pretraining.

### For cross-phase signal
If Phase 2 (MREM on RE grid) gives a 2–4× label-efficiency gain, and Phase 4 (MSM on spectrograms) matches the WavesFM 4× result, we have consistent evidence that SSL on RF spectrogram/grid representations works. That's a defensible multi-paper narrative.

## Differences vs our setup

- WavesFM operates on pre-computed spectrograms. Our Phase 4 target is tile-level attention over the same kind of input, at much smaller scale.
- They're 6G / mmWave; we're LTE-5 at 2 GHz-ish. Channel statistics differ.
- They use off-the-shelf ViT; we'd likely start with our existing 834k transformer and just swap the embedding to patch-projection.

## Related (same group)

- [IQFM](IQFM_Mashaal_2025.md) — the raw-IQ sibling of WavesFM.
- [WirelessJEPA](WirelessJEPA_Chu_2026.md) — the multi-antenna sibling with a JEPA objective.

The Calgary WAVES Lab has essentially been running **our same three-axis exploration** (IQ vs spectrogram vs multi-antenna) × (contrastive vs masked vs JEPA). Worth treating their output as the most directly comparable prior art.

## URLs

- https://arxiv.org/abs/2411.09996
- https://arxiv.org/html/2504.14100
