# WirelessJEPA — Multi-Antenna Foundation Model via Spatio-temporal Wireless Latent Predictions

**Citation:** Chu, Mashaal, Abou-Zeid. arXiv 2601.20190 (January 2026). WAVES Lab, University of Calgary (same group as IQFM).

## Core claim

A Joint-Embedding Predictive Architecture (LeCun-style JEPA) applied to multi-antenna wireless inputs — using spatio-temporal block masks and convolutional patch processing — produces generalizable representations without augmentation engineering, beating contrastive baselines especially on out-of-distribution tasks.

## Input representation

- Multi-antenna IQ reshaped into a **2D antenna-time grid.**
- Convolutional patch processing turns the grid into a sequence of patches.

## Model architecture

- JEPA: *predict a latent representation of a masked region from the unmasked region*, not reconstruct pixels.
- Two networks: a context encoder and a target encoder (the target is often an EMA of the context).
- Novel spatio-temporal block masks — mask a coherent 2D rectangle, not random pixels.

## Training task

- **Masked latent prediction.** No contrastive pairs, no data augmentations to engineer.
- Loss is on the latent space, not on reconstructed IQ.

## Downstream tasks / headline

- 6 tasks evaluated (specifics to be pulled on full read).
- Reports strong **out-of-distribution generalization** against contrastive baselines.

## What's reusable for our program

The JEPA framing is a *third* SSL option alongside masked reconstruction (LWM) and contrastive (IQFM). Each has different properties:

| SSL type | Pros | Cons |
|---|---|---|
| Masked reconstruction (LWM, WavesFM) | Simple; no augmentation engineering; dense gradient | Reconstructing pixels/IQ may waste capacity on low-level detail |
| Contrastive (IQFM) | Produces clean embedding geometry; proven few-shot | Needs hand-designed augmentations; negative sampling tricky |
| **JEPA (WirelessJEPA)** | No augmentations, no reconstruction; latent-space objective forces semantic compression | Newer; less mature tooling; EMA/target-network training adds complexity |

### For Phase 2 design
Treat JEPA as a **Phase 2 stretch option**. Default plan stays MREM. If masked reconstruction gives weak downstream probes, or if we see the model wasting capacity on fine-grained reconstruction, switch to JEPA as a fallback.

### For Phase 3 (blind)
JEPA's latent-prediction objective is philosophically aligned with BRF-WM's propagation/content disentanglement: you want the encoder to predict *slow-varying latent* (H) from context, not to reconstruct raw IQ. Worth revisiting when we build E13.

## Caveats

- Multi-antenna; our setup is SISO. Grid structure is different.
- JEPA training is more fiddly than MCM (EMA schedules, target network, latent collapse to avoid). Don't adopt as default without a specific reason.

## URL

https://arxiv.org/abs/2601.20190
