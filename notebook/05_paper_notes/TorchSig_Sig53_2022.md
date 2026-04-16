# TorchSig / Sig53 — Large-Scale RF Signal Classification Dataset + Toolkit

**Citation:** *Large Scale Radio Frequency Signal Classification.* arXiv 2207.09918 + GNU Radio Conference 2022.
**Code:** https://github.com/TorchDSP/torchsig
**Project page:** https://torchsig.com

## Core claim

A 5-million-sample, 53-class RF classification dataset (**Sig53**) plus a complex-valued PyTorch pipeline (**TorchSig**) establishes a reproducible benchmark for RF classification. Transformers outperform ConvNets on it without teacher distillation.

## What TorchSig actually is

Two things bundled together:

1. **Sig53 dataset** — 5M RF samples spanning 53 signal classes (digital modulations, analog modulations, some wideband types). Synthetic via a configurable generation pipeline.
2. **TorchSig library** — complex-valued PyTorch primitives for RF ML: complex conv layers, signal generators, data augmentations (fading, AWGN, phase noise, frequency offset, IQ imbalance), standard model baselines.

## Core result

On Sig53 modulation classification:
- Transformers beat ConvNets.
- No teacher-student distillation needed.
- Establishes a clean benchmark number.

## What's reusable for our program

### As a benchmark dataset
Sig53 is the closest thing to an "ImageNet for RF classification." If we want to report downstream classification accuracy for any of our pretrained encoders (E07, E12, E15), running them on Sig53 is the right comparison point.

**Concrete use case in Phase 2:**
- Pretrain MREM encoder on our LTE-5 synthetic grids.
- Freeze the encoder.
- Build a Sig53 classification head on top.
- Compare against the published TorchSig transformer baseline.

If our encoder beats the TorchSig baseline, that's a publishable transfer result. If it matches, still useful. If it loses, informative about the LTE-specificity of our pretraining distribution.

### As a generator library
`torchsig`'s augmentation pipeline (fading, frequency offset, IQ imbalance, phase noise) is already implemented and tested. For our impairment-generalization experiment (E05), this saves building the impairment injection ourselves.

### As an architecture reference
TorchSig ships complex-valued layers. If we ever go complex-valued (which is cleaner for RF than real-imag-concatenated), their implementation is a starting point — no need to re-derive complex Adam or complex LayerNorm.

## What to watch out for

- TorchSig's generator is synthetic; real OTA performance on Sig53-trained models is usually weaker (the sim-to-real gap we keep flagging).
- The 53 classes are curated for classification difficulty, not representative of any real band plan. Don't treat Sig53 as "how a receiver sees the world."
- License: BSD-3. Compatible with ours.

## URLs

- arXiv: https://ar5iv.labs.arxiv.org/html/2207.09918
- Code: https://github.com/TorchDSP/torchsig
- Project site: https://torchsig.com
