# IQFM — Wireless Foundational Model for I/Q Streams

**Citation:** Mashaal, Abou-Zeid. arXiv 2506.06718 (June 2025, rev. June 20, 2025). WAVES Lab, University of Calgary.
Original title: *Multi-Task Self-Supervised Learning for Generalizable IQ Representations*. Current branding: *IQFM: A Wireless Foundational Model for I/Q Streams in AI-Native 6G*.

## Core claim

A lightweight transformer pre-trained with contrastive SSL on raw multi-antenna over-the-air I/Q streams produces generalizable representations that excel at multiple downstream tasks with very few labels — using task-aware augmentations (cyclic time shifts, plus task-specific temporal/spatial variants).

## Input representation

- **Raw multi-antenna OTA I/Q streams.** No spectrograms, no handcrafted features, no FFT preprocessing.
- This is the most IQ-native of the RF foundation models catalogued so far.

## Model architecture

- Lightweight transformer encoder.
- Specifics (layer/head count, parameter size) not extracted in my skim — retrieve from the paper when implementing.
- LoRA adapters for downstream fine-tuning.

## Training task

- **Contrastive SSL** with a set of RF-specific augmentations:
  - Core augmentation: **cyclic time shifts** (exploits OFDM's periodicity; same idea as CP autocorrelation).
  - Plus task-specific temporal/spatial augmentations.
- No masked reconstruction. Contrastive only.

## Training data

- Multi-antenna OTA captures. Volume and provenance to be confirmed on full read.

## Downstream tasks / headline numbers

- **Modulation classification:** 99.67% with only **1 labeled sample per class.**
- **AoA (angle of arrival):** 65.45% with 1 sample/class.
- **With 500 samples/class + LoRA:**
  - Beam prediction: 94.15%
  - RF fingerprinting: 96.05%

## What's reusable for our program

This is arguably the closest match to our **"passive listening" cognitive-radio ambition**. Three specific takeaways:

### For E07 (MREM pretraining)
Contrastive SSL is a *genuine alternative* to masked reconstruction. Worth considering both as E07 options:
- **E07a — MREM (LWM recipe):** masked-RE MSE reconstruction.
- **E07b — contrastive (IQFM recipe):** positive pair = (clean view, cyclic-shifted view) of the same subframe.

Run both at matched params; compare downstream probe accuracy. Not a guaranteed head-to-head, but cheap to do.

### For E11 (IQ-domain conv stem)
IQFM demonstrates that a transformer over raw IQ works. Our Phase 3 IQ front-end doesn't need a large VGG-style stem — a simple learned patch projection over IQ windows should suffice.

### For E18 (few-shot downstream probes)
The "1 sample per class" headline is striking. If our pretrained encoder can match even a loose version of that (e.g., 10 samples per SNR bucket), we have a publishable label-efficiency story.

## Differences vs our setup

- IQFM is **multi-antenna**; our current setup is single-antenna (SISO). Multi-antenna extension is a Phase 4+ stretch goal.
- IQFM uses real OTA captures; we're fully synthetic. Sim-to-real gap applies.
- IQFM is contrastive; our Phase 2 baseline is masked reconstruction. Testing both is an open design question.

## URL

https://arxiv.org/abs/2506.06718
