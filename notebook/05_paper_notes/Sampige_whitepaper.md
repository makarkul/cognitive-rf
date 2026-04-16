# Sampige Semiconductors — RF Foundation Models whitepaper

**Source:** Internal Sampige whitepaper (`RF_Foundation_Models-WhitePaper.docx`) + companion deck (`RF_Foundation_Models.pptx`).

## Framing

RF-GPT is a useful proof-of-concept but has **six gaps**:
1. Image-patch tokenization isn't RF-native.
2. Text is the wrong prediction target.
3. Fully-synthetic validation.
4. No RRM (radio resource management) hook.
5. No temporal model.
6. No multi-antenna.

## Three foundational questions

- **Q1 — Vocabulary:** what are the "words" of RF?
- **Q2 — Tokenization + embedding:** how do we break a capture into tokens and map them to vectors?
- **Q3 — Prediction target:** `next-RF-token → next-channel-state → next-RRM-action`.

## Proposed pipeline

1. **Physics-informed feature extraction:** correlator bank + cyclostationary features + STFT.
2. **Entropy-driven adaptive segmentation:** equal information per token, not equal time.
3. **VQ-VAE codebook learning:** codebook sizes 512–8192; target perplexity plateau at 2k–8k.

Plus explicit study of embedding geometry:
- TwoNN intrinsic dimension.
- Linear probes for modulation, technology, channel.
- Contrastive geometric regularization.

## Prioritized applications

1. **Blind channel estimation** — primary near-term application.
2. **O-RAN Non-RT RIC rApp** delivering structured RF context to a Near-RT RIC xApp scheduler.

## Roadmap

- **Ph1 (m1–6):** corpus + tokenizer.
- **Ph2 (m7–12):** embeddings + blind channel estimation paper.
- **Ph3 (m13–18):** O-RAN xApp + OTA + Jio/Airtel partnership.

## How this aligns with the current 12-month program

| Sampige phase | Current program mapping |
|---|---|
| Ph1 tokenizer | Partially covered by Phase 1 (architecture ablation) + Phase 4 (spectrogram ViT, E15). We're not building a VQ-VAE yet; possible Phase 4 extension. |
| Ph2 blind channel estimation | Directly covered by our Phase 3 (E12–E14). |
| Ph3 O-RAN | Out of scope for year 1. Candidate for year 2. |

## What the whitepaper changes in the program

- **Validates the decision to pursue blind channel estimation as the phase-3 headline result.** The whitepaper names it as the near-term application.
- **Tokenizer work is implicit but unbuilt.** If VQ-VAE codebook learning is part of the long-term vision, it should be explored in a small experiment during Phase 4 — probably as E17 or a new E19.
- **Embedding geometry probes (intrinsic dim, linear probes) are already in the plan** as E09 and E10. Good alignment.
- **Physics-informed features vs learned tokenization** is an open design question. Current plan leans learned (the Linear(3, 128) input projection for E01, or a conv stem for E11). The whitepaper leans physics-informed. A comparison experiment is worth planning.

## Items to add to open questions

Already captured in [04_open_questions.md](../04_open_questions.md):
- Q8 (MREM vs supervised).
- Q10 (does encoder learn H implicitly?).
- Q12 (differentiable renderer vs end-to-end MSE).

Not yet captured — add:
- Q19. Physics-informed features (correlator bank + cyclostationary) vs pure learned tokenization — which is the right phase-4 starting point?
- Q20. Is VQ-VAE codebook learning worth the complexity for phase-4 tokenization, or does continuous embedding suffice?
