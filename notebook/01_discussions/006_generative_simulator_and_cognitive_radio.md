# Generative simulator, passive listening, and the cognitive-radio arc

**Context:** the long-term target is a cognitive wireless system — one that passively listens, understands its RF environment, and adapts its modem. This discussion clarifies two terms that came up during planning and places the 12-month program in context.

## Passive listening without reference bits

The E01 setup is *fully supervised*: the simulator generates bits, runs them through the channel, and supervises on the known labels. A cognitive receiver in the field won't have that — it has only the received IQ. So the relevant toolbox is **self-supervised** learning. Named options, in rough order of increasing ambition:

| Pretext task | What the model learns | Phase in our plan |
|---|---|---|
| **Masked-RE prediction** | Zero out a random 15% of REs on the 14×300 grid, ask the model to reconstruct them. BERT-on-a-resource-grid. Learns channel smoothness + pilot structure with no bit labels. | Phase 2 (E07) |
| **Masked-IQ prediction** (raw time domain) | Same idea, one level lower. Mask a chunk of IQ samples, predict them from the rest. wav2vec / HuBERT analog. Learns cyclic prefix, symbol boundaries, constellation from nothing. | Phase 4 |
| **Cyclostationary self-supervision** | OFDM waveforms have repeating structure (CP ↔ symbol tail correlate; pilots lie on a lattice; slot/frame boundaries repeat). Train the model to predict future periodicities from past ones. This is where "sniff the standard from thin air" lives. | Phase 4 |
| **Contrastive** | Two captures of the same modem embed close; two captures of different modems far apart. Teaches the model to *cluster* RF environments without labels. | Phase 4 |
| **Finite-alphabet EM / decision-directed** | Bootstrap: low-confidence predictions → threshold → pseudo-labels → retrain. Works when the model is already "mostly right." Useful as a fine-tuning step. | Phase 3 |

The realistic path is: **pretrain big on self-supervised RF**, **fine-tune small on task**. This is the wav2vec / BERT / MAE recipe ported to RF. E01 is the supervised warm-up proving the architecture can equalize a grid; E07 onward is the first step toward a self-supervised model.

## What is a "generative simulator"?

In our usage: a **parametric sampler** with knobs (channel profile, SNR, CFO, pilot value, bandwidth, modulation) that draws fresh synthetic examples from a distribution. Not a generative model in the GAN/diffusion sense.

The distinction matters:
- Training on a **fixed** synthetic dataset overfits to a handful of channel realizations.
- Training on a **sampler** — each batch has fresh channels, fresh SNR, fresh impairments — forces the model to handle the distribution. That's what our current `generate_batch` does.

"More generative" in our day-to-day usage just means wider knobs and more randomization per batch.

A **true** generative simulator would be a learned model of RF captures — diffusion or VAE trained on real IQ, conditioned on `(modulation, bandwidth, channel class)`. That would close the sim-to-real gap by making the "simulator" match reality. We're not close; it's a research frontier.

## The cognitive-radio ambition, componentized

Worth naming the pieces so the year plan has labels:

| Piece | Who does it | Where we are |
|---|---|---|
| **Sensing** (what's in the band?) | Classifier/detector over spectrograms. | Phase 4 (E15–E17) |
| **Understanding** (what's the modem doing?) | Self-supervised foundation model over IQ. | Phase 2–4 |
| **Decoding** (get bits without standard knowledge) | Learned receiver, pilot/reference-free. | E01 is the supervised warm-up; pilotless version is Phase 3 (E12–E14). |
| **Adaptation** (pick a waveform that suits the environment) | RL loop on top of the above. | Phase 4 (E18) |
| **Co-design of TX & RX** | Joint learned modem (both ends trained together). | Out of current scope; research frontier. |

The piece that's genuinely hard and genuinely unsolved is **"understanding"** — a general-purpose embedding of RF captures that supports downstream demod/classify/detect without per-task training. That's the RF equivalent of a foundation model.

## Our path vs. the field

This arc — sinusoid → OFDM → spectrogram ViT → self-supervised pretraining — is the conservative, evidence-at-each-step version of the RF-foundation-model direction. Public examples of that direction include:
- **LWM** (ASU, 2024) — masked channel modeling on DeepMIMO.
- **SoftBank Unified Transformer** (2025) — production-grade OAI+Aerial deployment.
- **RF-GPT** (Khalifa U., 2026) — VLM fine-tuned on spectrograms for language Q/A.
- **BRF-WM** (proposed) — blind RF world model with differentiable signal renderer.

See [05_paper_notes/](../05_paper_notes/) for details. The direction is not unique to us; the execution discipline is.

## Why each phase of the plan is ordered as it is

1. **Phase 1 — validate and shrink the supervised receiver.** Can't build SSL on a shaky supervised foundation. This gives us a known-good baseline to beat in phase 2.
2. **Phase 2 — prove SSL works on RE grids.** Cheapest path to the first external paper. Result decides whether the foundation-model bet is justified.
3. **Phase 3 — drop the pilots.** First concrete step toward a cognitive receiver. The "passive listening" ambition in its most testable form.
4. **Phase 4 — drop the standard.** Step up to raw IQ + multi-standard. This is the full cognitive-radio vision, in scope only if phases 1–3 work.

Each phase delivers a paper-shaped artifact, so even a partial success (e.g., phases 1–2 land but phase 3 stalls) produces publishable results.
