# "RFFormer" is not a real paper — what we actually mean

## Context

Early in the program I used the name **"RFFormer"** several times when
referring to the RF-transformer foundation-model cluster. A web search
and a reasonable literature review both come up empty — **there is no
paper or project called RFFormer.** The term was informal shorthand on
my part, and it does not correspond to any specific piece of work. This
note records what was actually meant so the notebook has a pointer for
anyone reading later (including future us).

## What I was conflating

Two distinct things got mashed together:

1. **DARPA RFMLS** — *Radio Frequency Machine Learning Systems*. A real
   DARPA MTO program (roughly 2017–2021) with the brief "apply modern ML
   to raw RF for classification, geolocation, and protocol identification
   without handcrafted features." Not a paper, a program.
2. **The transformer-on-RF paper cluster** — a collection of actual papers
   (LWM, IQFM, WavesFM, WirelessJEPA, T-PRIME, TorchSig, SoftBank,
   RF-GPT) that either came directly out of RFMLS or were seeded by the
   same research agenda.

"RFMLS" + "Transformer" got compressed into "RFFormer" in my head. It
shouldn't have.

## What actually exists in the RFMLS lineage

| Artefact | Relationship to RFMLS | Paper note |
|---|---|---|
| **RadioML 2016 / 2018 datasets** (DeepSig) | Produced under RFMLS funding; the canonical dataset that trained the first generation of RF classifiers | (not yet added — worth writing) |
| **TorchSig / Sig53** (TorchDSP) | Follow-on productization of the RFMLS methodology | [TorchSig_Sig53_2022.md](../05_paper_notes/TorchSig_Sig53_2022.md) |
| **T-PRIME** (Northeastern GENESYS) | Northeastern was a major RFMLS performer; T-PRIME is post-RFMLS continuation | [T-PRIME_Genesys_2024.md](../05_paper_notes/T-PRIME_Genesys_2024.md) |
| **RF fingerprinting work** (MIT Lincoln, Northeastern) | Original RFMLS thrust areas | Cited in [IQFM](../05_paper_notes/IQFM_Mashaal_2025.md) |

Everything else in our paper-notes folder (LWM, IQFM, WavesFM,
WirelessJEPA, RF-GPT, SoftBank) is **adjacent** to RFMLS — same research
agenda, different funding sources.

## Why this matters for the program

The DARPA RFMLS lineage is relevant to our program in one concrete way:
**it produced open datasets and open benchmarks.** Specifically:

- **TorchSig / Sig53** gives us an "ImageNet for RF classification" —
  useful as a downstream transfer benchmark for our pretrained encoders
  (Phase 2, Phase 4).
- **T-PRIME's 66 GB WiFi OTA corpus** gives us real-world captures for
  sim-to-real validation when our synthetic-only pipeline needs a reality
  check (Phase 3, Phase 4).

Without those artefacts, Phase 4 of our program (standard-agnostic
sniffing) would have to either generate its own real-OTA dataset (months
of effort) or accept a purely synthetic evaluation. Having RFMLS-descended
open data cuts that risk substantially.

## Takeaway for future correspondence

When the shorthand "RFFormer" appears in older notes, chats, or commit
messages, read it as:

> the RFMLS-adjacent cluster of transformer-on-RF papers catalogued in
> `05_paper_notes/`

Not as a specific paper. The correct name for the lineage is **DARPA
RFMLS**. The correct way to cite specific work is by the individual
paper (LWM, IQFM, WavesFM, etc.).

Going forward, the term is retired.
