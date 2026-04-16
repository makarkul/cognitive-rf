# Paper notes

Distilled notes on prior work relevant to the program. One file per paper. Each note extracts:
- Core claim in one sentence.
- Input representation (raw IQ, spectrogram, resource grid).
- Model architecture + param count.
- Training task/objective.
- Training data (synthetic? real? what standards?).
- Downstream tasks demonstrated.
- Key result / headline number.
- What's reusable for our program.

## Index

### Core RF foundation models

| ID | Paper | Year | Input | SSL type | Most relevant to |
|---|---|---|---|---|---|
| [LWM](LWM_Alikhani_2024.md) | Large Wireless Model — Alikhani, Charan, Alkhateeb (ASU) | 2024 | RE grid | masked recon | Phase 2 (MREM pretraining, E07) |
| [SoftBank](SoftBank_Kawai_2025.md) | Unified Transformer for Wireless Signal Processing — Kawai & Koodli | 2025 | RE grid | supervised | Phase 1 (ablation, E02, E03) |
| [RF-GPT](RF-GPT_Zou_2026.md) | RF-GPT: Teaching AI to See the Wireless World — Khalifa U. | 2026 | spectrogram (VLM) | vision-language | Phase 4 (spectrogram ViT, impairment sweep) |
| [IQFM](IQFM_Mashaal_2025.md) | IQFM — raw-IQ contrastive SSL (WAVES Lab, Calgary) | 2025 | raw IQ (multi-antenna) | contrastive | Phase 2 alt (E07b), Phase 3 (E11), Phase 4 (E18 few-shot) |
| [WirelessJEPA](WirelessJEPA_Chu_2026.md) | WirelessJEPA — latent prediction on antenna-time grid (WAVES Lab) | 2026 | raw IQ (multi-antenna) | JEPA | Phase 2 stretch, Phase 3 (E13) |
| [WavesFM](WavesFM_Aboulfotouh_2024.md) | WavesFM — masked spectrogram modeling (WAVES Lab) | 2024 | spectrogram | masked recon | Phase 4 (E15 spectrogram-tile ViT) |

### Datasets, classifiers, and tooling

| ID | Paper | Year | What it provides | Most relevant to |
|---|---|---|---|---|
| [T-PRIME](T-PRIME_Genesys_2024.md) | T-PRIME — transformer-based WiFi protocol ID (Northeastern GENESYS) | 2024 | 66 GB real OTA WiFi corpus + open-source transformer | Phase 3 sim-to-real probe, Phase 4 (E15 OTA validation) |
| [TorchSig / Sig53](TorchSig_Sig53_2022.md) | TorchSig + Sig53 — 5M-sample, 53-class RF benchmark (TorchDSP) | 2022 | Benchmark dataset + complex-valued PyTorch library | Phase 2 (transfer benchmark), Phase 1 (E05 impairments) |

### Program-level references

| ID | Paper | Year | Most relevant to |
|---|---|---|---|
| [BRF-WM](BRF-WM.md) | Blind RF World Model — concept deck | 2025 | Phase 3 (blind CFO + CIR, E12) |
| [Sampige whitepaper](Sampige_whitepaper.md) | RF Foundation Models — internal | 2026 | Program direction, all phases |

### Cross-reference

See discussion [008 — "RFFormer" is not a paper](../01_discussions/008_rfformer_is_not_a_paper.md)
for why the three Calgary WAVES Lab entries (IQFM, WirelessJEPA, WavesFM)
and the two Northeastern / TorchDSP entries (T-PRIME, TorchSig) all trace
to the **DARPA RFMLS** program lineage, and what that means for our
program's open-dataset / open-benchmark options.

## Writing paper notes

Each note should answer: "**what concrete experiment from this paper could we replicate at a small scale with our existing infrastructure?**" If the answer is "nothing," the paper is background reading, not a reference — note it more briefly.

Add new notes as papers are read. Keep this index updated.
