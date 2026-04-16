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

| ID | Paper | Year | Most relevant to |
|---|---|---|---|
| [LWM](LWM_Alikhani_2024.md) | Large Wireless Model — Alikhani, Charan, Alkhateeb (ASU) | 2024 | Phase 2 (MREM pretraining, E07) |
| [SoftBank](SoftBank_Kawai_2025.md) | Unified Transformer for Wireless Signal Processing — Kawai & Koodli | 2025 | Phase 1 (ablation, E02, E03) |
| [RF-GPT](RF-GPT_Zou_2026.md) | RF-GPT: Teaching AI to See the Wireless World — Khalifa U. | 2026 | Phase 4 (spectrogram ViT, impairment sweep) |
| [BRF-WM](BRF-WM.md) | Blind RF World Model — concept deck | 2025 | Phase 3 (blind CFO + CIR, E12) |
| [Sampige whitepaper](Sampige_whitepaper.md) | RF Foundation Models — internal | 2026 | Program direction, all phases |

## Writing paper notes

Each note should answer: "**what concrete experiment from this paper could we replicate at a small scale with our existing infrastructure?**" If the answer is "nothing," the paper is background reading, not a reference — note it more briefly.

Add new notes as papers are read. Keep this index updated.
