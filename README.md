# cognitive-rf

Research program: small transformers for cognitive wireless receivers.

The arc:
1. **Sinusoid recovery** — next-sample prediction transformer; prove a 29k-param model can learn AR(2) structure implicitly. *(complete)*
2. **OFDM recovery (supervised)** — 834k-param transformer receiver on the LTE-5 resource grid; beat classical LS+interp and the perfect-CSI ZF oracle. *(complete — E01)*
3. **OFDM recovery (self-supervised)** — masked-RE pretraining; label-efficient downstream tasks. *(planned — E07+)*
4. **Blind / pilotless operation** — recover CFO and coarse CIR from raw IQ without pilots. *(planned — E12+)*
5. **Standard-agnostic receiver** — spectrogram-tile ViT; sniff the standard from thin air. *(planned — E15+)*

The long-term target is a cognitive wireless system that listens passively, understands its RF environment, and builds a modem best suited for it.

## Repo layout

```
cognitive-rf/
├── notebook/          <-- the research artifact (read this first)
│   ├── 00_intro.md
│   ├── 01_discussions/   curated conversations
│   ├── 02_experiments/   one folder per experiment, fixed template
│   ├── 03_results_index.md
│   ├── 04_open_questions.md
│   ├── 05_paper_notes/   distilled prior work
│   └── logs/             weekly Friday logs
├── experiments/        code per experiment
├── shared/             reused across experiments (LTE params, channel models, pilot utils)
└── requirements.txt
```

## Process rules

1. `hypothesis.md` is written *before* the experiment runs.
2. Every experiment has `hypothesis.md`, `method.md`, `results.md`, `figures/`.
3. Negative results use the same template.
4. Weekly Friday log, ~1 page, non-negotiable.
5. Any figure in the notebook must regenerate from a single command.
6. Large files (checkpoints, IQ captures) go to external storage, not git.

## Quick start

See [notebook/00_intro.md](notebook/00_intro.md) for the full arc and [notebook/02_experiments/E01_ofdm_supervised_rx/results.md](notebook/02_experiments/E01_ofdm_supervised_rx/results.md) for the first completed result.
