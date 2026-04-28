# Discussions

Curated conversations that produced design decisions or insight but don't fit in a single experiment's writeup. Each file is distilled from a working chat thread.

## Index

- [001 — n_heads rationale for sinusoid recovery](001_n_heads_rationale_sinusoid.md) — why n_heads=2 for a sinusoid is exactly the AR order.
- [002 — ZF vs learned at high SNR](002_zf_vs_learned.md) — why the learned receiver beats the perfect-CSI oracle, and why that's not a violation.
- [003 — Oracle vs LS+interp baselines](003_oracle_vs_ls_interp.md) — what the two classical baselines mean and how to read the gaps.
- [004 — Training setup for E01](004_training_setup_ofdm.md) — on-the-fly data generation, not spectrograms, not autoregressive.
- [005 — n_heads rationale for OFDM](005_n_heads_rationale_ofdm.md) — why 4 heads maps onto 4 natural relational patterns (freq pilot, time pilot, global pool, local data).
- [006 — Generative simulator and the cognitive radio arc](006_generative_simulator_and_cognitive_radio.md) — what "generative simulator" means in our usage, and how the year plan connects to the cognitive-radio target.
- [007 — Notebook structure and process rules](007_notebook_structure.md) — why the notebook is the repo and how experiments get filed.
- [008 — "RFFormer" is not a paper](008_rfformer_is_not_a_paper.md) — clarifies a piece of imprecise shorthand; the real referent is DARPA RFMLS + the adjacent paper cluster in `05_paper_notes/`.
- [009 — Denoising autoregressor vs LM-style next-sample prediction](009_lm_vs_denoiser_equivalence.md) — when the canonical E00 setup (clean target) is equivalent to true LM-style training (noisy target), and where this matters for E07/E12.

## Writing discussion files

Good candidates for a new discussion file:
- A recurring question that keeps coming up.
- A design decision that needed a paragraph of reasoning.
- An analogy or framing that unlocked later progress.
- A negative result from a thought experiment.

Don't write one for:
- Single-paragraph answers.
- Things that belong in a specific experiment's `results.md`.
- Advertising — if there's no reasoning to preserve, skip it.

Keep each file short (1–3 pages). If it grows past that, it's an experiment.
