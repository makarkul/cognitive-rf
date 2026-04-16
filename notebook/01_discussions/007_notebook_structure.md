# Why the notebook is the repo

## The constraint

The year's work will produce ~18 experiments, 3 paper candidates, a stack of figures, and an arc connecting them. It needs to be:
- Shareable across 2 engineers + PI.
- Reproducible — any figure must regenerate from a fresh clone.
- Versioned — we need to see what an experiment looked like in month 3 vs month 9.
- Browseable — a new engineer should be able to read the arc in a day.

## What we rejected

| Tool | Why not |
|---|---|
| OneDrive / SharePoint / Google Drive | No version control, no branching, no diff. Fine for inputs (papers, raw captures); not for working artifacts. |
| Notion / Confluence | Pretty docs, but code and results can't live alongside reproducibly. Copy-paste breaks within a month. |
| Jupyter notebooks as primary artifact | `.ipynb` diffs badly; encourages hidden state. Use them *inside* experiments as scratchpads, not as the notebook itself. |
| Standalone W&B | Great for experiment tracking; not a notebook. No place for discussion prose or paper notes. |

## What we chose

A private GitHub repo, `makarkul/cognitive-rf`. Markdown + code + figures + discussions all in git.

Complementary tools:
- **W&B** for experiment tracking (training curves, hyperparameter sweeps, artifacts). Every `results.md` links to its W&B run.
- **MkDocs Material** (future, optional) to render the `notebook/` folder as a searchable website at year end.
- **GitHub Discussions** for informal threads; distilled into `01_discussions/` once they produce insight.
- **HuggingFace Hub or S3** for large files (checkpoints, IQ captures). Never commit to git.

## The experiment template

Every experiment writes to the same four files:

```
02_experiments/EXX_title/
├── hypothesis.md    <-- written BEFORE the run
├── method.md        <-- what code, what data, what config
├── results.md       <-- numbers, plots, verdict
└── figures/         <-- PDFs only
```

The single hardest discipline is writing `hypothesis.md` *before* running. It forces us to articulate what we expect and why. When the result surprises us, we know it surprised us — not "oh, that's what we expected all along."

## Process rules (one-line versions)

1. Hypothesis before experiment.
2. Negative results get the same template.
3. Every figure must regenerate from a single command.
4. Weekly Friday log, ~1 page, non-negotiable.
5. Large files go to external storage.
6. `open_questions.md` is append-only within a phase; pruned at phase boundaries.
7. Discussion files preserve reasoning that doesn't fit in a single experiment.

## Year-end artifact

At month 12 the `notebook/` folder is published (internally or externally) as a research wiki:
- One-page intro with the arc.
- 18 experiment pages.
- A discussions section.
- A paper-notes section.
- 3 paper PDFs linked at the top.

That is the thing you show to a partner, an investor, a new hire, or a peer reviewer. It's the whole year in one place.
