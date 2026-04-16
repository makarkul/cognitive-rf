# Results — E06: Interpretability probes on the E01 learned receiver

*Skeleton. Each probe fills its own section when it completes. Do not
edit the hypothesis.md once a probe has started — refutations live
here, not there.*

## Headline

**TBD** — fill in after all four probes complete. One or two sentences.

## Probe 1 — Linear probe for H[k]

### Status

Not run yet.

### Numbers

| Layer | R² @ pilot REs, 5 dB | R² @ pilot REs, 15 dB | R² @ pilot REs, 25 dB | R² @ data REs, 15 dB | R² @ data REs, 25 dB |
|---:|---:|---:|---:|---:|---:|
| 0 (post-emb) | — | — | — | — | — |
| 1 | — | — | — | — | — |
| 2 | — | — | — | — | — |
| 3 | — | — | — | — | — |
| 4 (final) | — | — | — | — | — |

### Did the hypothesis hold?

TBD.

### Figures

- `figures/probe_01_R2_per_layer.pdf` — not yet generated.

---

## Probe 2 — Pilot-cell output denoising

### Status

Not run yet.

### Numbers

| SNR (dB) | LS MSE | Model MSE | Denoising gain (dB) |
|---:|---:|---:|---:|
|  5 | — | — | — |
| 10 | — | — | — |
| 15 | — | — | — |
| 25 | — | — | — |

### Did the hypothesis hold?

TBD.

### Figures

- `figures/probe_02_pilot_denoising_gain.pdf` — not yet generated.

---

## Probe 3 — Per-cell ablation (the central test)

### Status

Not run yet.

### Numbers

| SNR (dB) | Learned (E01) | Learned ablated (diagonal attention) | Oracle | LS+interp |
|---:|---:|---:|---:|---:|
|  0.0 | 2.12e-1 | — | 2.10e-1 | 2.89e-1 |
| 10.0 | 4.19e-2 | — | 4.32e-2 | 7.02e-2 |
| 20.0 | 5.16e-3 | — | 5.82e-3 | 1.15e-2 |
| 25.0 | 1.23e-3 | — | 1.78e-3 | 5.07e-3 |

(Full 11-point table will be populated on run.)

### Did the hypothesis hold?

TBD.

### Figures

- `figures/probe_03_per_cell_ablation_ber.pdf` — not yet generated.

---

## Probe 4 — Perturbation kernel

### Status

Not run yet.

### Numbers

- Frequency kernel half-width (subcarriers, at 50 % of peak): TBD.
- Time kernel variation (max/min ratio across symbols): TBD.
- Per-position variation: TBD.

### Did the hypothesis hold?

TBD.

### Figures

- `figures/probe_04_perturbation_freq.pdf` — not yet generated.
- `figures/probe_04_perturbation_time.pdf` — not yet generated.
- `figures/probe_04_perturbation_2d.pdf` — not yet generated.

---

## Cross-probe story

*To be written after all four probes complete.* What coherent picture
do the four probes paint of what the E01 learned receiver is actually
doing internally? Which parts of the original "joint-grid attention
does MMSE + smoothing" story survive? Which get refined or refuted?

## Caveats

- All probes use the same single checkpoint. Results are a readout of
  one trained model, not of the architecture class in general.
- Linear probe R² is a lower bound on the information present in the
  residual stream — the model could represent `H` non-linearly and
  the linear probe would miss it. If R² is low, consider an MLP
  probe as a follow-up.
- Per-cell ablation (Probe 3) monkey-patches attention at inference
  time. The model was never trained with diagonal attention, so its
  behavior under ablation is not a "receiver that learned diagonal
  attention" — it's "a joint-grid receiver deprived of its
  attention partners at test time." Different question.
- Perturbation study assumes local linearity of the network around
  the operating point. True only for small `δ`; we use
  `|δ| = 0.14 ≈ noise std at 15 dB`.

## Follow-ups

- If Probe 1 R² is low, run an MLP probe as follow-up to rule out
  non-linear representations.
- If Probe 3 ablation doesn't degrade BER, investigate whether the
  model learned something alphabet-specific (probe: swap QPSK for
  random complex symbols at test time and see if BER explodes).
- If the learned kernel (Probe 4) is markedly different from a 2D
  Wiener filter, explicitly construct the Wiener baseline and
  compare (E04-style follow-up).
- Regardless of outcome, the probing infrastructure here will get
  reused for E09 (linear probes on the self-supervised encoder).

## Artifacts

- **Checkpoint**: `makarkul/cognitive-rf-E01/best.pt` (unchanged;
  we only read it).
- **Code**: `experiments/E06_probes_on_e01/`.
- **Figures**: `figures/probe_*.pdf` (to be generated).
- **W&B run**: none (deterministic probes, not training).
