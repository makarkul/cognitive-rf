# Hypothesis — E06: Interpretability probes on the E01 learned receiver

*Written BEFORE any probe is run. Do not edit after results come in —
add a follow-up note in `results.md` instead.*

## Question

The E01 learned receiver matches the perfect-CSI oracle within ±14 %
across 0–25 dB and edges 1.45× below oracle at 25 dB. We have a
plausible story for *why* (joint-grid attention does something
MMSE-like plus a learned smoothing kernel), but no direct evidence
for that story. **What is the model actually doing internally?**

Four probes, each targeting a specific piece of the story.

---

## Probe 1 — Linear probe for H[k]

### Question
Does the residual stream implicitly carry a channel estimate that a
linear head can decode?

### Prediction
Yes. At pilot REs the R² of a linear map from `hidden[t, k, :]` to
`H_true[t, k]` should reach **R² ≥ 0.95 by layer 2** and saturate by
layer 3. At data REs R² should be lower in the early layers (the
model hasn't propagated pilot evidence yet) and rise to **R² ≥ 0.80
by the final layer**.

### Why
- LS+interp achieves R² ≈ 0.85 on data REs (rough estimate from its
  BER ratio vs oracle); the model must do at least this well to match
  oracle.
- Layer 0 is pre-attention — it only sees `(Re, Im, is_pilot)` per
  RE, so R² on data cells must be low at layer 0 (no pooling yet).
- Layers 2–3 are where time + freq pooling completes; that's where
  data-RE R² should climb.

### Would change our plan if
- If pilot-RE **R² stays low even at layer 3**: the model isn't
  representing H explicitly in the residual stream. It might be
  operating in a different basis (e.g., directly emitting bit logits
  without ever naming H). That's interesting but harder to publish.
- If data-RE **R² is already high at layer 0**: the signal is in the
  input itself and attention is doing almost nothing. Would suggest
  we're over-parameterized.
- If **R² climbs monotonically through all 4 layers**: natural story
  of progressive channel-estimate refinement — simplest interpretive
  arc.

---

## Probe 2 — Pilot-cell output denoising

### Question
Does the model produce a cleaner estimate of H at pilot cells than
raw LS division `y_pilot / x_pilot`?

### Prediction
Yes. At pilot cells, compare:

- **LS MSE**: `MSE(y_pilot / x_pilot, H_true)` — the raw per-cell
  estimator.
- **Model MSE**: `MSE(model_internal_H_estimate, H_true)` using the
  linear head from Probe 1.

Prediction: **model MSE < LS MSE by at least 3 dB at SNR ≤ 15 dB**
and **≥ 6 dB at SNR = 25 dB**. The gap grows at higher SNR because
that's where smoothing across pilots starts beating per-cell noise.

### Why
- Raw LS has variance ∝ `N₀ / |x_pilot|²`, distributed independently
  across pilots.
- A 2D-Wiener-like smoother pooling evidence across 200 pilots can
  reduce that variance by a factor of up to 200 (in the limit of fully
  correlated H).
- The true gain is bounded by the effective rank of the pilot-lattice
  channel covariance — which for EPA at LTE-5 is maybe 20–40.
- So 3–6 dB improvement is a plausible middle-ground.

### Would change our plan if
- If **model MSE ≈ LS MSE**: the model isn't doing pilot smoothing.
  It must be getting its mid/high-SNR advantage some other way
  (maybe soft-decoding the alphabet). Re-examine Probes 3 and 4.
- If **model MSE < LS MSE by more than 10 dB**: the model is doing
  something closer to a full 2D Wiener filter. Worth reporting and
  comparing to a classical 2D Wiener baseline (E04-style follow-up).

---

## Probe 3 — Per-cell ablation (the central test)

### Question
Is the high-SNR gap over oracle actually caused by joint time-freq
attention?

### Prediction
Yes. If we mask attention so every RE can only attend to itself
(diagonal attention mask), the 25-dB BER should degrade from **1.23e-3
back up to ≈ 1.78e-3 (oracle level) or worse**. The degradation at
20 dB should be smaller (from 5.16e-3 → ≈ 5.82e-3); at 10 dB
negligible (already noise-limited).

### Why
- The "beats oracle" story is: the oracle is bounded by per-cell ZF
  with hard slicing; the learned RX beats it by pooling across the
  grid. If you remove the pooling, the model can only do per-cell
  decisions, which is exactly what the oracle does.
- If ablation drops BER to oracle level, the bound was joint-grid.
- If ablation barely hurts BER, the advantage was something else
  (soft alphabet decoding, better pilot-value memorization, etc.).

### Would change our plan if
- If **BER@25 dB climbs to or past 1.78e-3**: confirms joint-grid
  story. We can publish the E01 result with a mechanistic explanation.
- If **BER@25 dB stays below 1.5e-3 even under per-cell attention**:
  the model is doing *something* else we don't understand. This
  would be the most surprising result; follow-up is to probe the
  learned non-linearity (alphabet-aware slicing?).
- If **BER@25 dB climbs to 5e-3 or worse**: joint-grid attention was
  doing even more than we thought — it was also doing the equivalent
  of channel estimation from scratch. Would mean the model can't even
  reach oracle without the grid. Worth reporting.

---

## Probe 4 — Perturbation kernel

### Question
What does the learned frequency-domain smoothing kernel look like?
Specifically: how far (in subcarriers) does a perturbation at one RE
propagate in the output?

### Prediction
The learned kernel should look approximately like a truncated
**sinc-shaped** (Wiener) kernel in the frequency direction, with:

- **Peak at `d=0`** (same cell).
- **Significant weight out to `|d| ≈ 6 subcarriers`** (one pilot
  period) in frequency.
- **Small but non-zero weight beyond** — probably decaying like
  `1 / d` rather than sharply truncating.
- In the **time direction**, nearly flat across all 14 symbols
  (EPA is block-faded, so time-averaging gives ~14× gain).

### Why
- The true EPA coherence bandwidth is ~1 MHz, which is ~66
  subcarriers at LTE-5's 15 kHz spacing. So the channel is smooth
  across tens of SCs.
- A Wiener filter over 200 pilots on a 300-SC grid would naturally
  produce a smoothing kernel with main lobe ~6 SCs wide (the pilot
  spacing).
- Block-fading in time means time-averaging is the right move;
  expect a nearly-flat kernel along the symbol axis.

### Would change our plan if
- If kernel is **much narrower than predicted** (e.g., only ±1 SC):
  the model isn't exploiting the full coherence bandwidth. We can
  probably get better BER with a slightly larger context or more
  training.
- If kernel is **wider than predicted** (e.g., ±30 SCs): the model
  has learned a richer long-range dependency, possibly exploiting
  alphabet structure. Interesting but unexpected.
- If **time direction is non-flat**: the model has learned something
  we didn't encode (maybe slow Doppler). Worth investigating but
  unexpected since our EPA generator is block-faded, not Doppler-
  modulated.

---

## Dependencies

- E01 best.pt checkpoint (local + HF `makarkul/cognitive-rf-E01`).
- `ofdm_recovery/dataset.py` for on-the-fly subframe generation with
  known true `H`.
- No GPU required. All four probes run on 500–2000 subframes; laptop
  CPU is fine.
- No new data generation infrastructure.

## Estimated cost

- **Compute**: 0 GPU-hours. ~2 hours total CPU time for all four
  probes combined.
- **Engineer time**: ~1–2 days including writeup, figures, and
  committing hypotheses/results separately per our process rules.
- **Budget**: $0.
