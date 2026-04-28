# Hypothesis — E00: Sinusoid recovery with a tiny transformer

*Pre-program experiment, recorded retrospectively for completeness.
The original E00 work (Apr 2025 → Apr 2026) predates the cognitive-RF
notebook scaffold; this file reconstructs the falsifiable claims that the
original work was implicitly testing, plus the fresh LM-vs-denoiser
side-experiment commits added in W18 (Apr 2026).*

## Question

A pure sinusoid is exactly AR(2):
`x[n] = 2·cos(ω₀)·x[n-1] − x[n-2]`. Classical DSP fits two taps; Yule-Walker
returns them analytically. Two questions follow:

1. **Implicit AR-discovery.** Does a transformer trained on
   next-sample regression over noisy sinusoids of random
   `(freq, amplitude, phase)` *implicitly* discover the AR(2) structure
   in its attention pattern?
2. **Internal frequency representation.** Does the same trained model
   represent the sinusoid's frequency in its hidden state, in a form
   readable by a linear probe?

A follow-up question added in 2026 (after the original results were in):

3. **Supervision boundary.** Does that AR-discovery + frequency
   representation depend on the model being supervised against the
   *clean* signal, or does it survive when the only target available is
   the next *noisy* sample (true LM-style training)? And under which
   noise distributions does the equivalence hold?

## Prediction

1. **Attention shows peaks at lag 1 and lag 2** in the last layer's
   averaged-over-heads lag profile, across test frequencies. A
   random-init control shows no such structure.
2. **Linear ridge probe on the layer-2 mean-pooled hidden state recovers
   sinusoid frequency with R² > 0.9**. Random-init control sits near 0.
3. **For i.i.d. zero-mean amplitude/phase noise, LM-style training
   matches supervised denoising** on val SNR-gain (within run-to-run
   noise ≈ 0.3 dB) and on frequency-probe R². For **structured noise**
   (AR(1), Wiener phase walk, DC offset) the LM model's MSE-vs-clean is
   strictly worse than the denoiser's, with a gap that grows with the
   strength of the structured component.

## Why

For (1) and (2) — the AR(2) recurrence implies the conditional
expectation `E[x[n+1] | x[n], x[n-1]]` is linear in the two preceding
samples. Self-attention at lag 1 and 2 is the lowest-cost mechanism a
causal transformer has to express that map; if MSE pressure is enough to
discover it, the attention pattern is the diagnostic. Frequency must be
encoded somewhere to allow `2·cos(ω₀)` to be the AR(1) coefficient;
mean-pooled hidden states are the natural candidate.

For (3) — decompose the LM-optimal predictor:
`E[noisy[n+1] | history] = E[clean[n+1] | history] + E[ε[n+1] | history]`.
The two regimes have the same minimizer iff the second term is zero. That
holds for noise that is zero-mean and independent of signal/history; it
fails for AR-correlated noise, integrated noise (Wiener phase walk), and
deterministic offsets. See
[discussion 009](../../01_discussions/009_lm_vs_denoiser_equivalence.md)
for the full argument.

## Would change our plan if

- **AR(2) attention signature absent:** the transformer is solving the
  task by some other mechanism (e.g., the FFN doing all the work). Worth
  a follow-up that ablates the FFN.
- **Frequency probe R² stays ≈ 0:** frequency is encoded non-linearly
  (or not at all). Run an MLP probe; if still 0, the model is doing
  pure pattern-matching, not representation-learning.
- **LM ≡ denoiser on structured noise:** would invalidate the
  conditional-expectation argument and force a re-think of the
  premise behind E07's masked-RE recipe.
- **LM ≪ denoiser even on `iid`:** would suggest LM-style training has
  an optimization handicap (loss-floor noise) at this scale that
  matters even for AWGN — a real concern for E07.

## Dependencies

- `experiments/E00_sinusoid_recovery/{ts_transformer.py, train.py,
  signal_dataset.py, transformer_blocks.py}` — the original training
  pipeline.
- `experiments/E00_sinusoid_recovery/{analyze.py, probes.py}` — original
  diagnostics that establish (1) and (2).
- `experiments/E00_sinusoid_recovery/train_lm_vs_denoiser.py` — fresh
  side experiment for (3), with `--noise-mode` sweep covering
  `iid / ar1_coloured / wiener_phase / dc_offset`.

## Estimated cost

- Compute: 0 GPU-hours. ~0.5–1 h CPU for the original training; ~25 min
  CPU for the four-mode LM-vs-denoiser sweep.
- Engineer time: original E00 is sunk; LM-vs-denoiser sweep + writeup
  ≈ 0.5 day.
