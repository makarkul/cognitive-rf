# Denoising autoregressor vs LM-style next-sample prediction

When this question came up in conversation:

> "In the sinusoid recovery experiment, does the model train only on the
> noisy sinusoid, or is it also fed the pure sinusoid? If the cost function
> uses the pure sinusoid, that defies the premise of LM training — it
> should just work on the input stream and not reference."

The reader was right that the canonical E00 setup is *not* LM-style training,
and the distinction matters for how we frame the cognitive-RF program (in
particular E07, masked-RE pretraining). This file captures the framing, the
condition under which the two regimes are equivalent, and what the
side-experiment in `experiments/E00_sinusoid_recovery/train_lm_vs_denoiser.py`
actually shows.

## The two regimes

| Variant | Input at position `t` | Target at position `t` | Loss |
|---|---|---|---|
| **Supervised denoising autoregressor** (canonical E00 `train.py`) | `noisy[0..t]` | `clean[t+1]` | MSE-vs-clean |
| **LM-style next-sample prediction** | `noisy[0..t]` | `noisy[t+1]` | MSE-vs-noisy |

Both share the same input stream and the same shift-by-one shape; only the
target differs. The denoiser regime requires a clean reference at training
time. The LM regime does not — it is genuinely self-supervised on the
observed stream alone, the way an LLM is trained on the next token.

## Condition for equivalence

Decompose the LM-optimal predictor:

```
E[noisy[t+1] | noisy[0..t]] = E[clean[t+1] | noisy[0..t]] + E[ε[t+1] | noisy[0..t]]
                                 └────────── denoiser optimum ───────────┘   └── drift ──┘
```

The two regimes have the **same minimizer** if and only if the drift term
`E[ε[t+1] | noisy[0..t]] = 0`. That holds when the noise process is

- zero-mean, **and**
- independent of the signal and of its own past (i.i.d.).

AWGN, ADC quantisation noise (approximately), thermal noise, and shot noise
at high counts all satisfy this. Coloured / temporally correlated noise,
signal-correlated noise, deterministic offsets, and structured impairments
(CFO, IQ imbalance, PA non-linearity, oscillator phase noise) do not — for
all of those, the LM model has predictive information about future noise and
will leak it into its output.

## What we actually observed (E00 side experiment)

Same architecture, same seed, same data stream, four noise modes, 25 epochs
each. Headline scalars from
`experiments/E00_sinusoid_recovery/lm_vs_denoiser/`:

| Noise mode | What it tests | Δ MSE-clean (LM − den) | Δ SNR gain (LM − den, dB) | Drift RMSE |
|---|---|---:|---:|---:|
| `iid` | i.i.d. amp + phase noise (the AWGN-equivalent case) | −0.003 | **+0.26** | 0.063 |
| `ar1_coloured` (α=0.9) | AR(1) coloured amp noise | +0.001 | **−0.12** | 0.090 |
| `wiener_phase` | Integrated phase walk (oscillator-style drift) | +0.072 | **−0.70** | 0.318 |
| `dc_offset` (σ_μ=0.5) | Per-sequence DC bias (hardware-style) | +0.240 | **−6.91** | 0.483 |

Full per-mode tables (with absolute MSE values, freq-probe R², etc.) live
at [`experiments/E00_sinusoid_recovery/RESULTS.md` §7](../../experiments/E00_sinusoid_recovery/RESULTS.md#7-lm-style-vs-denoiser-side-experiment).

The drift-RMSE hierarchy matches the prediction exactly:
`iid < ar1 < wiener_phase < dc_offset`. The most striking single
number is `dc_offset` Δ SNR-gain = **−6.91 dB**: when the noise has a
deterministic component the LM model can predict, the LM's
Bayes-optimal output is biased by exactly that component, and there
is no way for further training to fix it without changing the loss.

## Why this matters for the cognitive-RF program

The arc points toward E07 (masked-RE self-supervised pretraining) and E12
(blind / pilotless operation). Both lean on the LM-style assumption that the
model can be trained on the observed stream alone. This discussion sets the
operating envelope:

1. **Under the AWGN floor that all receivers share, LM-style and supervised
   denoising are equivalent in theory and indistinguishable in practice
   (E00 `iid` mode confirms this).** That part of the program is on solid
   ground.
2. **Each additional impairment we plan to test in E05 (CFO, IQ imbalance,
   PA non-linearity, phase noise) is exactly the kind that breaks the
   i.i.d.-zero-mean assumption.** A masked-RE pretraining recipe that
   ignores this will silently learn to predict structured impairments
   instead of denoising through them.
3. **The right design check before committing to an SSL-only training
   recipe** is to repeat the E00 noise-mode sweep at the OFDM-grid scale.
   If supervised-vs-SSL gap matches the impairment hierarchy seen here, we
   know SSL needs an auxiliary supervised signal at impairment surfaces it
   cannot see through.

## Implications captured elsewhere

- E00 `RESULTS.md` §7 — full numbers + interpretation of the four-mode sweep.
- E00 `README.md` — framing note distinguishing the two regimes.
- E07 hypothesis (when written) — pretraining recipe should include or
  exclude the supervised target depending on the impairment list at hand.
- `04_open_questions.md` Q19 — explicit follow-up: does the impairment
  hierarchy at OFDM scale match the sinusoid-scale ranking?

## Caveats

- Single seed, single architecture (29 K params), single signal class
  (one tone), one batch of impairment magnitudes. The hierarchy
  (`iid < ar1 < wiener_phase < dc_offset`) is robust at the qualitative
  level but the specific deltas are noise-floor sensitive.
- "LM-style" here means strict next-noisy-sample prediction. Masked-RE
  pretraining (BERT-style) is *not* identical: it predicts masked
  observations from unmasked observations, which has a different
  conditional-expectation structure. The qualitative conclusions probably
  carry over (any zero-mean i.i.d. noise still cancels; structured noise
  still leaks) but the constants will be different.
- The denoiser regime is only available where you can synthesise data
  with both clean and noisy components. For real OTA captures we don't
  have a clean reference — there the LM regime is the only option, and
  the impairment-hierarchy result tells us where to expect it to fail.
