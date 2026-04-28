# Low-SNR Experiments and Parameter-Tuning Observations

This document summarizes empirical observations on how the tiny transformer
in this folder behaves as the input SNR drops, and what architectural levers
matter most when you need to recover usable signal from heavy noise.

All experiments use the next-sample-prediction setup from `train.py`, with
randomized sinusoids (freq 1–20 Hz, amplitude 0.5–2.0, phase 0–2π) at
`fs = 100 Hz` and a context length of 128 (unless scaled).

Scripts:
- [`snr_sweep.py`](snr_sweep.py) — evaluate a trained model across noise levels
- [`retrain_lowsnr.py`](retrain_lowsnr.py) — retrain at hard noise, base vs scaled

---

## 1. How the original model degrades as noise increases

Model: `TS_TRANSFORMER_CONFIG` (context 128, emb 32, 2 heads, 2 layers, 29 K params),
trained at `amp_std=0.2, phase_std=0.1` for 30 epochs.

| `amp_std` | `phase_std` | Input SNR (dB) | Output SNR (dB) | Gain (dB) | Val MSE |
|---|---|---|---|---|---|
| 0.10 | 0.05 | 3.05 | 17.31 | **+14.26** | 0.017 |
| 0.20 | 0.10 | 2.85 | 14.65 | **+11.80** | 0.031 |
| 0.40 | 0.20 | 2.14 | 9.61 | **+7.47** | 0.098 |
| 0.60 | 0.30 | 1.17 | 5.90 | **+4.72** | 0.231 |
| 0.80 | 0.40 | 0.12 | 3.26 | **+3.14** | 0.424 |
| 1.00 | 0.50 | -0.93 | 1.37 | **+2.30** | 0.654 |

**Observations**

- Gain collapses from +11.8 dB at the training noise to +2.3 dB at 5× noise.
- Two effects are entangled here: (i) the harder physical problem, and
  (ii) the train/test distribution mismatch.

---

## 2. Retraining at hard noise — same arch vs scaled arch

Hard setting: `amp_std=0.8, phase_std=0.4` (input SNR ≈ 0 dB). Trained for
10 epochs with 4000 sequences/epoch, AdamW, lr 1e-3.

| Scenario | Arch (ctx / emb / heads / layers) | Params | Train time | Out SNR | Gain | Val MSE |
|---|---|---|---|---|---|---|
| Original (no retrain) | 128 / 32 / 2 / 2 | 29 K | — | 3.26 dB | **+3.14** | 0.424 |
| (A) Base, retrained | 128 / 32 / 2 / 2 | 29 K | 144 s | 3.81 dB | **+3.69** | 0.368 |
| (B) Scaled, retrained | 256 / 72 / 4 / 3 | 208 K | 1213 s | 5.85 dB | **+6.00** | 0.220 |

Training trajectory (val SNR improvement per epoch):

| Epoch | (A) Base | (B) Scaled |
|---|---|---|
| 1 | +2.0 | +2.1 |
| 3 | +2.2 | +2.6 |
| 5 | +2.9 | +3.2 |
| 7 | +3.0 | +4.4 |
| 9 | +3.5 | +5.5 |
| 10 | +3.7 | +5.9 |

**Observations**

- **Retraining alone barely helps** (+3.14 → +3.69 dB). The base architecture
  is saturated; a pure AR(2) predictor has little room to average noise.
- **Capacity buys real gain**: 7× parameters and 2× context → +2.3 dB extra.
  The longer window is the dominant lever.
- (B) was still climbing at epoch 10 — with more training it likely reaches
  +7 to +8 dB.
- Neither config recovers the +11.8 dB gain seen at easy noise. Heavy phase
  noise makes the instantaneous frequency wobble, which is a genuinely
  harder signal-recovery problem.

---

## 3. Parameter-tweaking recommendations (DSP-grounded)

At lower SNR, **averaging more samples** is the dominant denoising mechanism.
The three DSP-anchored hyperparameters shift as follows:

| Parameter | Easy-noise rule | Low-SNR rule | Why |
|---|---|---|---|
| **`context_length`** | `≈ fs / f_min` | `≈ (fs / f_min) · k`, k ≈ 2–4 | Extra √k noise averaging |
| **`n_heads`** | = effective AR order | AR order + smoothing/MA heads | FIR-style low-pass averaging beyond AR |
| **`head_dim`** | `≈ 2 · log₂(ctx)` | same rule, scales with `ctx` | Keeps per-position discrimination under noise |
| **`n_layers`** | 2 | 3–4 | Iterative denoising refinement |
| **`emb_dim`** | falls out | falls out | `= n_heads · head_dim` |

Non-architectural levers (all three matter):

1. **Match training noise to deployment noise.** Out-of-distribution noise
   is the single largest source of degradation.
2. **Curriculum or mixed-σ training.** Sample noise from a range so the
   model is robust across a span of SNRs rather than specialized to one point.
3. **Train longer.** On CPU, (B) needed ~20 minutes for 10 epochs; a full
   30–40 epochs is feasible and typically adds 1–2 dB.

---

## 4. FFT-embedding front-end — a much stronger inductive bias

Replacing the scalar `Linear(1, emb_dim)` input projection with a
**causal sliding-window FFT** changes the picture dramatically. At every
position `t`, the token now carries the FFT of the last `W=32` samples
(real + imag bins), then projected to `emb_dim`. Everything downstream
(positional embedding, attention, FFN, head) is unchanged.

Implementation: [`ts_transformer_fft.py`](ts_transformer_fft.py).
Comparison driver: [`compare_fft.py`](compare_fft.py).

### 4a. Easy noise (`amp_std=0.2, phase_std=0.1`), 8 epochs, ~30 K params

| Variant | Params | Final MSE | Out SNR | Gain | Train time |
|---|---|---|---|---|---|
| Scalar `Linear(1, 32)` | 29,473 | 0.091 | 9.92 dB | **+7.31 dB** | 120 s |
| **FFT sliding window** | 30,529 | 0.021 | 16.39 dB | **+13.78 dB** | 113 s |

Convergence comparison (val SNR gain by training step):

| Step | Scalar | FFT |
|---|---|---|
| 0 | -4.6 | -3.6 |
| 100 | +0.8 | **+7.8** |
| 300 | +2.2 | +11.3 |
| 500 | +4.0 | +12.1 |
| 700 | +5.5 | +13.0 |
| 900 | +6.7 | +13.5 |

FFT reaches the scalar model's final result after just 100 steps — roughly
8× faster wall-clock to a given SNR target.

### 4b. Hard noise (`amp_std=0.8, phase_std=0.4`), 10 epochs, ~30 K params

| Variant | Params | Final MSE | Out SNR | Gain |
|---|---|---|---|---|
| Scalar `Linear(1, 32)` | 29,473 | 0.368 | 3.81 dB | **+3.69 dB** |
| **FFT sliding window** | 30,529 | 0.108 | 9.12 dB | **+9.00 dB** |

Cross-reference: the **scaled scalar model** from §2 — 7× larger (208 K params),
2× context, 4 heads, 3 layers — only reached **+6.0 dB** at hard noise.
The FFT model at the original size **beats it by +3 dB**.

### 4c. Why the FFT front-end helps so much

A pure sinusoid is a delta in the frequency domain. The FFT embedding
hands the transformer a **near-diagonal representation** of the task:

- Each token already encodes the spectrum of the last 32 samples
- Noisy input becomes a peak-plus-noise-floor in each token
- The transformer's job collapses to peak-tracking and bin-interpolation
- It no longer has to rediscover periodicity from raw time-domain samples

Equivalently, the FFT acts as **parallel matched filters** (one per bin)
before attention even runs — built-in processing gain.

### 4d. Trade-offs

| Aspect | Observation |
|---|---|
| Inference overhead | ~160 MACs/sample for the sliding FFT; negligible vs ~50 K MACs inside the blocks |
| Inductive bias | Strong — ideal for stationary/periodic signals; less safe for chirps or impulsive signals |
| Window size `W` | Classic STFT trade-off: large `W` = better frequency resolution, small `W` = better time resolution |
| Convergence | Reaches good performance in ~1 epoch; early-stopping friendly |
| Interpretability | Tokens are physically meaningful (FFT bins) — attention maps are easier to read |

### 4e. When to choose which embedding

- **FFT embedding** — band-limited / periodic / stationary signals. Huge sample-efficiency and SNR-gain win.
- **Scalar embedding** — general-purpose default when signal structure is unknown.
- **Learned `Conv1d(1, emb_dim, kernel_size=W)`** — middle ground: same compute as FFT, no frequency prior, model learns the filterbank.

---

## 5. Perspective vs. classical DSP

At input SNR ≈ 0 dB, a Kalman filter on the known sinusoid model typically
delivers **+8 to +12 dB** gain at ~30 MACs per output sample. The scaled
transformer in (B) reached **+6 dB** at millions of MACs per sample.

The transformer's value here is not efficiency — it is architectural
flexibility. The efficiency gap only closes when the signal model becomes
rich or uncertain enough that a compact classical estimator cannot be
written down (multi-tone, non-Gaussian noise, regime switches, chirps, etc.).

---

## 6. How to reproduce

```bash
# Evaluate the released model across a noise sweep
python snr_sweep.py --model-path ts_transformer.pth

# Retrain base and scaled configs at hard noise
python retrain_lowsnr.py --amp-std 0.8 --phase-std 0.4 --epochs 10

# Compare scalar-embedding vs FFT-embedding at matched param budget
python compare_fft.py --amp-std 0.2 --phase-std 0.1 --epochs 8
python compare_fft.py --amp-std 0.8 --phase-std 0.4 --epochs 10

# Side experiment: LM-style next-noisy-sample target vs supervised denoiser
python train_lm_vs_denoiser.py --epochs 25 --train-size 5000 --val-size 1000
```

Numbers will vary slightly run-to-run because the validation set is generated
on the fly.

---

## 7. LM-style vs denoiser side experiment

### Why this exists

The canonical training in `train.py` is a **supervised denoising
autoregressor**: input is noisy, but the MSE target at every position is the
*clean* next sample. That is not the same as the strict LLM-style setup,
where the only label the model ever sees is the next observed sample (here,
the next noisy sample). The phrase "next-sample prediction" can blur this
distinction; the experiment in
[`train_lm_vs_denoiser.py`](train_lm_vs_denoiser.py) makes it concrete.

| Variant | Input | Target | Loss |
|---|---|---|---|
| Denoiser (canonical) | `noisy[0..N-1]` | `clean[1..N]`  | MSE-vs-clean |
| LM-style              | `noisy[0..N-1]` | `noisy[1..N]`  | MSE-vs-noisy  |

Both variants train the **same 29 K-param model from the same seed and the
same data stream**; only the loss target differs. We then ask three
diagnostic questions of each trained model:

1. **Final denoising quality.** MSE of the model's output against the *clean*
   next sample (even when MSE-vs-clean is not the training loss), and the
   resulting SNR gain over a naive persistence baseline.
2. **AR(2) attention signature.** Does the last-layer attention show peaks at
   lag 1 and lag 2 — the AR(2) signature — under both variants?
3. **Frequency probe.** Can a linear ridge regression on each layer's mean
   hidden state predict the sinusoid's frequency? High R² says the model
   represents `f` internally; the random-init control sits near 0.

We also run a four-mode noise sweep (§7d below) to map the boundary at
which the LM-vs-denoiser equivalence holds.

### Setup

| Field | Value |
|---|---|
| Architecture | `TS_TRANSFORMER_CONFIG` (29 K params, ctx 128, emb 32, 2 heads, 2 layers) |
| Optimizer | AdamW, lr 1e-3, weight_decay 0.01 |
| Epochs | 25 |
| Train / val | 5000 / 1000 random sinusoids per epoch (deterministic per index) |
| Noise (i.i.d. baseline) | `amp_noise_std = 0.2`, `phase_noise_std = 0.1` |
| Seed | 42 (same init for both runs) |
| Hardware | Laptop CPU |

§7a–c below report the **i.i.d. baseline** (mode = `iid`); §7d reports the
four-mode sweep using the same architecture and training protocol.

### Results

#### 7a. Final denoising quality, i.i.d. mode (held-out 1000-sample validation set)

| Variant | Val MSE-vs-clean | Val MSE-vs-noisy | SNR_in (dB) | SNR_out (dB) | SNR gain (dB) |
|---|---:|---:|---:|---:|---:|
| Denoiser (target = clean) | 0.0437 | 0.0722 | 2.81 | 13.11 | **+10.30** |
| LM-style (target = noisy) | **0.0411** | **0.0696** | 2.81 | **13.37** | **+10.56** |

The two variants converge to **essentially the same denoising quality**.
The LM-style model is in fact ~0.3 dB *better* on the held-out set in this
run — a difference that is within run-to-run variation given a single seed
and a single 1000-sample validation set, but the headline is that the LM
target loses **nothing** observable here.

> Per-epoch curves: [`lm_vs_denoiser/iid/loss_curves.pdf`](lm_vs_denoiser/iid/loss_curves.pdf).
> Both curves track within ≤0.3 dB of each other across all 25 epochs.

#### 7b. Attention lag profile, i.i.d. mode

Last-layer attention, averaged over heads, evaluated on a clean sinusoid at
each of `f ∈ {3, 5, 10, 15} Hz`. The hypothesis going in was a "lag 1 + lag
2" AR(2) bar.

What the trained models actually show is more interesting: **the
attention weight peaks at integer multiples of the input period**
(`T = fs / f`), within the lags reachable inside the truncated 0–20 lag
window:

| Test freq | Period `T` (samples) | Visible peaks (lags) | Comment |
|---|---:|---|---|
| 3 Hz  | ≈ 33 | profile is nearly flat across 0–20 | first period sits beyond the displayed window |
| 5 Hz  | ≈ 20 | broad bump centered at 5–7 | first sub-period region |
| 10 Hz | ≈ 10 | sharp peak at 2–3, second cluster at 12–14 | one full period back |
| 15 Hz | ≈ 6.7 | sharp peaks at 2, 8, 14 | one, two, three periods back |

This is **period-aligned attention**, not a strict AR(2) "lag 1 + lag 2"
bar. The strict AR(2) recurrence `x[n] = 2cos(ω₀)·x[n-1] - x[n-2]` does
*not* require the model to attend at the period boundary — but apparently
the trained model does both: it learns a representation in which one
period back is a useful corroborating reference. With ~128-sample context
and frequencies in [1, 20] Hz, multiple periods always fit inside the
window, so attending to them is "free" capacity that gradient descent
fills.

The crucial point for the framing question is that **the denoiser and the
LM-style model produce visually indistinguishable lag profiles** at every
test frequency — same peak locations, same peak magnitudes, same
inter-peak structure. The pattern is being pulled out of the *input
dynamics*, not out of the supervisory signal.

> Figure: [`lm_vs_denoiser/iid/attention_lag_profile.pdf`](lm_vs_denoiser/iid/attention_lag_profile.pdf).

#### 7c. Linear frequency probe, i.i.d. mode

Ridge probe (closed-form, `λ = 1e-4`) on layer-wise mean-pooled hidden
states, trained on 1500 random sinusoids and evaluated on 400 held-out ones.
Higher R² = the layer's hidden state more linearly predicts the input
sinusoid's frequency.

| Layer | Denoiser R² | LM R² | RMSE denoiser (Hz) | RMSE LM (Hz) |
|---|---:|---:|---:|---:|
| layer_1 | 0.930 | 0.932 | 1.45 | 1.42 |
| layer_2 | 0.976 | 0.970 | 0.84 | 0.95 |

For reference, the canonical random-init control in `probes.py` reports
`R² ≈ 0` at every layer — so any reading well above 0 is structure created
by training. Both variants reach R² ≈ 0.97 at layer 2, with the denoiser
~0.005 higher.

> Figure: [`lm_vs_denoiser/iid/freq_probe.pdf`](lm_vs_denoiser/iid/freq_probe.pdf).

#### 7d. Noise-mode sweep — when does the LM ≡ denoiser equivalence break?

The i.i.d. baseline confirms the prediction
`E[noisy[n+1] | history] − E[clean[n+1] | history] = E[ε[n+1] | history] = 0`
when noise is zero-mean and independent. The follow-up question is how the
two regimes diverge once we relax that assumption.

`python train_lm_vs_denoiser.py --noise-mode sweep` runs the same protocol
under four noise distributions:

| Mode | Noise model | What it stresses |
|---|---|---|
| `iid` | i.i.d. amp + phase noise (baseline) | nothing — the optimal predictors coincide |
| `ar1_coloured` | amp noise replaced by AR(1), `α = 0.9`. Phase i.i.d. | future-noise is partially predictable |
| `wiener_phase` | phase noise integrated (Wiener walk). Amp i.i.d. | structured oscillator-style drift |
| `dc_offset` | per-sequence DC bias `μ ~ N(0, 0.5²)` | deterministic signal-independent bias |

Headline scalars from the 25-epoch sweep (`seed=42`, 5000 train / 1000 val
sinusoids per epoch, identical between regimes):

| Mode | denoiser MSE-clean | LM MSE-clean | denoiser SNR gain (dB) | LM SNR gain (dB) | Δ SNR gain (LM − den, dB) | Drift RMSE | denoiser layer-2 R² | LM layer-2 R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `iid`            | 0.0437 | 0.0411 | **+10.30** | **+10.56** | **+0.26** | 0.063 | 0.976 | 0.969 |
| `ar1_coloured`   | 0.0382 | 0.0393 | **+10.87** | **+10.75** | **−0.12** | 0.090 | 0.970 | 0.938 |
| `wiener_phase`   | 0.4171 | 0.4895 | **+2.75**  | **+2.05**  | **−0.70** | 0.318 | 0.917 | 0.980 |
| `dc_offset`      | 0.0615 | 0.3018 | **+10.68** | **+3.77**  | **−6.91** | 0.483 | 0.945 | 0.967 |

> Cross-mode summary figure: [`lm_vs_denoiser/noise_mode_sweep.pdf`](lm_vs_denoiser/noise_mode_sweep.pdf).
> Per-mode artifacts: `lm_vs_denoiser/<mode>/{loss_curves.pdf,
> attention_lag_profile.pdf, freq_probe.pdf, summary.json}`.

The predicted **drift-RMSE hierarchy** holds cleanly:
`iid (0.063) < ar1_coloured (0.090) < wiener_phase (0.318) < dc_offset (0.483)`.
The further the noise process is from i.i.d.-zero-mean, the more the LM
output diverges from the denoiser output — exactly what
`E[ε[t+1] | history]` predicts.

What each mode says:

- **`iid` — equivalence holds (predicted, confirmed).** Both regimes get
  ≈ +10.3 dB SNR gain. The +0.26 dB favouring LM is single-seed noise.
  This is the §7a–c regime.
- **`ar1_coloured` (α = 0.9) — small penalty.** The LM model gets
  Δ SNR-gain = −0.12 dB, almost in the noise. Both regimes get *better*
  absolute MSE-clean than under `iid` (0.038 vs 0.044) because AR(1)
  amplitude noise has lower effective variance after the model averages
  it over context. The cleaner signal of LM degradation is the
  freq-probe layer-2 R²: 0.970 (denoiser) vs 0.938 (LM). The LM is
  spending some representation capacity tracking the noise process.
- **`wiener_phase` — large absolute penalty for both, plus the cleanest
  Δ SNR-gain (−0.70 dB).** The integrated phase walk is an
  irreducibly hard task: the denoiser can recover only +2.75 dB
  (vs +10.3 under iid). The LM does worse still at +2.05 dB. *Both
  models are floored by the underlying problem*; the relative gap is
  the LM penalty.
  
  Surprising sub-result: the LM's layer-2 freq-probe R² (0.980) is
  **higher** than the denoiser's (0.917). Likely interpretation: the
  denoiser is forced to suppress the phase walk to recover the
  *nominal* sinusoid frequency, which costs it some linear frequency
  representation; the LM stays close to the input statistics where the
  *local instantaneous* frequency is more linearly readable. Same data,
  different learnt task — not "LM is better."
- **`dc_offset` — equivalence catastrophically broken (Δ SNR-gain
  −6.91 dB).** The denoiser learns to subtract the per-sequence bias
  and recovers full quality (+10.68 dB, matching `iid`). The LM regime
  collapses to +3.77 dB because the optimal LM target literally
  contains the predictable bias. This is the cleanest illustration in
  the sweep: when the noise has a deterministic component, the LM's
  Bayes-optimal output is biased by exactly that component, and there
  is no way for further training to fix it without changing the loss.

**Hierarchy summary by Δ SNR-gain magnitude:**
`iid (+0.26) ≈ ar1 (−0.12) << wiener_phase (−0.70) <<< dc_offset (−6.91)`.

### Interpretation

Under the **i.i.d. baseline** the LM-style and denoiser variants are
observationally **the same model**: matched SNR gain to within 0.3 dB,
matched frequency-probe R² to within 0.006, and visually matched
attention lag profiles. The "transformer discovers AR(2)" claim therefore
does not depend on access to clean references during training.

The mathematical reason: with i.i.d. zero-mean noise added to the signal,
the optimal predictor of the *next noisy sample* given the noisy history
decomposes as

```
E[noisy[t+1] | noisy[0..t]] = E[clean[t+1] | noisy[0..t]] + E[ε[t+1] | noisy[0..t]]
                            = E[clean[t+1] | noisy[0..t]] + 0
                            = the optimal denoiser
```

The two losses share the same minimiser; they differ only in their
irreducible loss floor (the LM-style target sits ≈ noise-variance higher,
which is exactly what the `MSE-vs-noisy` columns show — both variants land
near 0.07, the noise floor under our `amp_std=0.2, phase_std=0.1`
setting). Gradient descent therefore finds essentially the same fixed
point under both objectives.

The §7d sweep is the falsification test: relaxing the i.i.d.-zero-mean
condition should make the second term non-zero, biasing the LM optimum
away from the denoiser optimum. The smoke-test ordering already confirms
the qualitative prediction; the 25-epoch numbers fix the magnitude.

Practical consequence for the cognitive-RF program: **for any signal
recovery task where the noise is approximately independent of the signal
and zero-mean, the strict LM-style objective is a near-free substitute
for supervised denoising. As soon as the noise becomes structured
(coloured, integrated, biased, or signal-dependent), the LM regime will
silently inherit a bias from the conditional expectation of the future
noise.** This boundary is the key input to the E07 (masked-RE) recipe
design — see [`notebook/01_discussions/009_lm_vs_denoiser_equivalence.md`](../../notebook/01_discussions/009_lm_vs_denoiser_equivalence.md)
and [Q19 in open questions](../../notebook/04_open_questions.md).

### Caveats

- Single seed (42) per noise mode. The 0.3 dB delta in §7a is within
  plausible run-to-run variation; the conclusion there is "LM and
  denoiser are indistinguishable under i.i.d. noise," not "LM is
  strictly better." The four-mode hierarchy in §7d is robust to
  seed-level noise because the deltas at the structured-noise end are
  large.
- Both variants are trained on the *same* synthetic distribution per
  mode, so conclusions are about training-loss shape, not data
  distribution.
- The LM-style target's irreducible loss floor is the noise variance,
  not zero — `MSE-vs-noisy` cannot drop below ≈ noise power. We read
  the LM model's denoising quality from `MSE-vs-clean`, evaluated
  post-training, which it never optimised for directly. This is a fair
  readout under `iid` (where the optima coincide); it remains the right
  readout under structured-noise modes (where the optima diverge — that
  divergence is exactly what §7d measures).
- The frequency-probe random-init R² (≈ 0) is referenced from the
  canonical `probes.py` control rather than re-collected here.
- The §7d sweep uses one magnitude per impairment family
  (`α_AR1 = 0.9`, `σ_μ = 0.5`, `σ_φ = 0.1` for the Wiener walk). The
  shape of the equivalence boundary as a function of magnitude is a
  natural follow-up — see Q20 in
  [`notebook/04_open_questions.md`](../../notebook/04_open_questions.md).
