# The arc

This is the research notebook for a small-transformer cognitive wireless receiver. It is written as a story because that's how the work progresses: each experiment answers a question raised by the previous one.

## Where we started

A pure sinusoid is exactly AR(2): `x[n] = 2·cos(ω₀)·x[n-1] - x[n-2]`. Classical DSP fits two taps; Yule-Walker gives them exactly. Could a transformer doing next-sample prediction discover the same structure implicitly through attention?

**Sinusoid recovery (E00, pre-program)** — a 29k-param transformer trained on noisy sinusoids of random frequency/amplitude/phase, with MSE loss on the next clean sample. It learned. Attention heatmaps showed peaks at lags 1 and 2. A linear probe on the hidden state read out the sinusoid's frequency with R² > 0.9, while a random-init control failed. The representation was earned by training, not given by architecture.

Three DSP-anchored hyperparameters drove the design:
- `n_heads = AR order = 2` — one head per lag that matters
- `context_length ≈ N_FFT ≈ fs / f_min = 128` — cover the slowest period
- `head_dim ≈ 2·log₂(context_length) = 16` — Shannon floor for position discrimination

`emb_dim = n_heads × head_dim = 32` fell out. The model was not tuned; these three knobs constrained it.

## Where we are

**OFDM recovery, step 4a (E01)** — scale the same idea from a 1-D sinusoid to a 14×300 LTE-5 resource grid. 834k params, 4 heads, 4 layers, `d_model = 128`, supervised on bit labels (BCE). Trained on EPA multipath + AWGN + CRS-style pilots, 30k steps, ~$1.50 on a RunPod 4090.

Results (500-subframe eval, 4 M bits per SNR point): the learned receiver **matches the perfect-CSI ZF oracle within ±14% across 0–25 dB** and **beats EPA LS+interp by 1.4–4× across the full SNR range**. At high SNR it edges modestly below the oracle (1.23×10⁻³ vs 1.78×10⁻³ at 25 dB, a 1.45× reduction). An earlier 32-subframe eval had reported 30× below oracle at 25 dB; that was Poisson noise on ~19 error events — the corrected number is documented in the [E01 errata](02_experiments/E01_ofdm_supervised_rx/results.md).

The oracle gap at high SNR is the interesting piece. Perfect-CSI ZF floors out because it's a per-cell equalizer that amplifies noise on faded subcarriers. The transformer doesn't. It pools evidence across the joint time-frequency grid — frequency-domain smoothness, time-domain block-fading coherence, and the known pilot lattice — and does something MMSE-like plus soft-decoding-like in one shot. The gap is a statistically significant but modest ~30–45% BER reduction at 20–25 dB, not the 30× originally reported. See [Discussion 002: ZF vs learned](01_discussions/002_zf_vs_learned.md).

## Where we're going

The target is a **cognitive wireless receiver**: passively listens, understands its RF environment, and adapts its modem to suit. The four-phase program is:

| Phase | Months | Focus | Deliverable |
|---|---|---|---|
| 1 | 1–3 | Architecture & validation: depth/heads ablation, multi-head receiver, impairment sweep | internal report |
| 2 | 4–6 | Self-supervised pretraining: masked-RE modeling, label-efficiency, interpretability probes | paper candidate |
| 3 | 7–9 | Blind / pilotless operation: CFO + CIR from raw IQ, differentiable signal renderer | paper candidate |
| 4 | 10–12 | Standard-agnostic receiver: spectrogram-tile ViT, multi-standard corpus | program summary |

See [03_results_index.md](03_results_index.md) for the per-experiment plan and [04_open_questions.md](04_open_questions.md) for the running list of unknowns.

## How to read this notebook

- **Experiments** are the primary record. Each has `hypothesis.md`, `method.md`, `results.md`, and `figures/`. `hypothesis.md` is written before the experiment runs.
- **Discussions** capture reasoning and decisions that don't fit in a single experiment — design rationale, analogies, negative results from thought-experiments. They are distilled from working conversations.
- **Paper notes** are our distillation of prior art. One file per paper.
- **Logs** are Friday one-pagers. They're the glue that connects experiments in time.

The notebook is append-only during a phase and gets a light editing pass at each phase boundary.
