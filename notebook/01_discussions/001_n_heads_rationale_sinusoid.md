# Why 2 heads for the sinusoid model?

From the E00 (pre-program) sinusoid recovery work. The rationale is clean in this case because a sinusoid *is* an exact AR(2) process.

## The DSP fact

A pure sinusoid of angular frequency `ω₀` satisfies the second-order linear recurrence:

```
x[n] = 2·cos(ω₀)·x[n-1] - x[n-2]
```

Given two past samples, the third is determined exactly. That is the *entire* generative model of a sinusoid. In DSP language: PACF cuts off at lag 2.

So a model that predicts the next sample from history needs only two effective "lag taps." Anything beyond lag 2 is redundant.

## The mapping to attention

Attention is effectively a learned lag selector: for each query position, the softmax picks which source positions to pool. One head = one softmax = one lag pattern.

- For AR(p), you want roughly p heads, so each can specialize on one lag.
- For AR(2) = sinusoid, `n_heads = 2`.

Post-hoc attention visualizations on the trained model confirmed: one head had its peak at lag 1, the other at lag 2. The DSP prediction held.

## The full design table

| Knob | Value | Why |
|---|---|---|
| `n_heads` | 2 | AR order. |
| `context_length` | 128 | ~`fs / f_min = 100 / 1 = 100` → rounded up to a power-of-2-ish window. Must cover the slowest period. |
| `head_dim` | 16 | Shannon sizing: `~2·log₂(context_length) = 2·log₂(128) = 14` → round up to 16. |
| `d_model` | 32 | Falls out: `n_heads × head_dim = 2 × 16`. |

`emb_dim` is not tuned; it's forced by the two DSP-anchored choices.

## Channel-coding intuition for `head_dim`

`head_dim` is the dimension of the Q·K dot products, which is the "codeword length" the attention uses to discriminate positions. The `1/√head_dim` scaling in attention is exactly matched-filter processing gain — longer codeword = better interference rejection between patterns.

Classical bounds (Shannon, spherical codes, BCH) all give roughly `d ≥ 2·log₂(N)` for reliable position discrimination among N items at per-dim SNR ~1. For our `context_length=128`, `d ≥ 14`. We use 16 — essentially at the Shannon floor with 1× safety factor.

Larger LLMs use 3–5× safety because text tasks are harder (ambiguity, long-range dependencies, semantic overload of positions). For AR(2) on sinusoids, 1× suffices.

## Depth rationale

We used 2 layers. Layer 1 locks onto `ω₀` (the dominant AR structure). Layer 2 models residual distortions — amplitude noise, phase noise — on top of the AR(2) carrier.

Does layer 1 also need 2 heads? Yes. Even within layer 1, the lags aren't summable — lag 1 and lag 2 carry different information. Merging them into one head would smear the AR(2) kernel. So `n_heads = 2` applies *per layer*, not just across the network.

## Why this matters for the rest of the program

The sinusoid case gave us a recipe: three DSP-anchored hyperparameters (`n_heads` from signal AR structure, `context_length` from slowest period, `head_dim` from Shannon) force `emb_dim` via multiplication. The OFDM case ([005](005_n_heads_rationale_ofdm.md)) applies the same logic, with "AR order" replaced by "count of distinct relational patterns on the resource grid."

This is the transferable framing: DSP tells you what structure exists in the signal, and the transformer's head/dim factorization is the right knob to dial.
