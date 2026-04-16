# Why 4 heads for the OFDM receiver?

Same DSP-anchored logic as the sinusoid case ([001](001_n_heads_rationale_sinusoid.md)), but for a 2-D resource grid.

## The design table

| Knob | Value | Why |
|---|---|---|
| `context_length` | 4200 REs | Fixed by the grid (14 symbols × 300 subcarriers). |
| `head_dim` | 32 | Shannon sizing: `~2·log₂(context_length) = 2·log₂(4200) ≈ 24` → round up to 32. |
| `n_heads` | 4 | Number of distinct relational patterns a head should specialize on (see below). |
| `d_model` | 128 | Falls out: `n_heads × head_dim = 4 × 32`. |

So 128 isn't picked directly — it's `n_heads × head_dim`, and the two factors come from "how many relational patterns matter" and "how big must each head be for Q·K dot products to stay discriminative over 4200 positions."

## The core fact about `n_heads`

One attention head produces *one softmax* — it picks *one* distribution over source tokens to pool from. A head can specialize on "look at pilots to my left/right in frequency," but can't *simultaneously* specialize on "look at pilots above/below in time" — that would need a different softmax. **One head = one relational pattern.** So `n_heads` = "how many distinct things do I want to look at in parallel per layer."

## What the receiver needs to do

To decode the bit at data RE `(t, k)`:

1. **Estimate `H[t, k]`.** Channel is block-faded, so the answer lives in the 200 pilot REs. Four useful ways to reach them — each wants its own head:

   | Head | Direction | What it does |
   |---|---|---|
   | 1 | Vertical (freq) | Interpolate H from nearest pilot SCs in the same symbol. Classical LS + linear-interp analog. |
   | 2 | Horizontal (time) | Interpolate H from nearest pilot symbols at a reachable SC. Nearest-time-pilot analog. |
   | 3 | Global (all pilots) | Average over all 200 pilots. Valid because block-fading → `H` is constant across the subframe, so pooling gives a low-variance coarse estimate. |
   | 4 | Local 2D neighborhood (data REs) | Look at nearby data REs. Their constellations cluster (shared `H`), so the head refines `Ĥ` via blind/decision-directed channel tracking. |

2. **Equalize + decide.** Linear op per RE; handled by the MLP, not attention.

These four attention kernels are genuinely different — sparse-on-freq-axis, sparse-on-time-axis, uniform-over-pilots, local-2D-neighborhood. You can't collapse them into one softmax without losing information.

## What about 2 heads? 8 heads?

- **2 heads:** would have to merge patterns (e.g., combine freq and time pilot lookups into one smeared softmax). Expect worse channel estimation.
- **8 heads at `d_model=128`:** `head_dim` drops to 16 — below the Shannon floor `2·log₂(4200) ≈ 24`. Worse per-head position discrimination.

The trade-off is baked in by `d_model = n_heads × head_dim`: more heads buys parallel patterns but spends head-dim discrimination budget.

## Honest caveat

This is a *design hypothesis*, not a proof. We didn't sweep 2/4/8/16 heads at matched params. 4 was the smallest number that (a) left `head_dim ≥ 24` at `d_model = 128` and (b) matched the count of relational patterns we could name. It worked. A sweep (scheduled as E02) will tighten or falsify the argument.

Contrast with the sinusoid case: there `n_heads = AR order = 2` was crisp because a pure sinusoid is *exactly* AR(2) — DSP literally hands you the number. For OFDM there's no single exact analog, so the count comes from cataloguing relational patterns.
