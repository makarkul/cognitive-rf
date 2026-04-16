# How does a learned receiver outperform perfect-CSI ZF?

**Context:** In E01, the learned transformer beats the perfect-CSI ZF oracle at high SNR — BER 7.4e-5 vs 2.4e-3 at 25 dB. This looks surprising on first read ("the oracle has more information!"). The resolution is that "perfect-CSI ZF" is not the Bayes-optimal receiver for this channel. It's optimal only under narrow assumptions that don't hold.

## Why the oracle is beatable

Three reasons the learned RX gets past ZF:

### 1. ZF amplifies noise on faded subcarriers

Per-SC ZF does `x̂[k] = y[k] / H[k]`. When `|H[k]|` is small (a deep fade), noise gets divided by a tiny number → that subcarrier contributes ~0.5 BER and dominates the average. This is the Rayleigh BER floor (~2.4e-3 for the oracle at 25 dB). MMSE (`H*/(|H|² + N₀)`) already beats ZF because it shrinks toward zero on faded bins instead of blowing up. The learned model can implement something MMSE-like — or better — for free.

### 2. ZF is per-cell. It throws away joint structure.

A faded subcarrier at index `k` has neighbors whose channels are highly correlated (EPA's ~400 ns delay spread → coherence bandwidth ~1 MHz → adjacent SCs see nearly the same `H`). The QPSK symbol on the faded bin is lost under ZF, but it can often be inferred from the pattern at neighbors because the constellation drift is smooth across frequency. Same story in time: the channel is block-faded across all 14 symbols, so whatever distortion the model sees in symbol 0 repeats in symbols 1–13. A 14×300 attention grid pools that evidence; per-cell ZF cannot.

### 3. Pilots are more informative than LS makes them

LS pilot estimation treats each pilot independently: `Ĥ = y_pilot / x_pilot`. That estimate has the full noise variance baked in. But pilots are on a regular grid and H is smooth → the correct estimator is a 2D Wiener filter over the pilot lattice. Linear interpolation is a cheap approximation; the transformer can learn the actual Wiener kernel from data. Even the perfect-CSI oracle doesn't benefit from this (it already has true H), but it's *still* stuck with ZF — so the learned RX's advantage here comes from **what it does with H**, not from estimating it better.

## The deeper framing

"Perfect-CSI ZF" is often cited as an upper bound, and it sounds like one, but it's only an upper bound over receivers that (a) use per-cell hard decisions and (b) don't exploit the signal's finite alphabet.

The *true* upper bound is joint ML:

```
argmax_{bits} p(y | bits, H)
```

over all 4000 data bits simultaneously, conditioned on the QPSK alphabet. That is computationally intractable (2^4000 hypotheses), but a 4-layer transformer can approximate it cheaply because it only has to learn the typical channel statistics, not enumerate hypotheses.

So the learned RX isn't breaking physics — it's closing the gap between ZF and ML-over-the-grid. That's exactly where the "beating the oracle" headroom lives.

## Caveat

32 subframes/SNR is noisy. The 30× number at 25 dB should tighten on a longer run (scheduled as E04). But the *direction* — learned < oracle at high SNR — is robust and matches the deep-RX literature.

## Implications for later work

- **Phase 2 (MREM pretraining):** if the supervised model already does something MMSE/ML-like, a self-supervised encoder pretrained on masked-RE reconstruction should produce a channel-aware embedding that a tiny linear head can turn into bits. The amount of labeled data needed for fine-tuning drops.
- **Phase 3 (blind):** the joint-grid intuition generalizes — without pilots, the model would have to find *some* known structure (CP, modulation alphabet, pilot-like periodicities) to anchor its channel estimate. The intuition for why the model could still win is the same: per-cell classical receivers throw away information that joint attention recovers.
