# What do "oracle" and "LS+interp" mean in the E01 results?

Two classical (non-learned) baselines. Both do zero-forcing equalization and hard-slice QPSK; they differ only in how they obtain the channel estimate `Ĥ`.

## EPA oracle (perfect CSI)

Also called the *genie receiver*. We cheat and hand the receiver the true complex channel response `H[k]` straight from the simulator. No estimation error. Then ZF + slice:

```
x̂[k] = y[k] / H[k]
bits = qpsk_slice(x̂[k])
```

This is the standard "upper bound for a per-cell ZF receiver" — how well you'd do *if* your channel estimator were perfect. A real receiver can never reach this because no one hands you `H`.

## EPA LS + interp (realistic)

What a real receiver actually does:

1. **LS** (least squares) at pilot cells: at the 200 pilot positions, compute `Ĥ = y_pilot / x_pilot`. Since we know the pilot symbol (fixed `(1+j)/√2`), this gives a noisy estimate of `H` at those cells.
2. **Interp** (interpolation) to fill in the other 4000 cells:
   - Linear interpolation across frequency (between every-6th-SC pilots).
   - Nearest-neighbor across time.
3. Same ZF + slice as the oracle, using `Ĥ` instead of `H`.

This is textbook OFDM and roughly what real LTE hardware does.

## Reading the gap

| Pair | What the gap shows |
|---|---|
| LS+interp vs oracle | The channel-estimator penalty (estimator residual). |
| Learned vs LS+interp | What the learned RX gains over the practical baseline. |
| Learned vs oracle | What the learned RX gains over *any* per-cell ZF, including an idealized one. |

In E01:
- Mid-SNR (15 dB): oracle 1.16e-2, LS+interp 2.15e-2, learned 1.13e-2. Learned ≈ oracle; both 2× better than LS+interp. The learned RX closed the estimator gap.
- High-SNR (25 dB): oracle 2.4e-3, LS+interp 5.2e-3, learned 7.4e-5. Learned beats both. The oracle gap is real (discussed in [002](002_zf_vs_learned.md)).

## What the learned RX is given as input

Importantly, the learned RX is **not** given the true H. It's given:
- `Re(y[t,k])`, `Im(y[t,k])` — the same received grid as LS+interp.
- `is_pilot[t,k]` — a binary mask flagging pilot cells.

The pilot *value* is not in the input; the model memorizes it from seeing the constant `(1+j)/√2` at flagged positions across millions of training samples.

So the fair comparison is learned vs LS+interp — both have identical information access. Beating the oracle on top of that is the interesting result and is discussed in [002_zf_vs_learned.md](002_zf_vs_learned.md).
