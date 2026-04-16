# How was the E01 OFDM receiver trained?

Addresses three common misconceptions:
1. It's not a spectrogram input.
2. It's not next-sample prediction (no autoregression).
3. It's supervised, with labels the simulator generates.

## Training loop (on-the-fly, no saved dataset)

For each batch of 32 subframes:

1. **Draw a random SNR** uniformly in [0, 25] dB. One SNR per subframe, not per batch.
2. **Draw a fresh EPA channel** per subframe (block-fading: the channel is constant across all 14 symbols of that subframe, but differs between subframes).
3. **Build the TX grid:**
   - Random bits → QPSK → place on 300 active SCs × 14 symbols = 4200 REs.
   - Overwrite the 200 pilot REs with `(1+j)/√2`.
4. **Simulate:**
   - Multiply each SC by `H[k]` (block-fading, so same `H` across all 14 symbols).
   - IFFT + CP → time domain.
   - Add complex AWGN at the drawn SNR.
   - Strip CP + FFT → post-FFT grid `y[t,k]`, shape (14, 300), complex.
5. **Pack model input** of shape `(B=32, 14, 300, 3)`:
   - Channel 0: `Re(y[t,k])`
   - Channel 1: `Im(y[t,k])`
   - Channel 2: `is_pilot[t,k]` (binary)
6. **Forward pass.** `Linear(3, 128)` embeds each RE → factorized 2D positional embedding → flatten to 4200 tokens → 4 transformer blocks (bidirectional, no causal mask) → `Linear(128, 2)` per RE.
7. **Loss.** BCE on the 4000 *data* REs only. Pilot REs are masked out of the loss. Targets = the true bits.
8. **Backprop, AdamW, step.** 30,000 times.

## What this is *not*

- **Not spectrogram input.** The model never sees raw IQ. The FFT lives in the simulator; the model sees the post-FFT RE grid.
- **Not a fixed dataset.** Every batch is freshly synthesized. The model sees ~32 × 30000 ≈ 1M unique subframes, each with its own channel, noise, and bits. No repeats.
- **Not per-cell.** All 4200 REs attend to each other. That's why the model can exploit frequency-domain smoothness and time-domain coherence.
- **Not aware of QPSK directly.** We don't tell the model "these are QPSK symbols." It infers the finite-alphabet structure from the BCE supervision.
- **No channel-estimation loss.** We don't supervise `Ĥ`. The model is free to estimate H internally or skip that step entirely. We only score bits.
- **Not autoregressive.** BERT-style encoder, not GPT-style decoder. All REs processed simultaneously, bidirectionally. Contrast with the sinusoid case, which *was* autoregressive (next-sample prediction).

## Why the pilot mask matters

The `is_pilot` channel tells the model which REs contain a known reference. Combined with the fact that the pilot value is the same every time (the model sees the same constant at flagged positions across millions of subframes), the model can learn:

> "Cells flagged as pilot contain a known reference. I can compare the *received* value at those cells to the known transmit value to infer the channel — and propagate that inference to nearby unflagged cells."

That's channel estimation, learned implicitly. No explicit supervision required.

If we ever wanted configurable pilot values (like real DMRS scrambling), we'd need two extra input channels for `(pilot_expected_I, pilot_expected_Q)`, zero at data cells. We don't currently — pilots are fixed.

## Simulator vs real receiver

- **Training:** simulator generates `(rx_grid, tx_bits)`; model sees only `rx_grid`; loop computes BCE against `tx_bits`.
- **Inference:** just `model(rx_grid)`. No labels, no loss. That's what a real RX does.

The sim-to-real gap — whether a model trained on a synthetic EPA distribution works on a real LTE capture — is not tested yet. See [Q14 in open_questions](../04_open_questions.md) and the planned impairment sweep (E05).

## Contrast with phase 4 (future)

In E15 (spectrogram-tile ViT), the input will be raw IQ samples (or a spectrogram of raw IQ) *before* FFT. The model will have to find symbol boundaries, do its own FFT-equivalent, and demodulate. Much harder. E01 is the warm-up proving the transformer can equalize a given grid; E15 asks whether it can find and tokenize the grid from a raw capture.
