# Method — E01: OFDM supervised receiver on LTE-5 EPA

## Code

- Entry points (from the original worktree, to be ported into `experiments/E01_ofdm_supervised_rx/`):
  - Training: `ofdm_recovery/train.py`
  - Evaluation: `ofdm_recovery/evaluate.py`
- Data generation: `ofdm_recovery/dataset.py`, `ofdm_signal.py`, `channel.py`, `pilots.py`
- Model: `ofdm_recovery/model.py`
- Checkpoint: `checkpoints/best.pt` (saved at step 26000, best val_BER 4.33e-2)

## Data

- **Source:** synthetic, on-the-fly. No saved dataset.
- **Signal:** LTE-5 MHz FDD downlink, 7.68 MHz sample rate, N_FFT=512, 300 active subcarriers, normal CP.
- **Modulation:** uncoded QPSK, Gray-mapped.
- **Pilots:** CRS-style. Symbols {0, 4, 7, 11}, every 6th active SC, fixed value `(1+j)/√2`. 200 pilots + 4000 data cells per subframe.
- **Channel:** 3GPP EPA, 7 taps (delays 0/30/70/90/110/190/410 ns, powers 0/-1/-2/-3/-8/-17.2/-20.8 dB). Block-fading Rayleigh, E[|H|²]=1 per SC on average. Fresh channel realization per subframe.
- **Noise:** complex AWGN scaled so post-FFT per-active-SC SNR = drawn value.
- **SNR curriculum:** uniform in [0, 25] dB, one per subframe.

## Model

- Architecture: 4 transformer encoder blocks, bidirectional (no causal mask).
- Input: `(B, 14, 300, 3)` tensor. Last dim = `(Re(y), Im(y), is_pilot_flag)` per RE.
- Embedding: `Linear(3, 128)` per RE, then factorized 2D positional embedding (14-way symbol + 300-way subcarrier, both `d_model=128`).
- Attention: 4 heads, `head_dim=32`, pre-norm.
- FFN: `d_ff=512`, GELU.
- Output: `Linear(128, 2)` per RE → 2 bit logits at data REs.
- **Parameter count: 834,306.**

## Training

- Optimizer: AdamW, lr=3e-4, weight_decay=0.01.
- Schedule: 500-step linear warmup, then cosine decay to 0 at step 30000.
- Batch: 32 subframes.
- Loss: BCE with logits, on the 4000 data REs only (pilot REs masked out of the loss).
- Steps: 30000.
- Hardware: 1× RTX 4090 on RunPod Community (~$0.40/hr).
- Wallclock: ~3.5 hours.
- Data gen: ~24 ms/batch (~5% of step time). Bottleneck is attention over 4200 tokens.
- Best checkpoint: step 26000. No improvement by step 30000.

## Evaluation

- Metrics: BER per SNR point (bit error rate averaged over data REs).
- Baselines:
  - AWGN theory `Q(√SNR)` (flat-channel floor).
  - EPA perfect-CSI ZF oracle (uses true `H[k]` + ZF + QPSK slice).
  - EPA LS + linear-freq-interp + nearest-time-interp + ZF + QPSK slice.
- SNR points: 0 to 25 dB in 2.5 dB steps (11 points).
- Samples: 32 subframes per SNR point ≈ 256k data bits per point.
- Seeds: fixed for this report; error bars from more samples scheduled for E04.

## W&B run ID

Not tracked for E01 — predates the W&B adoption decision. Future runs (E02+) will log to W&B.
