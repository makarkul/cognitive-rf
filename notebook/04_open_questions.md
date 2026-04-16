# Open questions

Running list of things we don't yet understand. Append-only within a phase; pruned at phase boundaries (answered questions move to the relevant experiment's `results.md` or a discussion file).

## From E01 (OFDM supervised receiver)

- **Q1. Is the high-SNR win over the ZF oracle robust?** 32 subframes/SNR is light. Scheduled for E04: 500 subframes/SNR, tighter error bars on the 17.5–25 dB region.
- **Q2. What is the learned RX actually doing?** Candidate probes (scheduled for E06, E09):
  - Compare learned output at pilot cells vs LS estimate (is it denoising the channel estimate?).
  - Linear probe on hidden state to predict `H[k]` — same pattern as the sinusoid frequency probe.
  - Perturb one subcarrier and watch neighbor predictions move (measures learned frequency-domain smoothing kernel).
- **Q3. How does it handle distribution shift?** Trained on EPA. Does it degrade gracefully on EVA (longer delay spread)? On flat AWGN? (scheduled for E05.)
- **Q4. How small can the model go?** 834k params; SoftBank paper claims 1 layer × 1 head is enough on CDL-C. Test on EPA in E02.

## Architecture / design

- **Q5. Is the `is_pilot` binary flag the right conditioning?** An alternative: feed the expected pilot value at pilot cells (and zero at data cells), letting the model learn residuals directly. Two extra input channels. Worth a small ablation.
- **Q6. Does early LayerNorm hurt on RE-grid inputs?** SoftBank paper omits it explicitly to preserve signal magnitude. We keep it. Ablate in E02.
- **Q7. Factorized vs dense positional embedding.** We use 14-way + 300-way factorized (saves 495k params). Is the saving free, or does it cap capacity? Test with a run at dense-positional to see.

## Self-supervised pretraining (phase 2)

- **Q8. Does masked-RE modeling produce a better encoder than supervised training at matched params?** LWM says yes on DeepMIMO channels; we need to verify on EPA resource grids.
- **Q9. What mask ratio is best?** LWM/BERT use 15%; MAE uses 75%. Resource grids have strong local structure — likely closer to MAE than BERT.
- **Q10. Does the encoder learn `H` implicitly, or just bit-level features?** Linear probe in E09.

## Blind / pilotless (phase 3)

- **Q11. Can the conv stem find symbol boundaries without being told?** E11 measures this via a learned STO probe.
- **Q12. Does a differentiable signal renderer stabilize training, or is end-to-end MSE sufficient?** BRF-WM proposes the former; E13 tests both.
- **Q13. At what pilot density does the classical LS+interp pipeline fail first — 50%? 25%? 10%?** Establishes the axis on which the learned RX needs to win.

## Cross-cutting

- **Q14. Sim-to-real gap.** All training is synthetic. The first OTA capture is going to expose a gap. How much randomization at training time is needed to close it? Candidate impairments: CFO, phase noise, IQ imbalance, PA nonlinearity, sample-timing offset, fractional-delay ISI (if CP is insufficient).
- **Q15. Is there a meaningful multi-antenna extension at our compute budget?** SoftBank does 1UE SIMO and 2UE MU-MIMO. Adding an antenna dim to our grid is cheap; the question is what the attention pattern becomes.
- **Q16. Is contrastive or clustering SSL useful in addition to MREM?** wav2vec / CLIP-like losses could provide complementary signal. Out of scope for phase 2, tag for phase 4.

## Methodological

- **Q17. Should each experiment use a fresh random seed and report ±1 SD, or a fixed seed for reproducibility?** Default so far: fixed seed for results, fresh seeds for robustness checks. Revisit if we hit a close call.
- **Q18. Weights & Biases or TensorBoard?** W&B recommended for team visibility; add in phase 1.
