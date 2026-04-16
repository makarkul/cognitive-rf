# E06 — Interpretability probes on the E01 learned receiver

Four read-out probes on the frozen E01 checkpoint. No training, no
fine-tuning. CPU only.

See the notebook for the design:

- `notebook/02_experiments/E06_probes_on_e01/hypothesis.md` — what
  we predict each probe will show, *before* we run anything.
- `notebook/02_experiments/E06_probes_on_e01/method.md` — exact
  protocol per probe.
- `notebook/02_experiments/E06_probes_on_e01/results.md` — where
  numbers land as each probe completes.

## Layout

```
experiments/E06_probes_on_e01/
├── README.md                           — this file
├── _common.py                          — ckpt load, hidden-state extraction, batch helper
├── probe_01_H_linear.py                — linear probe for H[t,k] from residual stream
├── probe_02_pilot_output_vs_ls.py      — model H-estimate MSE vs raw LS MSE at pilots
├── probe_03_per_cell_ablation.py       — re-run BER sweep with attention → diagonal
├── probe_04_perturbation_kernel.py     — Dirac-perturbation response in freq/time
├── run_all_probes.py                   — sequential runner for all four
└── logs/                               — per-probe console logs
```

## Running

Each probe is standalone. Sequential recommended (each commits its
results.md block and figure before the next runs):

```bash
# From repo root
python experiments/E06_probes_on_e01/probe_01_H_linear.py \
    --ckpt experiments/E01_ofdm_supervised_rx/checkpoints/best.pt \
    --n-subframes 500

python experiments/E06_probes_on_e01/probe_02_pilot_output_vs_ls.py \
    --ckpt experiments/E01_ofdm_supervised_rx/checkpoints/best.pt \
    --n-subframes 500

python experiments/E06_probes_on_e01/probe_03_per_cell_ablation.py \
    --ckpt experiments/E01_ofdm_supervised_rx/checkpoints/best.pt \
    --n-subframes 500

python experiments/E06_probes_on_e01/probe_04_perturbation_kernel.py \
    --ckpt experiments/E01_ofdm_supervised_rx/checkpoints/best.pt

# Or all at once:
python experiments/E06_probes_on_e01/run_all_probes.py \
    --ckpt experiments/E01_ofdm_supervised_rx/checkpoints/best.pt
```

All probes output to `experiments/E06_probes_on_e01/figures/`.
Expected total wall-clock: ~2 h on a laptop CPU.

## Checkpoint

The probes load `best.pt` from E01. If you do not have it locally:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; \
           hf_hub_download('makarkul/cognitive-rf-E01', 'best.pt', \
                           local_dir='experiments/E01_ofdm_supervised_rx/checkpoints')"
```

## Notes

- All four probes are **read-out only** — the model is never updated.
- Results are committed one probe at a time to preserve the audit
  trail (hypotheses-before-results).
- Probe 3 (per-cell ablation) is the one that will most directly
  answer whether the E01 high-SNR gap over oracle is real joint-grid
  structure. Run it last, because by then Probes 1 + 2 + 4 have
  framed what "joint-grid structure" looks like inside the model.
