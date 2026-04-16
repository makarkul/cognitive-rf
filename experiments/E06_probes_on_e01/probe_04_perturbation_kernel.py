"""Probe 4 — Perturbation kernel.

Perturb one RE's input by a small delta. Measure the model's output
response at every other RE. Plot the resulting influence kernel in
frequency, in time, and as a 2D heatmap. Repeat for a handful of
reference positions and average.

See notebook/02_experiments/E06_probes_on_e01/hypothesis.md §"Probe 4".
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from _common import load_model, generate_probe_batch, make_rng


def measure_kernel(model, rx_clean, ref_pos, delta_re=0.1, delta_im=0.1):
    """Compute |Δŷ[t, k]| across the whole 14×300 grid after perturbing
    one RE's input at ref_pos=(t0, k0).

    Returns a (14, 300) numpy array of magnitudes.
    """
    # TODO(E06):
    # 1. model(rx_clean) → output_clean, shape (14, 300, 2) logits.
    # 2. rx_perturbed = rx_clean with
    #    rx[t0, k0, 0] += delta_re
    #    rx[t0, k0, 1] += delta_im
    # 3. model(rx_perturbed) → output_perturbed.
    # 4. Return |output_perturbed - output_clean| on a 14×300 grid
    #    by reducing the last dim (mean or norm).
    raise NotImplementedError("measure_kernel")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--snr-db", type=float, default=20.0)
    p.add_argument("--n-ref-positions", type=int, default=5,
                   help="Number of reference REs to average over")
    p.add_argument("--delta", type=float, default=0.14,
                   help="Perturbation magnitude (≈ noise std at 15 dB)")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).parent / "figures"))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = make_rng(args.seed)
    model, _ = load_model(args.ckpt, device=args.device)

    # TODO(E06):
    # 1. Generate one batch of clean subframes at args.snr_db.
    # 2. Pick a representative subframe.
    # 3. Reference positions: (7, 150) center, (0, 150) top edge,
    #    (13, 150) bottom edge, (7, 0) left edge, (7, 299) right edge.
    #    Extend with pilot-adjacent + far-from-pilot if desired.
    # 4. For each ref_pos, call measure_kernel → (14, 300).
    # 5. Stack + average across ref_pos; also keep per-position copies.
    # 6. Plots:
    #    - Frequency slice |Δŷ[t0, k0 + d]| vs d ∈ [-50, +50],
    #      per-ref overlay + mean.
    #    - Time slice |Δŷ[t0 + τ, k0]| vs τ ∈ [-7, +7].
    #    - 2D heatmap of the mean kernel.
    # 7. Save:
    #    - probe_04_perturbation_freq.pdf
    #    - probe_04_perturbation_time.pdf
    #    - probe_04_perturbation_2d.pdf
    # 8. Report half-width @ 50% of peak + max/min time ratio in
    #    results.md §"Probe 4".

    raise NotImplementedError(
        "probe_04: not implemented yet. Scaffold only."
    )


if __name__ == "__main__":
    main()
