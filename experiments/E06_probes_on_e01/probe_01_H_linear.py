"""Probe 1 — Linear probe for H[t, k] from the residual stream.

For each layer, fit a linear map from hidden_state[B, 14, 300, 128] to
H_true[B, 14, 300] (as 2 real channels), separately for pilot and data
REs. Report R² on a held-out split.

See notebook/02_experiments/E06_probes_on_e01/hypothesis.md §"Probe 1".
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from _common import load_model, generate_probe_batch, register_residual_hooks, make_rng


def fit_linear_probe(X, Y, ridge=1e-3):
    """Ridge regression. X: (N, D), Y: (N, K). Returns (W, b, R2)."""
    # TODO(E06): implement closed-form ridge
    # W = (X^T X + λ I)^-1 X^T Y
    # b = mean(Y) - W @ mean(X)
    # R2 = 1 - SSres/SStot on held-out split
    raise NotImplementedError("fit_linear_probe: see method.md §Probe 1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to E01 best.pt")
    p.add_argument("--n-subframes", type=int, default=500)
    p.add_argument("--snr-list", type=str, default="5,15,25",
                   help="Comma-separated SNR values in dB")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).parent / "figures"))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    snrs = [float(s) for s in args.snr_list.split(",")]

    model, _ = load_model(args.ckpt, device=args.device)
    hidden_cache, handles = register_residual_hooks(model)

    # TODO(E06):
    # 1. For each SNR in snrs:
    #    - Generate n_subframes with fresh H_true + is_pilot.
    #    - Forward-pass under torch.no_grad(); snapshot hidden_cache.
    #    - For each layer L:
    #        - Gather hidden[L] at pilot cells and at data cells.
    #        - Fit linear probe (pilot cells only → H_true at pilot cells).
    #        - Fit linear probe (data cells only → H_true at data cells).
    #    - Record R² per (layer, SNR, pilot/data).
    # 2. Plot R² vs layer, one line per (SNR × pilot/data).
    # 3. Save to probe_01_R2_per_layer.pdf.
    # 4. Append markdown table to
    #    notebook/02_experiments/E06_probes_on_e01/results.md,
    #    §"Probe 1".

    for h in handles:
        h.remove()

    raise NotImplementedError(
        "probe_01: not implemented yet. Scaffold only. "
        "See hypothesis.md + method.md for the spec."
    )


if __name__ == "__main__":
    main()
