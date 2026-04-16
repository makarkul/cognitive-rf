"""Probe 2 — Pilot-cell output denoising.

Compare the model's internal H-estimate at pilot cells (via Probe 1's
layer-3 linear head) against the raw LS estimate y_pilot / x_pilot.
Report the denoising gain in dB as a function of SNR.

See notebook/02_experiments/E06_probes_on_e01/hypothesis.md §"Probe 2".
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from _common import load_model, generate_probe_batch, register_residual_hooks, make_rng


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-subframes", type=int, default=500)
    p.add_argument("--snr-list", type=str, default="5,10,15,25")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).parent / "figures"))
    p.add_argument("--probe-head", type=str, default=None,
                   help="Path to probe_01 layer-3 pilot head (.pt). "
                        "If omitted, refit here.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    snrs = [float(s) for s in args.snr_list.split(",")]

    model, _ = load_model(args.ckpt, device=args.device)

    # TODO(E06):
    # 1. Either load probe_01's layer-3 pilot-fit head, or refit here
    #    on a held-in calibration batch at 15 dB.
    # 2. For each SNR:
    #    - Generate n_subframes with known H_true, is_pilot, x_tx_pilots.
    #    - Compute LS estimate: H_LS = y / x_tx at pilot cells only.
    #    - Extract layer-3 hidden states at pilot cells.
    #    - Apply linear head → H_model estimate at pilot cells.
    #    - MSE(H_LS vs H_true) and MSE(H_model vs H_true).
    #    - Denoising gain = 10 log10(MSE_LS / MSE_model).
    # 3. Plot denoising gain vs SNR.
    # 4. Save to probe_02_pilot_denoising_gain.pdf.
    # 5. Append to results.md §"Probe 2".

    raise NotImplementedError(
        "probe_02: not implemented yet. Scaffold only."
    )


if __name__ == "__main__":
    main()
