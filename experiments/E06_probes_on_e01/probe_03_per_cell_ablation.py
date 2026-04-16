"""Probe 3 — Per-cell ablation (the central test).

Monkey-patch attention to be diagonal-only (each RE attends only to
itself). Re-run the BER vs SNR sweep and compare to the unablated E01
baseline, oracle, and LS+interp.

See notebook/02_experiments/E06_probes_on_e01/hypothesis.md §"Probe 3".

CENTRAL TEST: if ablated BER at 25 dB climbs from 1.23e-3 (E01
unablated) to ~1.78e-3 (oracle) or worse, the "joint-grid attention
beats oracle" story is confirmed mechanistically. If BER stays low,
the advantage came from somewhere else and our hypothesis is wrong.
"""

import argparse
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from _common import load_model, make_rng


@contextmanager
def diagonal_attention(model):
    """Context manager that forces every transformer block's attention
    to a diagonal mask while inside the block, then restores.

    The mask should make each RE attend only to itself — all
    off-diagonal pre-softmax scores set to -inf.
    """
    # TODO(E06):
    # 1. Inspect LearnedReceiver's TransformerBlock to find where
    #    scaled dot-product attention is computed.
    # 2. Either:
    #    (a) monkey-patch the attention forward method to inject a
    #        diagonal mask, or
    #    (b) wrap each block's forward with a version that adds the
    #        mask argument.
    # 3. On __exit__, restore the original forward.
    #
    # A simpler implementation: the LearnedReceiver likely has an
    # attn_mask parameter on its TransformerBlock.forward or
    # MultiheadAttention call — pass an all-(-inf)-except-diagonal
    # mask sized (4200, 4200).
    raise NotImplementedError(
        "diagonal_attention: wire up the mask injection. See method.md."
    )
    yield
    # restore happens here on exit


def ber_sweep_ablated(model, device, snr_db_list, n_subframes, batch_size, seed):
    """Run BER vs SNR under the ablation context."""
    # TODO(E06):
    # Adapt evaluate.learned_ber (from E01) but wrap the forward pass
    # in the diagonal_attention context manager.
    raise NotImplementedError("ber_sweep_ablated: implement using E01's evaluate.py as template")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--snr-min", type=float, default=0.0)
    p.add_argument("--snr-max", type=float, default=25.0)
    p.add_argument("--snr-step", type=float, default=2.5)
    p.add_argument("--n-subframes", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--out", type=str,
                   default=str(Path(__file__).parent / "figures"))
    p.add_argument("--sanity-check", action="store_true",
                   help="Before ablation, run unablated sweep and verify "
                        "it matches published E01 BER (within 1%). "
                        "Recommended on first run.")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    snrs = np.arange(args.snr_min, args.snr_max + 1e-9, args.snr_step)

    model, _ = load_model(args.ckpt, device=args.device)

    # TODO(E06):
    # 1. If --sanity-check: run ber_sweep (unablated) at same SNR
    #    points; diff against published E01 numbers; abort if any
    #    point differs by more than 1 %.
    # 2. Run ber_sweep_ablated under the diagonal_attention CM.
    # 3. Plot learned (unablated), learned (ablated), oracle,
    #    LS+interp, AWGN theory on one semilogy chart.
    # 4. Save to probe_03_per_cell_ablation_ber.pdf.
    # 5. Append numbers to results.md §"Probe 3".

    raise NotImplementedError(
        "probe_03: not implemented yet. This is the central test — "
        "implement carefully and run the sanity check first."
    )


if __name__ == "__main__":
    main()
