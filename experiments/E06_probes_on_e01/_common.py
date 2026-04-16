"""Shared helpers for E06 probes: ckpt loading, hidden-state extraction, data.

All four probes import from here so the checkpoint-loading and
feature-extraction paths stay identical across probes.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch


# Make sibling E01 code importable.
_E01_DIR = Path(__file__).resolve().parent.parent / "E01_ofdm_supervised_rx"
if str(_E01_DIR) not in sys.path:
    sys.path.insert(0, str(_E01_DIR))


def load_model(ckpt_path, device="cpu"):
    """Load the E01 LearnedReceiver from a checkpoint.

    Returns (model, ckpt_dict). Model is set to eval() with grads
    disabled. The ckpt_dict is the raw torch.load output.
    """
    from model import LearnedReceiver

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cargs = ckpt["args"]
    model = LearnedReceiver(
        d_model=cargs["d_model"],
        n_heads=cargs["n_heads"],
        n_layers=cargs["n_layers"],
        d_ff=cargs["d_ff"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt


def generate_probe_batch(batch_size, rng, snr_db):
    """Wrapper around dataset.generate_batch that fixes SNR.

    Returns the full dict with rx_grid, tx_bits, data_mask, plus
    H_true, x_tx when the dataset exposes them.
    """
    from dataset import generate_batch

    return generate_batch(batch_size, rng, snr_range=(snr_db, snr_db))


def register_residual_hooks(model):
    """Register forward hooks on every transformer block's output.

    Returns (hidden_cache, handles) where hidden_cache is a dict
    indexed by layer number (0 = post-embedding, 1..n_layers =
    post-block-i). Call handles.remove() for each handle when done.

    Caller is responsible for:
    - calling `hidden_cache.clear()` before each forward pass;
    - removing hooks when done.
    """
    hidden_cache = {}
    handles = []

    # TODO(E06): inspect LearnedReceiver's module structure and
    # register hooks on:
    #   - the post-embedding tensor (layer 0)
    #   - each TransformerBlock's output (layers 1..n_layers)
    # Hook signature:
    #   def hook(module, inputs, output):
    #       hidden_cache[layer_idx] = output.detach().cpu()

    raise NotImplementedError(
        "register_residual_hooks: inspect LearnedReceiver and wire up "
        "hooks. See notebook/02_experiments/E06_probes_on_e01/method.md "
        "for which layers to capture."
    )


def make_rng(seed):
    return np.random.default_rng(seed)
