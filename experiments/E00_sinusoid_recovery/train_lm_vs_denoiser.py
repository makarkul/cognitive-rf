"""LM-style next-noisy-sample vs supervised denoiser side experiment.

Question
--------
The canonical E00 training in ``train.py`` is a *denoising autoregressor*: it
feeds the model a noisy input stream but uses the clean signal as the MSE
target. A reader who sees "next-sample prediction transformer" might assume an
LLM-style setup where the only label is the next observed sample (here, the
next *noisy* sample). Those are different problems.

This script trains the same architecture under both regimes from the same
seed and same data stream, then asks two diagnostic questions:

1. Does each variant still discover AR(2)-like structure (peaks at lag 1, 2
   in the attention lag profile)?
2. Does each variant still encode the sinusoid frequency in its hidden state
   (linear-probe R² well above the random-init control)?

Outputs (written to ``lm_vs_denoiser/``):

* ``denoiser.pth``, ``lm.pth``                 — the two trained checkpoints
* ``loss_curves.pdf``                          — train/val MSE for both
* ``attention_lag_profile.pdf``                — AR-lag bar plots, side by side
* ``freq_probe.pdf``                           — frequency probe scatter
* ``summary.json``                             — final metrics

Run
---
    python experiments/E00_sinusoid_recovery/train_lm_vs_denoiser.py \\
        --epochs 12 --train-size 4000 --val-size 800

Defaults are tuned to finish in a few minutes on a CPU (no GPU required).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from signal_dataset import generate_noisy_sinusoid
from ts_transformer import TS_TRANSFORMER_CONFIG, TimeSeriesTransformer, count_parameters


# ---------------------------------------------------------------------------
# Paired dataset: returns (input, clean_target, noisy_target) per sample
# ---------------------------------------------------------------------------

class PairedSinusoidDataset(Dataset):
    """Yields ``(input_seq, clean_target, noisy_target)`` triples.

    ``clean_target = clean[1..N]``   (denoiser target — same as ``train.py``)
    ``noisy_target = noisy[1..N]``   (LM-style target — next noisy sample)

    Both targets are aligned to the same ``input_seq = noisy[0..N-1]`` so that
    the two regimes train on identical inputs and only differ in their loss.

    Noise modes (selected by ``noise_mode``):

    - ``iid``           – per-sample i.i.d. amplitude + phase noise (default).
    - ``ar1_coloured``  – amplitude noise replaced by AR(1) coloured noise:
                          ``e[n] = alpha*e[n-1] + sqrt(1-alpha^2)*w[n]``,
                          ``w ~ N(0, amp_noise_std^2)``. Phase noise stays i.i.d.
    - ``wiener_phase``  – phase noise integrated into a Wiener walk:
                          ``theta[n] = theta[n-1] + nu[n]``,
                          ``nu ~ N(0, phase_noise_std^2)``. Amplitude i.i.d.
    - ``dc_offset``     – per-sequence DC bias ``mu ~ N(0, dc_offset_std)``
                          added to the noisy stream (clean stays unbiased).
                          Amplitude + phase noise stay i.i.d.

    All modes use the same per-index RNG so denoiser and LM regimes see
    byte-identical streams.
    """

    NOISE_MODES = ("iid", "ar1_coloured", "wiener_phase", "dc_offset")

    def __init__(self, context_length, dataset_size, fs=100.0,
                 freq_range=(1.0, 20.0), amp_range=(0.5, 2.0),
                 amp_noise_std=0.2, phase_noise_std=0.1,
                 noise_mode="iid", alpha_ar1=0.9, dc_offset_std=0.5,
                 input_demean=False, seed=0):
        if noise_mode not in self.NOISE_MODES:
            raise ValueError(
                f"noise_mode must be one of {self.NOISE_MODES}, got {noise_mode!r}"
            )
        self.context_length = context_length
        self.dataset_size = dataset_size
        self.fs = fs
        self.freq_range = freq_range
        self.amp_range = amp_range
        self.amp_noise_std = amp_noise_std
        self.phase_noise_std = phase_noise_std
        self.noise_mode = noise_mode
        self.alpha_ar1 = alpha_ar1
        self.dc_offset_std = dc_offset_std
        self.input_demean = input_demean
        # Per-index deterministic RNG so denoiser/LM see byte-identical streams.
        self.base_seed = seed

    def __len__(self):
        return self.dataset_size

    def _gen_amp_noise(self, rng, n):
        if self.noise_mode == "ar1_coloured":
            w = rng.standard_normal(n) * self.amp_noise_std * math.sqrt(
                1.0 - self.alpha_ar1 ** 2
            )
            e = np.empty(n, dtype=np.float64)
            # Stationary initial sample.
            e[0] = rng.standard_normal() * self.amp_noise_std
            for k in range(1, n):
                e[k] = self.alpha_ar1 * e[k - 1] + w[k]
            return e
        return rng.standard_normal(n) * self.amp_noise_std

    def _gen_phase_noise(self, rng, n):
        if self.noise_mode == "wiener_phase":
            steps = rng.standard_normal(n) * self.phase_noise_std
            return np.cumsum(steps)
        return rng.standard_normal(n) * self.phase_noise_std

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.base_seed * 1_000_003 + idx)
        freq = float(rng.uniform(*self.freq_range))
        amplitude = float(rng.uniform(*self.amp_range))
        phase = float(rng.uniform(0.0, 2 * math.pi))
        n = self.context_length + 1

        t = np.arange(n) / self.fs
        amp_noise = self._gen_amp_noise(rng, n)
        phase_noise = self._gen_phase_noise(rng, n)
        clean = amplitude * np.sin(2 * math.pi * freq * t + phase)
        noisy = (amplitude + amp_noise) * np.sin(
            2 * math.pi * freq * t + phase + phase_noise
        )

        if self.noise_mode == "dc_offset":
            mu = float(rng.standard_normal()) * self.dc_offset_std
            noisy = noisy + mu

        clean = clean.astype(np.float32)
        noisy = noisy.astype(np.float32)

        # Option-1 preprocessing: subtract per-sequence mean from the noisy
        # stream and from the LM target. The clean target is offset-free by
        # construction (the dc_offset mode adds μ only to the noisy stream),
        # so it stays untouched. This mirrors a real receiver's AGC stage.
        if self.input_demean:
            mean_in = float(np.mean(noisy[: self.context_length]))
            noisy = noisy - mean_in

        input_seq = torch.tensor(noisy[: self.context_length])
        clean_tgt = torch.tensor(clean[1: self.context_length + 1])
        noisy_tgt = torch.tensor(noisy[1: self.context_length + 1])
        return input_seq, clean_tgt, noisy_tgt


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, device, target_kind):
    """One pass over ``loader``. ``target_kind`` is 'clean' or 'noisy'."""
    model.train()
    total = 0.0
    count = 0
    for x, y_clean, y_noisy in loader:
        x = x.to(device)
        target = (y_clean if target_kind == "clean" else y_noisy).to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, target)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.shape[0]
        count += x.shape[0]
    return total / count


def evaluate(model, loader, device):
    """Return MSE-vs-clean, MSE-vs-noisy, and SNR_in / SNR_out (dB)."""
    model.eval()
    sse_clean = 0.0
    sse_noisy = 0.0
    sse_naive = 0.0
    sig_pow = 0.0
    n_elem = 0
    with torch.no_grad():
        for x, y_clean, y_noisy in loader:
            x = x.to(device)
            y_clean = y_clean.to(device)
            y_noisy = y_noisy.to(device)
            pred = model(x)
            sse_clean += torch.sum((pred - y_clean) ** 2).item()
            sse_noisy += torch.sum((pred - y_noisy) ** 2).item()
            # Naive predictor: use noisy[t] as the prediction for clean[t+1]
            sse_naive += torch.sum((x - y_clean) ** 2).item()
            sig_pow += torch.sum(y_clean ** 2).item()
            n_elem += y_clean.numel()
    mse_clean = sse_clean / n_elem
    mse_noisy = sse_noisy / n_elem
    sig_p = sig_pow / n_elem
    snr_in = 10 * math.log10(sig_p / max(sse_naive / n_elem, 1e-12))
    snr_out = 10 * math.log10(sig_p / max(mse_clean, 1e-12))
    return {
        "mse_clean": mse_clean,
        "mse_noisy": mse_noisy,
        "snr_in_db": snr_in,
        "snr_out_db": snr_out,
        "snr_gain_db": snr_out - snr_in,
    }


def train_one(model, train_loader, val_loader, device, epochs, lr, target_kind):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    train_curve = []
    val_clean_curve = []
    val_noisy_curve = []
    snr_curve = []
    for ep in range(epochs):
        t0 = time.time()
        train_mse = run_epoch(model, train_loader, optimizer, device, target_kind)
        val = evaluate(model, val_loader, device)
        train_curve.append(train_mse)
        val_clean_curve.append(val["mse_clean"])
        val_noisy_curve.append(val["mse_noisy"])
        snr_curve.append(val["snr_gain_db"])
        print(
            f"  [{target_kind:8s}] ep {ep + 1:02d}/{epochs}  "
            f"train_mse={train_mse:.4f}  "
            f"val_mse_clean={val['mse_clean']:.4f}  "
            f"val_mse_noisy={val['mse_noisy']:.4f}  "
            f"SNR gain={val['snr_gain_db']:+.2f} dB  "
            f"({time.time() - t0:.1f}s)"
        )
    return {
        "train_curve": train_curve,
        "val_mse_clean": val_clean_curve,
        "val_mse_noisy": val_noisy_curve,
        "snr_gain_db": snr_curve,
        "final": evaluate(model, val_loader, device),
    }


# ---------------------------------------------------------------------------
# Diagnostic 1 — attention lag profile (AR(2) signature)
# ---------------------------------------------------------------------------

def compute_attention_lag_profile(model, freq, fs, max_lag=20):
    """Return (lag_weights, ar_coeff_2cos_omega) for one test frequency.

    Mirrors ``analyze.plot_attention_lag_profile`` for the last block, averaged
    over heads. Input is a clean sinusoid (no noise) so the structure isn't
    masked by perturbations.
    """
    cfg_ctx = model.pos_emb.weight.shape[0]
    device = next(model.parameters()).device
    clean = np.sin(2 * math.pi * freq * np.arange(cfg_ctx) / fs).astype(np.float32)
    x = torch.from_numpy(clean).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        x_in = x.unsqueeze(-1)
        h = model.input_proj(x_in) + model.pos_emb(
            torch.arange(cfg_ctx, device=device)
        )
        last_weights = None
        for block in model.trf_blocks:
            h_norm = block.norm1(h)
            att = block.att
            b, n_tok, _ = h_norm.shape
            keys = (
                att.W_key(h_norm)
                .view(b, n_tok, att.num_heads, att.head_dim)
                .transpose(1, 2)
            )
            queries = (
                att.W_query(h_norm)
                .view(b, n_tok, att.num_heads, att.head_dim)
                .transpose(1, 2)
            )
            values = (
                att.W_value(h_norm)
                .view(b, n_tok, att.num_heads, att.head_dim)
                .transpose(1, 2)
            )
            scores = queries @ keys.transpose(2, 3)
            mask_bool = att.mask.bool()[:n_tok, :n_tok]
            scores.masked_fill_(mask_bool, -torch.inf)
            weights = torch.softmax(scores / keys.shape[-1] ** 0.5, dim=-1)
            last_weights = weights[0].mean(dim=0).cpu().numpy()  # (seq, seq)
            ctx_vec = (weights @ values).transpose(1, 2).contiguous().view(
                b, n_tok, att.d_out
            )
            ctx_vec = att.out_proj(ctx_vec)
            h = h + ctx_vec
            shortcut = h
            h = block.norm2(h)
            h = block.ff(h)
            h = h + shortcut

    lag_w = np.zeros(max_lag, dtype=np.float64)
    counts = np.zeros(max_lag, dtype=np.int64)
    for q in range(cfg_ctx):
        for k in range(q + 1):
            lag = q - k
            if lag < max_lag:
                lag_w[lag] += last_weights[q, k]
                counts[lag] += 1
    lag_w /= np.maximum(counts, 1)
    omega = 2 * math.pi * freq / fs
    return lag_w, 2 * math.cos(omega)


# ---------------------------------------------------------------------------
# Diagnostic 2 — linear frequency probe
# ---------------------------------------------------------------------------

def collect_layer_features(model, freqs_hz, fs, ctx_len,
                           amp_noise_std, phase_noise_std,
                           amp_range=(0.5, 2.0), seed=0):
    """Run sinusoids through the model, return per-layer mean-pooled features."""
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    feats = {}
    activations = {}

    handles = []
    for i, blk in enumerate(model.trf_blocks):
        def make_hook(name):
            def hook(_m, _inp, out):
                activations[name] = out.detach()
            return hook
        handles.append(blk.register_forward_hook(make_hook(f"layer_{i + 1}")))

    model.eval()
    with torch.no_grad():
        for i, f in enumerate(freqs_hz):
            a = float(rng.uniform(*amp_range))
            phi = float(rng.uniform(0.0, 2 * math.pi))
            noisy, _ = generate_noisy_sinusoid(
                ctx_len, f, a, phi, fs,
                amp_noise_std=amp_noise_std, phase_noise_std=phase_noise_std,
            )
            x = torch.from_numpy(noisy).unsqueeze(0).to(device)
            _ = model(x)
            for name, act in activations.items():
                vec = act.mean(dim=1).squeeze(0).cpu().numpy()
                feats.setdefault(
                    name,
                    np.empty((len(freqs_hz), vec.shape[-1]), dtype=np.float32),
                )
                feats[name][i] = vec
            activations.clear()

    for h in handles:
        h.remove()
    return feats


def ridge_probe(X_train, y_train, X_test, y_test, weight_decay=1e-4):
    """Closed-form ridge regression. Returns dict with R²/RMSE on the test split."""
    Xa = np.concatenate([X_train, np.ones((X_train.shape[0], 1), dtype=np.float32)], axis=1)
    Xt = np.concatenate([X_test, np.ones((X_test.shape[0], 1), dtype=np.float32)], axis=1)
    lam = weight_decay * Xa.shape[0]
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1], dtype=np.float32)
    b = Xa.T @ y_train.astype(np.float32)
    w = np.linalg.solve(A, b)
    pred_test = Xt @ w
    ss_res = float(np.sum((y_test - pred_test) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse = float(np.sqrt(np.mean((y_test - pred_test) ** 2)))
    return {"r2_test": r2, "rmse_test": rmse, "pred_test": pred_test}


def run_freq_probe(model, fs, ctx_len, amp_noise_std, phase_noise_std,
                   freq_range=(1.0, 20.0), n_train=1500, n_test=400, seed=0):
    rng_train = np.random.default_rng(seed)
    rng_test = np.random.default_rng(seed + 1)
    f_train = rng_train.uniform(*freq_range, size=n_train).astype(np.float32)
    f_test = rng_test.uniform(*freq_range, size=n_test).astype(np.float32)
    feats_train = collect_layer_features(
        model, f_train, fs, ctx_len, amp_noise_std, phase_noise_std, seed=seed + 2
    )
    feats_test = collect_layer_features(
        model, f_test, fs, ctx_len, amp_noise_std, phase_noise_std, seed=seed + 3
    )
    out = {}
    for name in sorted(feats_train.keys()):
        r = ridge_probe(feats_train[name], f_train, feats_test[name], f_test)
        out[name] = {
            "r2_test": r["r2_test"],
            "rmse_test": r["rmse_test"],
            "pred_test": r["pred_test"].tolist(),
            "true_test": f_test.tolist(),
        }
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(denoiser_log, lm_log, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = range(1, len(denoiser_log["train_curve"]) + 1)

    ax = axes[0]
    ax.plot(epochs, denoiser_log["val_mse_clean"], "o-", label="denoiser val MSE-vs-clean")
    ax.plot(epochs, lm_log["val_mse_clean"], "s-", label="LM val MSE-vs-clean")
    ax.plot(epochs, denoiser_log["val_mse_noisy"], "o--", alpha=0.6,
            label="denoiser val MSE-vs-noisy")
    ax.plot(epochs, lm_log["val_mse_noisy"], "s--", alpha=0.6,
            label="LM val MSE-vs-noisy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_title("Validation MSE — both targets, both models")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, denoiser_log["snr_gain_db"], "o-", label="denoiser")
    ax.plot(epochs, lm_log["snr_gain_db"], "s-", label="LM-style")
    ax.axhline(y=0, color="gray", linestyle=":")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SNR gain over naive predictor (dB)")
    ax.set_title("Denoising performance vs naive baseline")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_attention_lag(denoiser_model, lm_model, freqs, fs, save_path):
    n = len(freqs)
    fig, axes = plt.subplots(2, n, figsize=(4.0 * n, 7.0), sharey=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, f in enumerate(freqs):
        for row, (model, label, color) in enumerate(
            [(denoiser_model, "denoiser", "#065A82"),
             (lm_model, "LM-style", "#B85042")]
        ):
            lag_w, ar_coef = compute_attention_lag_profile(model, f, fs)
            ax = axes[row, col]
            ax.bar(range(len(lag_w)), lag_w, color=color, alpha=0.85)
            ax.axvline(x=1, color="red", linestyle="--", alpha=0.4)
            ax.axvline(x=2, color="orange", linestyle="--", alpha=0.4)
            ax.set_title(f"{label}  f={f:.0f} Hz\n2cos(w)={ar_coef:+.2f}")
            ax.set_xlabel("Lag (samples)")
            if col == 0:
                ax.set_ylabel("Avg attention weight")
            ax.grid(alpha=0.2)

    fig.suptitle(
        "AR(2) attention signature — last layer, averaged over heads\n"
        "(peaks at lag 1 and 2 = AR(2) structure was learned)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


def plot_freq_probe(denoiser_probe, lm_probe, save_path):
    layer_names = sorted(denoiser_probe.keys())
    fig, axes = plt.subplots(2, len(layer_names),
                             figsize=(4.0 * len(layer_names), 7.5))
    if len(layer_names) == 1:
        axes = axes.reshape(2, 1)
    lo, hi = 1.0, 20.0

    for col, name in enumerate(layer_names):
        for row, (label, probe, color) in enumerate(
            [("denoiser", denoiser_probe[name], "#065A82"),
             ("LM-style", lm_probe[name], "#B85042")]
        ):
            ax = axes[row, col]
            ax.scatter(probe["true_test"], probe["pred_test"], s=6, alpha=0.5, color=color)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, alpha=0.6)
            ax.set_xlabel("true frequency (Hz)")
            ax.set_ylabel("predicted frequency (Hz)")
            ax.set_title(f"{label}  |  {name}\nR²={probe['r2_test']:.3f}  "
                         f"RMSE={probe['rmse_test']:.2f} Hz")
            ax.set_xlim(lo - 1, hi + 1)
            ax.set_ylim(lo - 1, hi + 1)
            ax.grid(alpha=0.3)

    fig.suptitle(
        "Linear frequency probe on hidden states — does the model encode f?",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Per-mode driver
# ---------------------------------------------------------------------------

def run_mode(args, cfg, mode, out_root, device):
    """Train (denoiser, LM) under one noise mode and write per-mode artifacts.

    Returns a summary dict with the headline scalars for cross-mode plotting.
    """
    mode_dir = out_root / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========================== mode = {mode} ==========================")
    print(f"  writing artifacts under {mode_dir}")

    ds_kwargs = dict(
        amp_noise_std=args.amp_noise,
        phase_noise_std=args.phase_noise,
        noise_mode=mode,
        alpha_ar1=args.alpha_ar1,
        dc_offset_std=args.dc_offset_std,
        input_demean=args.input_demean,
    )
    train_ds = PairedSinusoidDataset(
        cfg["context_length"], args.train_size, seed=args.seed, **ds_kwargs
    )
    val_ds = PairedSinusoidDataset(
        cfg["context_length"], args.val_size, seed=args.seed + 17, **ds_kwargs
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    torch.manual_seed(args.seed)
    denoiser = TimeSeriesTransformer(cfg).to(device)
    torch.manual_seed(args.seed)
    lm_model = TimeSeriesTransformer(cfg).to(device)

    print(f"\n--- [{mode}] Training DENOISER (target = clean[1..N]) ---")
    denoiser_log = train_one(
        denoiser, train_loader, val_loader, device,
        args.epochs, args.lr, target_kind="clean",
    )
    torch.save(denoiser.state_dict(), mode_dir / "denoiser.pth")

    print(f"\n--- [{mode}] Training LM-STYLE (target = noisy[1..N]) ---")
    lm_log = train_one(
        lm_model, train_loader, val_loader, device,
        args.epochs, args.lr, target_kind="noisy",
    )
    torch.save(lm_model.state_dict(), mode_dir / "lm.pth")

    plot_loss_curves(denoiser_log, lm_log, mode_dir / "loss_curves.pdf")

    print(f"\n--- [{mode}] Attention lag profile ---")
    test_freqs = [3.0, 5.0, 10.0, 15.0]
    plot_attention_lag(denoiser, lm_model, test_freqs,
                       fs=100.0, save_path=mode_dir / "attention_lag_profile.pdf")

    print(f"\n--- [{mode}] Linear frequency probe ---")
    den_probe = run_freq_probe(
        denoiser, fs=100.0, ctx_len=cfg["context_length"],
        amp_noise_std=args.amp_noise, phase_noise_std=args.phase_noise,
        seed=args.seed,
    )
    lm_probe = run_freq_probe(
        lm_model, fs=100.0, ctx_len=cfg["context_length"],
        amp_noise_std=args.amp_noise, phase_noise_std=args.phase_noise,
        seed=args.seed,
    )
    plot_freq_probe(den_probe, lm_probe, mode_dir / "freq_probe.pdf")

    # Drift diagnostic — empirical mean(LM - denoiser) on the val set.
    drift_rmse, drift_bias = _output_drift(denoiser, lm_model, val_loader, device)

    summary = {
        "mode": mode,
        "config": cfg,
        "ds_params": {k: v for k, v in ds_kwargs.items()},
        "args": {k: v for k, v in vars(args).items() if k != "noise_mode"},
        "denoiser_final": denoiser_log["final"],
        "lm_final": lm_log["final"],
        "delta_mse_clean": lm_log["final"]["mse_clean"] - denoiser_log["final"]["mse_clean"],
        "delta_snr_gain_db": lm_log["final"]["snr_gain_db"] - denoiser_log["final"]["snr_gain_db"],
        "drift_rmse": drift_rmse,
        "drift_bias": drift_bias,
        "denoiser_freq_probe": {
            k: {"r2_test": v["r2_test"], "rmse_test": v["rmse_test"]}
            for k, v in den_probe.items()
        },
        "lm_freq_probe": {
            k: {"r2_test": v["r2_test"], "rmse_test": v["rmse_test"]}
            for k, v in lm_probe.items()
        },
    }
    with open(mode_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n--- [{mode}] Done. Wrote {mode_dir / 'summary.json'} ---")
    return summary


def _output_drift(denoiser, lm_model, val_loader, device):
    """Compute RMS(LM_pred - denoiser_pred) and mean-bias on the val set."""
    denoiser.eval()
    lm_model.eval()
    sse = 0.0
    sbias = 0.0
    n = 0
    with torch.no_grad():
        for x, _, _ in val_loader:
            x = x.to(device)
            d = denoiser(x)
            l = lm_model(x)
            diff = l - d
            sse += torch.sum(diff ** 2).item()
            sbias += torch.sum(diff).item()
            n += diff.numel()
    return math.sqrt(sse / max(n, 1)), sbias / max(n, 1)


# ---------------------------------------------------------------------------
# Cross-mode summary
# ---------------------------------------------------------------------------

def plot_sweep_summary(results, save_path):
    """Bar plot of LM-vs-denoiser deltas across noise modes."""
    modes = [r["mode"] for r in results]
    deltas_mse = [r["delta_mse_clean"] for r in results]
    deltas_snr = [r["delta_snr_gain_db"] for r in results]
    drifts = [r["drift_rmse"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    colors = ["#2a9d8f" if d <= 0.005 else "#e76f51" for d in deltas_mse]
    ax.bar(modes, deltas_mse, color=colors, alpha=0.85)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.6)
    ax.set_ylabel("Δ MSE-vs-clean  (LM − denoiser)")
    ax.set_title("Denoising-quality penalty under LM-style training\n"
                 "(>0 = LM worse than denoiser)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[1]
    colors = ["#2a9d8f" if d >= -0.1 else "#e76f51" for d in deltas_snr]
    ax.bar(modes, deltas_snr, color=colors, alpha=0.85)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.6)
    ax.set_ylabel("Δ SNR gain (dB)  (LM − denoiser)")
    ax.set_title("SNR-gain penalty under LM-style training\n"
                 "(<0 = LM loses SNR)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    ax = axes[2]
    ax.bar(modes, drifts, color="#264653", alpha=0.85)
    ax.set_ylabel("RMS(LM_pred − denoiser_pred)  on val set")
    ax.set_title("Output-divergence between regimes\n"
                 "(0 = identical, >0 = LM diverges)")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle("LM-style vs denoiser across noise modes", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp-noise", type=float, default=0.2)
    parser.add_argument("--phase-noise", type=float, default=0.1)
    parser.add_argument(
        "--noise-mode",
        choices=list(PairedSinusoidDataset.NOISE_MODES) + ["sweep"],
        default="iid",
        help="Single mode, or 'sweep' to run all four sequentially.",
    )
    parser.add_argument("--alpha-ar1", type=float, default=0.9,
                        help="AR(1) coefficient for ar1_coloured noise.")
    parser.add_argument("--dc-offset-std", type=float, default=0.5,
                        help="Std-dev of per-sequence DC offset (dc_offset mode).")
    parser.add_argument("--input-demean", action="store_true",
                        help="Option-1 preprocessing: subtract per-sequence mean "
                             "from the noisy input + LM target (clean target "
                             "untouched). Mirrors a real receiver's AGC stage.")
    parser.add_argument("--scaled-arch", action="store_true",
                        help="Use a scaled architecture (emb_dim=128, n_heads=4, "
                             "n_layers=4, ~200k params, ~7x default) to test "
                             "whether the LM-style penalty is a capacity issue.")
    parser.add_argument("--out-dir", type=str, default="lm_vs_denoiser")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    out_root = (here / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Writing results under {out_root}")

    cfg = TS_TRANSFORMER_CONFIG.copy()
    if args.scaled_arch:
        cfg.update({"emb_dim": 128, "n_heads": 4, "n_layers": 4})
    print(f"Model config: {cfg}")
    print(f"Model parameters: "
          f"{count_parameters(TimeSeriesTransformer(cfg)):,}")

    if args.noise_mode == "sweep":
        modes = list(PairedSinusoidDataset.NOISE_MODES)
    else:
        modes = [args.noise_mode]

    results = []
    for mode in modes:
        results.append(run_mode(args, cfg, mode, out_root, device))

    if len(modes) > 1:
        plot_sweep_summary(results, out_root / "noise_mode_sweep.pdf")
        with open(out_root / "sweep_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out_root / 'sweep_summary.json'}")

        print("\n--- Sweep summary ---")
        print(f"{'mode':<16}{'Δ MSE-clean':>14}{'Δ SNR-gain dB':>16}"
              f"{'drift RMSE':>14}")
        for r in results:
            print(
                f"{r['mode']:<16}{r['delta_mse_clean']:+14.5f}"
                f"{r['delta_snr_gain_db']:+16.3f}{r['drift_rmse']:14.4f}"
            )
    print("\nDone.")


if __name__ == "__main__":
    main()
