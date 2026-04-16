"""Sequential runner for all four E06 probes.

Thin wrapper. Each probe is still a standalone script; this file just
walks through them in the recommended order (1 → 2 → 4 → 3) so that
Probe 3 (the central ablation test) lands last, after Probes 1, 2, 4
have established what "joint-grid structure" looks like inside the
model.

Usage:

    python experiments/E06_probes_on_e01/run_all_probes.py \
        --ckpt experiments/E01_ofdm_supervised_rx/checkpoints/best.pt

Each sub-probe writes its own PDF into `figures/` and appends a block
to `notebook/02_experiments/E06_probes_on_e01/results.md`. See
`method.md` for the per-probe protocol.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROBES = [
    ("probe_01_H_linear.py",           "Probe 1 — Linear probe for H[t, k]"),
    ("probe_02_pilot_output_vs_ls.py", "Probe 2 — Pilot-cell output denoising"),
    ("probe_04_perturbation_kernel.py","Probe 4 — Perturbation kernel"),
    ("probe_03_per_cell_ablation.py",  "Probe 3 — Per-cell ablation (central test)"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-subframes", type=int, default=500)
    p.add_argument("--stop-on-error", action="store_true",
                   help="Abort the whole sweep if any probe raises.")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    logs_dir = here / "logs"
    logs_dir.mkdir(exist_ok=True)

    results = []
    for script, title in PROBES:
        banner = f"\n{'=' * 72}\n{title}\n{'=' * 72}\n"
        print(banner, flush=True)

        log_path = logs_dir / (Path(script).stem + ".log")
        cmd = [
            sys.executable, str(here / script),
            "--ckpt", args.ckpt,
            "--device", args.device,
        ]
        # probe_04 doesn't take --n-subframes; others do.
        if "probe_04" not in script:
            cmd += ["--n-subframes", str(args.n_subframes)]

        t0 = time.time()
        try:
            with open(log_path, "w") as f:
                f.write(banner)
                f.write(f"cmd: {' '.join(cmd)}\n\n")
                f.flush()
                rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            results.append((title, status, dt, log_path))
            print(f"  → {status} in {dt:.1f}s, log: {log_path}")
            if rc != 0 and args.stop_on_error:
                print("Stopping because --stop-on-error was set.")
                break
        except Exception as e:
            dt = time.time() - t0
            results.append((title, f"EXC: {e}", dt, log_path))
            print(f"  → EXCEPTION {e}")
            if args.stop_on_error:
                break

    # Summary
    print("\n" + "=" * 72)
    print("E06 probe sweep summary")
    print("=" * 72)
    for title, status, dt, log in results:
        print(f"  [{status:>10}]  {title:<50}  ({dt:6.1f}s)  {log}")


if __name__ == "__main__":
    main()
