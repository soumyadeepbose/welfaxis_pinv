"""Re-fit transfer matrices from cached sweep state, without a GPU.

Every sweep writes its per-cell P(True) values to cache/, so alternative slope
fits cost nothing. This exists so the alpha-restricted fit can be produced for
sweeps that already ran, including partial ones stopped by the time budget.

    python refit.py                 # full grid and |alpha|<=2, every cached sweep
    python refit.py --alpha-max 2   # just the restricted fit
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import config
import steer as S


def main() -> None:
    ap = argparse.ArgumentParser(description="re-fit transfer from cached sweeps")
    ap.add_argument("--alpha-max", type=float, default=None,
                    help="restrict the fit to |alpha| <= this (default: both fits)")
    args = ap.parse_args()

    sweeps = sorted(config.CACHE.glob("sweep_*.json"))
    if not sweeps:
        print(f"no cached sweeps under {config.CACHE}")
        return

    fits = [None, 2.0] if args.alpha_max is None else [args.alpha_max]
    for path in sweeps:
        state = json.loads(path.read_text(encoding="utf-8"))
        ctx = state.get("context") or "avg"
        tag = ctx + ("_deflated" if state.get("deflated") else "")
        tag += "_diag" if state.get("diagonal_only") else ""
        for amax in fits:
            out = S.transfer_from_sweep(state, alpha_max=amax)
            name = f"transfer_{tag}" + ("_alpha2" if amax else "") + ".json"
            config.dump_json(config.RESULTS / name, out)

            T = np.array(out["T_norm"], dtype=float)
            n = T.shape[0]
            off = T[~np.eye(n, dtype=bool)]
            nc = out.get("null_control_mean_ratio")
            with np.errstate(invalid="ignore"):
                off_mean = float(np.nanmean(off)) if np.isfinite(off).any() else float("nan")
            print(f"{name:46s} diag={np.round(out['diagonal'], 4).tolist()}")
            print(f"{'':46s} off-diag T_norm={off_mean:.3f}  "
                  f"null/diag={nc if nc is None else round(nc, 3)}  "
                  f"masked={len(out['masked_cells'])}")


if __name__ == "__main__":
    main()
