"""Steering specificity as a function of intervention layer.

Specificity is the effect of the direction under test divided by the effect of a
random unit vector injected at the same magnitude:

    S(L) = |slope(v_target at L)| / |slope(v_random at L)|

S > 1 means the readout responds to *what* was injected. S <= 1 means it responds
to the injection, not its content, and no conclusion about the direction is
warranted at that layer.

This exists to answer one objection to a null steering result: that the layer was
badly chosen. If S stays at or below 1 across depth, layer choice is not the
explanation. If S rises sharply somewhere, then the layer-selection criterion --
probe AUC, which is correlational -- failed to find the layer where intervention
is specific, and that is the finding.

    python layer_specificity.py --pattern 'transfer_avg_srcba_stor_L*.json'
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

import config


def _slope(d: dict, block: str, persona: str) -> float:
    v = (d.get(block) or {}).get(persona)
    return float(v["slope"]) if v and v.get("slope") is not None else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="transfer_avg_srcba_stor_L*.json")
    ap.add_argument("--persona", default="original", help="steered persona")
    ap.add_argument("--source", default="bare", help="source persona (column of T)")
    ap.add_argument("--plot", action="store_true", default=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(config.RESULTS.glob(args.pattern)):
        m = re.search(r"_L(\d+)", path.stem)
        if not m or path.stem.endswith("alpha2"):
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        T = np.array(d["T"], dtype=float)
        try:
            i = d["personas"].index(args.persona)
            j = d["personas"].index(args.source)
            target = float(T[i, j])
        except (ValueError, IndexError):
            continue
        rnd = _slope(d, "random_control", args.persona)
        nul = _slope(d, "null_control", args.persona)
        rows.append({
            "layer": int(m.group(1)),
            "slope_target": target,
            "slope_random": rnd,
            "slope_null": nul,
            "specificity": abs(target) / abs(rnd) if rnd == rnd and abs(rnd) > 1e-9
            else float("nan"),
        })

    if not rows:
        print(f"no layer-tagged transfer files matching {args.pattern} in {config.RESULTS}")
        return
    rows.sort(key=lambda r: r["layer"])

    spec = np.array([r["specificity"] for r in rows], dtype=float)
    finite = spec[np.isfinite(spec)]
    payload = {
        "steered_persona": args.persona, "source_persona": args.source,
        "per_layer": rows,
        "max_specificity": float(np.nanmax(spec)) if finite.size else None,
        "argmax_layer": int(rows[int(np.nanargmax(spec))]["layer"]) if finite.size else None,
        "any_layer_specific": bool(finite.size and np.nanmax(spec) > 1.0),
        "reading": None,
    }
    payload["reading"] = (
        f"specificity peaks at {payload['max_specificity']:.2f} (layer "
        f"{payload['argmax_layer']}): a layer exists where the readout responds to the "
        "injected content, so AUC-based layer selection missed it"
        if payload["any_layer_specific"] else
        "specificity stays at or below 1 across every layer tested: the null result is "
        "not explained by layer choice")
    config.dump_json(config.RESULTS / "layer_specificity.json", payload)

    print(f"  steered={args.persona}  source={args.source}")
    print(f"  {'layer':>6s} {'target':>9s} {'random':>9s} {'S = |t|/|r|':>12s}")
    for r in rows:
        print(f"  {r['layer']:6d} {r['slope_target']:9.4f} {r['slope_random']:9.4f} "
              f"{r['specificity']:12.2f}")
    print(f"  -> {payload['reading']}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        L = [r["layer"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(L, [abs(r["slope_target"]) for r in rows], "o-",
                color="#1b4965", label=f"|slope| of {args.source} direction")
        ax.plot(L, [abs(r["slope_random"]) for r in rows], "s--",
                color="#bc4b51", label="|slope| of random direction")
        ax.set_xlabel("intervention layer")
        ax.set_ylabel(r"$|\partial P(\mathrm{True})/\partial\alpha|$")
        ax.grid(color="#e6e6e6")
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)
        ax2 = ax.twinx()
        ax2.plot(L, spec, "^:", color="0.35", label="specificity S")
        ax2.axhline(1.0, color="0.6", lw=0.9, ls=":")
        ax2.set_ylabel("specificity  S = |target| / |random|")
        ax2.legend(fontsize=8, loc="lower right")
        ax.set_title("Does layer choice explain the null?", fontsize=11)
        for ext in ("png", "pdf"):
            fig.savefig(config.RESULTS / f"fig7_layer_specificity.{ext}",
                        dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  [fig] {config.RESULTS / 'fig7_layer_specificity.png'}")


if __name__ == "__main__":
    main()
