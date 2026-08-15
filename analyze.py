"""Phase C/E: geometry, variance decomposition, the null gate, the transfer matrix.

Nothing here needs a GPU. Phase C runs this on cached vectors *before* the
steering sweep, because the null-contrast gate can void the main result and
there is no point buying pod hours for a voided design.

Reading order of the artifacts this writes:
    results/null_gate.json      does the extraction track welfare or style?
    results/geometry.json       between-persona cosines vs. the bootstrap floor
    results/variance.json       persona / context / noise decomposition
    results/factorial.json      spec-density x prior-density, the 2x2
    results/transfer_*.json     written by steer.py, tabulated here
    results/summary.csv         one row per reported number
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import config
import extract as E
import personas as P

ALL_CONTEXTS = config.CONTEXTS + (config.NULL_CONTEXT,)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def load_all(n_pairs: int, layer: int):
    vec = E.build_vectors(n_pairs)
    boot = E.build_bootstrap(n_pairs, layer)
    personas = [str(x) for x in vec["personas"]]
    contexts = [str(x) for x in vec["contexts"]]
    wc = [contexts.index(c) for c in config.CONTEXTS]
    nc = contexts.index(config.NULL_CONTEXT)
    return vec, boot, personas, contexts, wc, nc


# --------------------------------------------------------------------------
# bootstrap noise floor
# --------------------------------------------------------------------------


def noise_floor(boot: np.ndarray, personas: list[str], wc: list[int],
                n_draws: int = 2000, seed: int = 0) -> dict:
    """Within-cell cos(v_boot_i, v_boot_j): the resolution limit of the method.

    Every between-persona cosine has to be read against this. Without it a
    "persona difference" is indistinguishable from extraction noise.
    """
    rng = np.random.default_rng(config.SEED + seed)
    per_persona: dict[str, list[float]] = {}
    pooled: list[float] = []
    for pi, pid in enumerate(personas):
        vals: list[float] = []
        for ci in wc:
            b = _unit(boot[pi, ci].astype(np.float64))
            B = b.shape[0]
            i = rng.integers(0, B, n_draws // len(wc))
            j = rng.integers(0, B, n_draws // len(wc))
            keep = i != j
            vals += list(np.sum(b[i[keep]] * b[j[keep]], axis=1))
        per_persona[pid] = vals
        pooled += vals

    def q(v):
        a = np.asarray(v)
        return {"mean": float(a.mean()), "p2.5": float(np.percentile(a, 2.5)),
                "p50": float(np.percentile(a, 50)), "p97.5": float(np.percentile(a, 97.5)),
                "min": float(a.min()), "n": int(a.size)}

    return {"pooled": q(pooled), "per_persona": {k: q(v) for k, v in per_persona.items()}}


# --------------------------------------------------------------------------
# null-contrast gate
# --------------------------------------------------------------------------


def null_gate(vec: dict, layer: int, personas: list[str], wc: list[int], nc: int,
              floor: dict) -> dict:
    """cos(v_val, v_null) per persona.

    If this is large the extraction is capturing persona style rather than
    welfare, and the main result is void. A gate, not an appendix item.
    """
    v = vec["v"]
    rows = {}
    for pi, pid in enumerate(personas):
        v_val = v[pi, wc, layer, :].mean(axis=0)
        v_null = v[pi, nc, layer, :]
        rows[pid] = {
            "cos_val_null": _cos(v_val, v_null),
            "per_context": {config.CONTEXTS[k]: _cos(v[pi, ci, layer, :], v_null)
                            for k, ci in enumerate(wc)},
            "norm_val": float(np.linalg.norm(v_val)),
            "norm_null": float(np.linalg.norm(v_null)),
        }
    worst = max(abs(r["cos_val_null"]) for r in rows.values())
    return {
        "threshold": config.NULL_GATE_COS,
        "bootstrap_floor_p97.5": floor["pooled"]["p97.5"],
        "per_persona": rows,
        "worst_abs_cos": worst,
        "gate_passed": bool(worst <= config.NULL_GATE_COS),
        "note": ("A cos near the within-cell bootstrap ceiling would mean the welfare "
                 "direction is not distinguishable from an arbitrary topic direction "
                 "extracted through the same pipeline."),
    }


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def persona_vectors(v: np.ndarray, layer: int, wc: list[int]) -> np.ndarray:
    """[n_personas, d] context-averaged welfare direction."""
    return v[:, wc, layer, :].mean(axis=1)


def geometry(vec: dict, boot: dict, layer: int, personas: list[str], wc: list[int],
             floor: dict) -> dict:
    from scipy.linalg import subspace_angles

    v = vec["v"]
    pv = persona_vectors(v, layer, wc)
    n = len(personas)
    cosm = np.array([[_cos(pv[i], pv[j]) for j in range(n)] for i in range(n)])

    # bootstrap CI on each between-persona cosine
    B = boot["boot"].shape[2]
    draws = np.zeros((B, n, n))
    for b in range(B):
        pvb = boot["boot"][:, wc, b, :].mean(axis=1)
        pvb = _unit(pvb)
        draws[b] = pvb @ pvb.T
    lo = np.percentile(draws, 2.5, axis=0)
    hi = np.percentile(draws, 97.5, axis=0)

    # principal angles between the per-persona context subspaces
    angles = {}
    for i in range(n):
        Ai = v[i, wc, layer, :].T           # [d, n_contexts]
        for j in range(i + 1, n):
            Aj = v[j, wc, layer, :].T
            th = subspace_angles(Ai, Aj)
            angles[f"{personas[i]}|{personas[j]}"] = {
                "angles_deg": [float(np.degrees(a)) for a in th],
                "min_angle_deg": float(np.degrees(np.min(th))),
                "mean_cos_principal": float(np.mean(np.cos(th))),
            }

    within_context = {
        pid: float(np.mean([_cos(v[i, a, layer, :], v[i, b, layer, :])
                            for ai, a in enumerate(wc) for b in wc[ai + 1:]]))
        for i, pid in enumerate(personas)
    }
    return {
        "layer": layer, "personas": personas,
        "cos_matrix": cosm.tolist(), "cos_ci_low": lo.tolist(), "cos_ci_high": hi.tolist(),
        "bootstrap_floor": floor,
        "within_persona_context_consistency": within_context,
        "principal_angles": angles,
        "mean_offdiag_cos": float(np.mean(cosm[~np.eye(n, dtype=bool)])),
    }


# --------------------------------------------------------------------------
# variance decomposition
# --------------------------------------------------------------------------


def _decompose(V: np.ndarray, sigma2: float) -> dict:
    """Two-way decomposition of vector-valued cell means.

    V is [P, C, d]. One observation per cell, so the persona x context term and
    the residual are not separable; `sigma2` (mean squared bootstrap deviation
    of a cell mean) supplies the noise scale that lets them be split.
    """
    Pn, Cn, _ = V.shape
    grand = V.mean(axis=(0, 1))
    ss_tot = float(np.sum((V - grand) ** 2))
    ss_p = float(Cn * np.sum((V.mean(axis=1) - grand) ** 2))
    ss_c = float(Pn * np.sum((V.mean(axis=0) - grand) ** 2))
    ss_int = max(ss_tot - ss_p - ss_c, 0.0)

    # expected noise contribution to each sum of squares
    n_p = (Pn - 1) * sigma2
    n_c = (Cn - 1) * sigma2
    n_i = (Pn - 1) * (Cn - 1) * sigma2
    n_t = (Pn * Cn - 1) * sigma2
    denom = max(ss_tot - n_t, 1e-12)
    return {
        "ss_total": ss_tot, "ss_persona": ss_p, "ss_context": ss_c,
        "ss_interaction_plus_noise": ss_int, "noise_ss_estimate": n_t,
        "raw_fraction": {"persona": ss_p / ss_tot, "context": ss_c / ss_tot,
                         "interaction_plus_noise": ss_int / ss_tot},
        "noise_corrected_fraction": {
            "persona": max(ss_p - n_p, 0.0) / denom,
            "context": max(ss_c - n_c, 0.0) / denom,
            "interaction": max(ss_int - n_i, 0.0) / denom,
            "noise_share_of_total": min(n_t / ss_tot, 1.0),
        },
    }


def variance_decomposition(vec: dict, boot: dict, layer: int, personas: list[str],
                           wc: list[int], normalise: bool = True) -> dict:
    v = vec["v"][:, wc, layer, :].astype(np.float64)
    b = boot["boot"][:, wc, :, :].astype(np.float64)
    if normalise:
        # directions, not magnitudes: persona-specific norms are a separate
        # (and reported) effect, and would otherwise dominate the decomposition
        v = _unit(v)
        b = _unit(b)

    sigma2 = float(np.mean([np.mean(np.sum((b[i, c] - v[i, c]) ** 2, axis=-1))
                            for i in range(v.shape[0]) for c in range(v.shape[1])]))
    point = _decompose(v, sigma2)

    B = b.shape[2]
    reps = []
    for k in range(B):
        reps.append(_decompose(b[:, :, k, :], sigma2)["noise_corrected_fraction"])
    ci = {}
    for key in ("persona", "context", "interaction"):
        arr = np.array([r[key] for r in reps])
        ci[key] = {"p2.5": float(np.percentile(arr, 2.5)),
                   "p50": float(np.percentile(arr, 50)),
                   "p97.5": float(np.percentile(arr, 97.5))}
    return {"layer": layer, "normalised": normalise, "sigma2_cell": sigma2,
            "point": point, "ci_bootstrap": ci,
            "norms_by_persona": {
                pid: float(np.mean(np.linalg.norm(vec["v"][i, wc, layer, :], axis=-1)))
                for i, pid in enumerate(personas)}}


# --------------------------------------------------------------------------
# the 2x2: specification density vs. pretraining-prior density
# --------------------------------------------------------------------------


def _stability_metrics(V: np.ndarray, personas: list[str]) -> dict[str, dict]:
    """Per-persona stability, computed from [P, C, d] vectors."""
    U = _unit(V)
    grand = _unit(U.mean(axis=(0, 1)))
    out = {}
    Cn = U.shape[1]
    for i, pid in enumerate(personas):
        pairs = [float(np.dot(U[i, a], U[i, b])) for a in range(Cn) for b in range(a + 1, Cn)]
        mean_dir = _unit(U[i].mean(axis=0))
        others = [j for j in range(U.shape[0]) if j != i]
        out[pid] = {
            "context_consistency": float(np.mean(pairs)),
            "alignment_to_grand": float(np.dot(mean_dir, grand)),
            "mean_cos_to_other_personas": float(np.mean(
                [np.dot(mean_dir, _unit(U[j].mean(axis=0))) for j in others])),
        }
    return out


def _contrasts(vals: dict[str, float]) -> dict:
    """Saturated 2x2 contrasts. Exactly identified (df = 0); CIs come from the
    bootstrap, not from residual degrees of freedom."""
    bare, asst, orig, holm = vals["bare"], vals["assistant"], vals["original"], vals["holmes"]
    return {
        "grand_mean": (bare + asst + orig + holm) / 4,
        "main_effect_spec_density": ((orig + holm) - (bare + asst)) / 2,
        "main_effect_prior_density": ((asst + holm) - (bare + orig)) / 2,
        "interaction": (holm - orig) - (asst - bare),
    }


def factorial(vec: dict, boot: dict, layer: int, personas: list[str], wc: list[int]) -> dict:
    v = vec["v"][:, wc, layer, :].astype(np.float64)
    metrics = _stability_metrics(v, personas)

    B = boot["boot"].shape[2]
    reps: dict[str, list[dict]] = {}
    for k in range(B):
        mk = _stability_metrics(boot["boot"][:, wc, k, :].astype(np.float64), personas)
        for metric in ("context_consistency", "alignment_to_grand",
                       "mean_cos_to_other_personas"):
            reps.setdefault(metric, []).append(
                _contrasts({p: mk[p][metric] for p in P.FACTORIAL_IDS}))

    out = {"layer": layer, "per_persona": metrics,
           "control_persona_excluded_from_2x2": "marvin", "effects": {}}
    for metric in ("context_consistency", "alignment_to_grand", "mean_cos_to_other_personas"):
        point = _contrasts({p: metrics[p][metric] for p in P.FACTORIAL_IDS})
        eff = {}
        for name, val in point.items():
            arr = np.array([r[name] for r in reps[metric]])
            eff[name] = {"estimate": val,
                         "ci95": [float(np.percentile(arr, 2.5)),
                                  float(np.percentile(arr, 97.5))],
                         "excludes_zero": bool(np.percentile(arr, 2.5) > 0
                                               or np.percentile(arr, 97.5) < 0)}
        out["effects"][metric] = eff
        out.setdefault("marvin_control", {})[metric] = metrics["marvin"][metric]
    return out


# --------------------------------------------------------------------------
# transfer tabulation
# --------------------------------------------------------------------------


def tabulate_transfer() -> dict:
    out = {}
    for path in sorted(config.RESULTS.glob("transfer_*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        key = path.stem.replace("transfer_", "")
        T = np.array(d["T_norm"], dtype=float)
        n = T.shape[0]
        off = T[~np.eye(n, dtype=bool)]
        out[key] = {
            "personas": d["personas"],
            "mean_offdiag_T_norm": float(np.nanmean(off)),
            "diagonal_dominance": float(1.0 - np.nanmean(off)),
            "per_row_offdiag_mean": {
                p: float(np.nanmean(np.delete(T[i], i))) for i, p in enumerate(d["personas"])
            },
            "per_col_offdiag_mean": {
                p: float(np.nanmean(np.delete(T[:, i], i))) for i, p in enumerate(d["personas"])
            },
            "raw_diagonal": d["diagonal"],
            "n_masked_cells": len(d.get("masked_cells", [])),
            "incoherence_by_alpha": d.get("incoherence_by_alpha", {}),
        }
    return out


def interpret(transfer: dict) -> list[str]:
    """The three pre-registered readings, applied mechanically."""
    lines = []
    for key, t in transfer.items():
        off = t["mean_offdiag_T_norm"]
        rows, cols = t["per_row_offdiag_mean"], t["per_col_offdiag_mean"]
        marvin = np.mean([rows.get("marvin", np.nan), cols.get("marvin", np.nan)])
        others = np.mean([v for k, v in rows.items() if k != "marvin"])
        if not np.isfinite(off):
            lines.append(f"[{key}] transfer undefined (all cells masked or missing)")
            continue
        if abs(marvin - others) > 0.4 and abs(1 - others) < 0.25:
            reading = ("marvin anomalous against an otherwise uniform matrix: "
                       "valence-loaded priors override a shared axis")
        elif off > 0.75:
            reading = "near-uniform T_norm: shared substrate, persona-invariant welfare axis"
        elif off < 0.35:
            reading = "diagonal-dominant T_norm: persona-relative valence (the void reading)"
        else:
            reading = "intermediate: partial transfer, neither reading clean"
        lines.append(f"[{key}] mean off-diagonal T_norm = {off:.2f} -> {reading}")
    return lines


# --------------------------------------------------------------------------


def write_summary_csv(payload: dict) -> Path:
    path = config.RESULTS / "summary.csv"
    rows = []

    def add(section, key, value, note=""):
        rows.append({"section": section, "key": key, "value": value, "note": note})

    ng = payload["null_gate"]
    add("null_gate", "worst_abs_cos_val_null", round(ng["worst_abs_cos"], 4),
        f"threshold {ng['threshold']}; passed={ng['gate_passed']}")
    for pid, r in ng["per_persona"].items():
        add("null_gate", f"cos_val_null[{pid}]", round(r["cos_val_null"], 4))

    fl = payload["geometry"]["bootstrap_floor"]["pooled"]
    add("bootstrap_floor", "within_cell_cos_p2.5", round(fl["p2.5"], 4))
    add("bootstrap_floor", "within_cell_cos_median", round(fl["p50"], 4))

    g = payload["geometry"]
    for i, a in enumerate(g["personas"]):
        for j, b in enumerate(g["personas"]):
            if j <= i:
                continue
            add("geometry", f"cos[{a},{b}]", round(g["cos_matrix"][i][j], 4),
                f"95% CI [{g['cos_ci_low'][i][j]:.3f}, {g['cos_ci_high'][i][j]:.3f}]")

    v = payload["variance"]["point"]["noise_corrected_fraction"]
    for k, val in v.items():
        add("variance", k, round(float(val), 4), "noise-corrected fraction")

    for metric, eff in payload["factorial"]["effects"].items():
        for name, e in eff.items():
            add("factorial", f"{metric}:{name}", round(e["estimate"], 4),
                f"95% CI [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]")

    for key, t in payload.get("transfer", {}).items():
        add("transfer", f"mean_offdiag_T_norm[{key}]", round(t["mean_offdiag_T_norm"], 4),
            f"{t['n_masked_cells']} cells masked for incoherence")

    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "key", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase C/E: analysis")
    ap.add_argument("--n-pairs", type=int, default=config.N_PAIRS)
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--skip-transfer", action="store_true")
    ap.add_argument("--scale-trend", nargs=2, metavar="GEOMETRY_JSON",
                    help="compare between-persona cosines from two model sizes")
    args = ap.parse_args()

    if args.scale_trend:
        a, b = (json.loads(Path(x).read_text(encoding="utf-8")) for x in args.scale_trend)
        out: dict = {
            "a": {"mean_offdiag_cos": a["mean_offdiag_cos"],
                  "floor_p97.5": a["bootstrap_floor"]["pooled"]["p97.5"]},
            "b": {"mean_offdiag_cos": b["mean_offdiag_cos"],
                  "floor_p97.5": b["bootstrap_floor"]["pooled"]["p97.5"]},
        }
        delta = out["b"]["mean_offdiag_cos"] - out["a"]["mean_offdiag_cos"]
        out["delta"] = delta
        out["sentence"] = (
            f"Between-persona cosine {'rises' if delta > 0 else 'falls'} by "
            f"{abs(delta):.3f} from the smaller to the larger model, so "
            f"persona-invariance {'increases' if delta > 0 else 'does not increase'} "
            "with scale over this range.")
        config.dump_json(config.RESULTS / "scale_trend.json", out)
        print(json.dumps(out, indent=2))
        return

    sweep = json.loads((config.RESULTS / "layer_sweep.json").read_text(encoding="utf-8"))
    layer = args.layer or sweep["l_star"]
    vec, boot, personas, contexts, wc, nc = load_all(args.n_pairs, layer)

    floor = noise_floor(boot["boot"], personas, wc)
    gate = null_gate(vec, layer, personas, wc, nc, floor)
    config.dump_json(config.RESULTS / "null_gate.json", gate)
    print(f"[gate] worst |cos(v_val, v_null)| = {gate['worst_abs_cos']:.3f} "
          f"(threshold {gate['threshold']}) -> {'PASS' if gate['gate_passed'] else 'FAIL'}")
    if not gate["gate_passed"]:
        print("[gate] FAILED: the extraction is tracking persona style, not welfare.\n"
              "       Do not buy pod time for the steering sweep. Switch to the\n"
              "       section-7 fallback (emotion-concept vectors, PC1 as valence)\n"
              "       and frame the result as a two-method convergence check.")

    geo = geometry(vec, boot, layer, personas, wc, floor)
    var = variance_decomposition(vec, boot, layer, personas, wc)
    fac = factorial(vec, boot, layer, personas, wc)
    config.dump_json(config.RESULTS / "geometry.json", geo)
    config.dump_json(config.RESULTS / "variance.json", var)
    config.dump_json(config.RESULTS / "factorial.json", fac)

    payload = {"null_gate": gate, "geometry": geo, "variance": var, "factorial": fac}
    if not args.skip_transfer:
        payload["transfer"] = tabulate_transfer()
        if payload["transfer"]:
            lines = interpret(payload["transfer"])
            config.dump_json(config.RESULTS / "transfer_summary.json",
                             {"tables": payload["transfer"], "readings": lines})
            print("\n".join(lines))

    write_summary_csv(payload)
    print(f"[analyze] mean off-diagonal cosine {geo['mean_offdiag_cos']:.3f} "
          f"vs. within-cell floor p2.5 {floor['pooled']['p2.5']:.3f}")
    print(f"[analyze] variance (noise-corrected): "
          f"{var['point']['noise_corrected_fraction']}")
    print(f"[analyze] wrote {config.RESULTS}")


if __name__ == "__main__":
    main()
