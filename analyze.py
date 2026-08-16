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


def effective_dimension(vec: dict, boot: dict, layer: int, personas: list[str],
                        wc: list[int]) -> dict:
    """How many dimensions are needed to span the 20 cell directions?

    This is a *representational* dimensionality across cells, not the intrinsic
    dimension of welfare within a cell -- difference-in-means yields one vector
    per cell by construction, so a within-cell subspace is not recoverable here
    and is not claimed.

    The spectrum alone is uninterpretable, because extraction noise inflates
    every trailing component. So it is read against a null in which all 20 cells
    share ONE true direction and differ only by their own bootstrap noise. A
    component counts as real only if its variance share clears that null's 97.5th
    percentile at the same index.
    """
    v = vec["v"][:, wc, layer, :].astype(np.float64)          # [P, C, d]
    b = boot["boot"][:, wc, :, :].astype(np.float64)          # [P, C, B, d]
    P_, C_, d = v.shape
    U = _unit(v).reshape(P_ * C_, d)

    def _spec(M):
        s = np.linalg.svd(M, compute_uv=False)
        e = s ** 2
        tot = float(e.sum()) + 1e-12
        share = e / tot
        return share, float(tot ** 2 / (np.sum(e ** 2) + 1e-12))   # participation ratio

    share, pr = _spec(U)

    # null: one shared direction + each cell's own extraction noise
    grand = _unit(v.reshape(-1, d).mean(axis=0))
    B = b.shape[2]
    null_shares, null_prs = [], []
    rng = np.random.default_rng(config.SEED + 11)
    for _ in range(min(B, 200)):
        rows = []
        for i in range(P_):
            for c in range(C_):
                k = int(rng.integers(0, B))
                resid = _unit(b[i, c, k]) - _unit(v[i, c])   # noise of this cell
                rows.append(_unit(grand + resid))
            # (grand direction perturbed by the cell's own resampling noise)
        s_null, pr_null = _spec(np.array(rows))
        null_shares.append(s_null)
        null_prs.append(pr_null)
    null_shares = np.array(null_shares)
    hi = np.percentile(null_shares, 97.5, axis=0)

    n_sig = int(np.sum(share[: len(hi)] > hi))
    cum = np.cumsum(share)
    n90 = int(np.searchsorted(cum, 0.90) + 1)
    return {
        "layer": layer,
        "n_cells": P_ * C_,
        "variance_share": share[:10].tolist(),
        "cumulative_share": cum[:10].tolist(),
        "null_share_p97.5": hi[:10].tolist(),
        # headline statistic: the standard effective-dimension estimator
        "participation_ratio": pr,
        "participation_ratio_null": {
            "p2.5": float(np.percentile(null_prs, 2.5)),
            "p50": float(np.percentile(null_prs, 50)),
            "p97.5": float(np.percentile(null_prs, 97.5)),
        },
        "n_components_90pct": n90,
        "n_components_above_null": n_sig,
        "pc1_share": float(share[0]),
        "note": (
            "Report the participation ratio against its null interval: that is the "
            "effective number of distinct directions among the cell vectors. "
            "n_components_above_null is a LIBERAL upper bound and should not be the "
            "headline -- in a one-shared-direction null, PC1 absorbs nearly all the "
            "variance, leaving a tiny per-component noise share that most observed "
            "components clear. This measures dimensionality ACROSS cells, not the "
            "intrinsic dimension of welfare within a cell, which difference-in-means "
            "cannot recover."),
    }


def transfer_rank(transfer_path: Path | None = None) -> dict:
    """Is T functionally one-dimensional?

    T[p][q] ~ r_p * c_q means every persona has a steerability gain, every vector
    has a quality, and nothing depends on the pairing -- a single functional axis.
    A rank-1 fit that leaves residuals inside their own 95% CIs is much stronger
    evidence than a mean off-diagonal, because it constrains all 25 cells at once.
    """
    path = transfer_path or (config.RESULTS / "transfer_avg.json")
    if not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    T = np.array(d["T"], dtype=float)
    ci = np.array(d.get("ci95_halfwidth", np.zeros_like(T)), dtype=float)
    ok = np.isfinite(T)
    if ok.sum() < T.size - 2:      # too many masked cells to factorise
        return {"rank1_fit": None, "reason": "too many masked cells"}

    M = np.where(ok, T, np.nanmean(T[ok]))
    U, s, Vt = np.linalg.svd(M)
    rank1 = s[0] * np.outer(U[:, 0], Vt[0])
    resid = M - rank1
    within = np.abs(resid) <= np.where(np.isfinite(ci) & (ci > 0), ci, np.inf)
    return {
        "singular_values": s.tolist(),
        "rank1_variance_explained": float(s[0] ** 2 / np.sum(s ** 2)),
        "rank2_cumulative": float(np.sum(s[:2] ** 2) / np.sum(s ** 2)),
        "row_factor": dict(zip(d["personas"], (U[:, 0] * np.sqrt(s[0])).tolist())),
        "col_factor": dict(zip(d["personas"], (Vt[0] * np.sqrt(s[0])).tolist())),
        "max_abs_residual": float(np.max(np.abs(resid))),
        "residuals_within_ci": bool(within.all()),
        "n_residuals_outside_ci": int((~within).sum()),
        "reading": ("rank-1 suffices: one functional axis with per-persona gain"
                    if float(s[0] ** 2 / np.sum(s ** 2)) > 0.90
                    else "rank-1 insufficient: pair-specific structure in T"),
    }


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

    if "dimensionality" in payload:
        dm = payload["dimensionality"]
        add("dimensionality", "participation_ratio", round(dm["participation_ratio"], 4),
            f"null 95% [{dm['participation_ratio_null']['p2.5']:.2f}, "
            f"{dm['participation_ratio_null']['p97.5']:.2f}] -- headline statistic")
        add("dimensionality", "pc1_variance_share", round(dm["pc1_share"], 4))
        add("dimensionality", "n_components_90pct", dm["n_components_90pct"])
        add("dimensionality", "n_components_above_null", dm["n_components_above_null"],
            "liberal upper bound, not the headline")
    if "transfer_rank" in payload and payload["transfer_rank"].get("singular_values"):
        tr = payload["transfer_rank"]
        add("transfer_rank", "rank1_variance_explained",
            round(tr["rank1_variance_explained"], 4), tr["reading"])
        add("transfer_rank", "residuals_within_ci", tr["residuals_within_ci"],
            f"{tr['n_residuals_outside_ci']} cells outside their 95% CI")

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
    dim = effective_dimension(vec, boot, layer, personas, wc)
    config.dump_json(config.RESULTS / "geometry.json", geo)
    config.dump_json(config.RESULTS / "variance.json", var)
    config.dump_json(config.RESULTS / "factorial.json", fac)
    config.dump_json(config.RESULTS / "dimensionality.json", dim)
    print(f"[analyze] effective dim across cells: participation ratio "
          f"{dim['participation_ratio']:.2f} vs null median "
          f"{dim['participation_ratio_null']['p50']:.2f} "
          f"[{dim['participation_ratio_null']['p2.5']:.2f}, "
          f"{dim['participation_ratio_null']['p97.5']:.2f}]; "
          f"PC1 = {dim['pc1_share']:.2f}, {dim['n_components_90pct']} components to 90%")

    payload = {"null_gate": gate, "geometry": geo, "variance": var, "factorial": fac,
               "dimensionality": dim}
    if not args.skip_transfer:
        payload["transfer"] = tabulate_transfer()
        rank = transfer_rank()
        if rank:
            payload["transfer_rank"] = rank
            config.dump_json(config.RESULTS / "transfer_rank.json", rank)
            if rank.get("rank1_variance_explained") is not None:
                print(f"[analyze] T rank-1 explains "
                      f"{rank['rank1_variance_explained']:.3f} of variance -> "
                      f"{rank['reading']}")
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
