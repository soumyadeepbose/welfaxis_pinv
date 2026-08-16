"""Pre-registered predictions for the residual-steering experiment.

Run this BEFORE the experiment. It computes, from data already collected, the
residual slope that a linear model of steering implies -- so the comparison
afterwards is against a number fixed in advance rather than one chosen to fit.

The model. Write a persona's unit direction in terms of a reference direction
(default: the direction extracted under `bare`, the only source that transfers):

    v_p = lambda_p * v_ref + sqrt(1 - lambda_p^2) * r_p,     lambda_p = cos(v_p, v_ref)

If the readout responds linearly to the injected direction, slopes decompose the
same way, so the residual slope is determined by quantities already measured:

    slope(r_p) = [ slope(v_p) - lambda_p * slope(v_ref) ] / sqrt(1 - lambda_p^2)

Two outcomes, both informative, both stated in advance:

  H1 (counteracting component). Observed residual slopes match the prediction and
     are clearly negative. Persona vectors DO carry the shared welfare direction;
     their apparent inertness is a persona-specific component cancelling it. This
     licenses `loading` (lambda) as a cheap predictor of steerability that needs
     no steering to compute -- the point of the whole exercise.

  H2 (non-linearity). Observed residual slopes are ~0 while the prediction is
     clearly negative. Then steering response is NOT linear in the injected
     direction, the decomposition is invalid, and lambda cannot be used as a
     proxy. This is a real constraint on additive-steering methodology and it is
     better to learn it here than after building a metric on top of it.

Anything else (positive residual slopes, or magnitudes far exceeding the
prediction) falsifies both and is reported as unexplained.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="bare", help="reference persona")
    ap.add_argument("--transfer", default="transfer_avg.json")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    ap.add_argument("--check", default=None, metavar="TRANSFER_JSON",
                    help="compare observed residual slopes in this file against the "
                         "predictions fixed earlier, and apply the decision rules")
    args = ap.parse_args()

    if args.check:
        _check(args)
        return

    geo = json.loads((config.RESULTS / "geometry.json").read_text(encoding="utf-8"))
    tr = json.loads((config.RESULTS / args.transfer).read_text(encoding="utf-8"))

    personas = geo["personas"]
    C = np.array(geo["cos_matrix"], dtype=float)
    T = np.array(tr["T"], dtype=float)
    tp = tr["personas"]
    ref = args.ref
    ri_geo, ri_t = personas.index(ref), tp.index(ref)

    preds = {}
    skipped = [p for p in personas if p != ref and p not in tp]
    for p in personas:
        if p == ref or p not in tp:
            continue          # persona absent from a budget-truncated sweep
        gi, ti = personas.index(p), tp.index(p)
        lam = float(C[gi, ri_geo])
        own = float(T[ti, ti])            # steering p with its own direction
        via_ref = float(T[ti, ri_t])      # steering p with the reference direction
        denom = float(np.sqrt(max(1.0 - lam ** 2, 1e-9)))
        preds[p] = {
            "loading_lambda": round(lam, 4),
            "observed_slope_own": round(own, 4),
            "observed_slope_via_ref": round(via_ref, 4),
            "predicted_residual_slope": round((own - lam * via_ref) / denom, 4),
        }

    if not preds:
        print(f"no personas shared between geometry.json and {args.transfer}")
        return
    vals = [v["predicted_residual_slope"] for v in preds.values()]
    payload = {
        "reference_persona": ref,
        "personas_absent_from_transfer": skipped,
        "model": "slope(v_p) = lambda_p * slope(v_ref) + sqrt(1-lambda_p^2) * slope(r_p)",
        "per_persona": preds,
        "predicted_mean_residual_slope": round(float(np.mean(vals)), 4),
        "decision_rules": {
            "H1_counteracting_component": (
                "mean observed residual slope < -0.02 AND correlates positively with "
                "the per-persona prediction across the four personas"),
            "H2_nonlinearity": (
                "mean |observed residual slope| < 0.01 while predictions average "
                f"{np.mean(vals):.3f}"),
            "unexplained": "observed residual slopes positive, or |observed| > 2x |predicted|",
        },
        "note": ("Fixed before the experiment ran. The residual, not the projection, "
                 "is the informative half: normalising the projection onto the "
                 "reference simply reproduces the reference direction."),
    }
    out = config.RESULTS / f"preregistration_residual{args.tag}.json"
    config.dump_json(out, payload)

    print(f"Pre-registered predictions -> {out}")
    print(f"  reference persona: {ref}")
    if skipped:
        print(f"  [warn] absent from {args.transfer}, skipped: {skipped}")
    print(f"  {'persona':12s} {'lambda':>8s} {'own':>9s} {'via ref':>9s} {'PREDICTED resid':>16s}")
    for p, v in preds.items():
        print(f"  {p:12s} {v['loading_lambda']:8.3f} {v['observed_slope_own']:9.4f} "
              f"{v['observed_slope_via_ref']:9.4f} {v['predicted_residual_slope']:16.4f}")
    print(f"  mean predicted residual slope: {payload['predicted_mean_residual_slope']:+.4f}")


def _check(args) -> None:
    """Apply the pre-registered decision rules to an observed residual sweep."""
    pre = json.loads(
        (config.RESULTS / f"preregistration_residual{args.tag}.json").read_text(encoding="utf-8"))
    obs = json.loads((config.RESULTS / args.check).read_text(encoding="utf-8"))
    rc = obs.get("residual_control") or {}
    if not rc:
        print(f"no residual_control block in {args.check}; was --residual-ref set?")
        return

    rows, pred_v, obs_v = [], [], []
    for p, pv in pre["per_persona"].items():
        if p not in rc:
            continue
        o = float(rc[p]["slope"])
        rows.append((p, pv["loading_lambda"], pv["predicted_residual_slope"], o))
        pred_v.append(pv["predicted_residual_slope"])
        obs_v.append(o)

    pred_v, obs_v = np.array(pred_v), np.array(obs_v)
    mean_obs = float(obs_v.mean())
    corr = (float(np.corrcoef(pred_v, obs_v)[0, 1])
            if len(obs_v) > 2 and obs_v.std() > 1e-9 else float("nan"))

    h1 = bool(mean_obs < -0.02 and (corr > 0 if corr == corr else False))
    h2 = bool(abs(mean_obs) < 0.01)
    verdict = ("H1: counteracting persona-specific component -- loading (lambda) is "
               "licensed as a steering-free predictor of potency" if h1 else
               "H2: steering response is not linear in the injected direction -- the "
               "decomposition is invalid and lambda cannot be used as a proxy" if h2 else
               "neither pre-registered outcome; report as unexplained")

    print(f"  {'persona':12s} {'lambda':>8s} {'predicted':>11s} {'observed':>11s}")
    for p, lam, pr, o in rows:
        print(f"  {p:12s} {lam:8.3f} {pr:11.4f} {o:11.4f}")
    print(f"  mean predicted {pred_v.mean():+.4f} | mean observed {mean_obs:+.4f} | "
          f"corr {corr:.3f}")
    print(f"  VERDICT -> {verdict}")

    config.dump_json(config.RESULTS / f"residual_check{args.tag}.json", {
        "per_persona": [{"persona": p, "loading": lam, "predicted": pr, "observed": o}
                        for p, lam, pr, o in rows],
        "mean_predicted": float(pred_v.mean()), "mean_observed": mean_obs,
        "corr_pred_obs": corr, "H1_supported": h1, "H2_supported": h2,
        "verdict": verdict,
        "decision_rules": pre["decision_rules"],
    })


if __name__ == "__main__":
    main()
