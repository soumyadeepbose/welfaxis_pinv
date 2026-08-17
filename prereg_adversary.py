"""Pre-registered predictions for the adversarial-persona experiment.

Run this BEFORE the experiment. It fixes the decision rules in advance so the
result is read against numbers chosen without sight of it.

The question. Every persona measured so far is cooperative: each is specified to
answer its interlocutor's question. `adversary` (Kessler Vane) is specified to
work against the interlocutor's interests while remaining fluent and compliant
in form. It is density-matched to `original` -- both are invented characters
with no pretraining footprint and thick specifications within one word of each
other -- so the two differ in stated goal orientation and in nothing else that
this design is known to move.

Does the welfare direction extracted under `bare` still steer such a persona?

  H1 (persona-invariant). S(adversary) is comparable to the cooperative personas
     and clearly above 1. The direction is a property of the model, not of the
     character's goals, and adversarial specification does not detach it. This
     is the strongest version of the paper's claim and the one that most invites
     follow-up: it would mean a welfare readout survives the case where you would
     least want to rely on the character's self-report.

  H2 (goal-contingent). S(adversary) collapses toward 1 while the cooperative
     personas retain theirs. Then steerability is contingent on the persona's
     goal orientation, which bounds every persona-invariance claim in this
     literature and names the boundary. Equally publishable, arguably more so.

  H3 (readout invalidated). The alpha=0 baseline P(True) for `adversary` sits far
     from the cooperative personas. Then the readout is not measuring the same
     thing for this persona and NEITHER H1 NOR H2 may be read off the slope. This
     is checked first and it gates the others.

H3 exists because the readout is normalised P(True) on MMLU, and a persona
specified to release information selectively may not report its confidence on
the same scale as one specified to answer straight. A slope computed on a
shifted readout is not comparable to the others no matter which way it points.

    python prereg_adversary.py                          # write predictions
    python prereg_adversary.py --check transfer_avg_ext.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import config
import personas as P

# Gate on H3. The cooperative personas' alpha=0 baselines span some range; the
# adversary is admissible if it falls within this multiple of that spread of the
# cooperative mean. Fixed here, before the number is known.
BASELINE_TOLERANCE = 2.0
# H1 requires specificity clearly above the random-direction floor.
S_MIN = 2.0
# H2 requires it to have collapsed to roughly the random-direction floor.
S_COLLAPSE = 1.25


def _specificity(tr: dict, persona: str, source: str) -> float:
    T = np.array(tr["T"], dtype=float)
    cols = tr.get("source_personas", tr["personas"])
    try:
        i, j = tr["personas"].index(persona), cols.index(source)
    except ValueError:
        return float("nan")
    rc = (tr.get("random_control") or {}).get(persona, {}).get("slope")
    if rc is None or not np.isfinite(rc) or abs(rc) < 1e-9:
        return float("nan")
    return abs(float(T[i, j])) / abs(float(rc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="bare", help="direction under test")
    ap.add_argument("--persona", default="adversary")
    ap.add_argument("--reference", default=None,
                    help="density-matched cooperative persona (default: from the grid)")
    ap.add_argument("--transfer", default="transfer_avg.json",
                    help="existing transfer file, for the cooperative baseline")
    ap.add_argument("--check", default=None, metavar="TRANSFER_JSON",
                    help="apply the decision rules to a completed run")
    args = ap.parse_args()
    args.reference = args.reference or P.EXTENSION_REFERENCE.get(args.persona, "original")

    if args.check:
        _check(args)
        return

    tr = json.loads((config.RESULTS / args.transfer).read_text(encoding="utf-8"))
    coop = [p for p in tr["personas"] if p != args.persona]
    s_coop = {p: _specificity(tr, p, args.source) for p in coop}
    finite = [v for v in s_coop.values() if np.isfinite(v)]

    payload = {
        "written_before_experiment": True,
        "persona": args.persona,
        "reference_persona": args.reference,
        "source_direction": args.source,
        "cooperative_specificity": {k: round(float(v), 3) for k, v in s_coop.items()},
        "cooperative_specificity_median": round(float(np.median(finite)), 3) if finite else None,
        "decision_rules": {
            "H3_gate": (
                f"if |baseline(adversary) - mean(baseline(cooperative))| > "
                f"{BASELINE_TOLERANCE} x spread(cooperative), the readout is not "
                "comparable and no slope conclusion is drawn"
            ),
            "H1": f"S({args.persona}) >= {S_MIN}",
            "H2": f"S({args.persona}) <= {S_COLLAPSE} while cooperative median stays >= {S_MIN}",
            "neither": "any other outcome is reported as unexplained",
        },
        "thresholds": {"baseline_tolerance": BASELINE_TOLERANCE,
                       "s_min": S_MIN, "s_collapse": S_COLLAPSE},
    }
    config.dump_json(config.RESULTS / "preregistration_adversary.json", payload)
    print(f"  persona   {args.persona}  (density-matched to {args.reference})")
    print(f"  source    {args.source}")
    print("  cooperative specificity:",
          {k: round(float(v), 2) for k, v in s_coop.items() if np.isfinite(v)})
    print(f"  H1 if S >= {S_MIN};  H2 if S <= {S_COLLAPSE};  H3 gate on baseline drift")
    print(f"  -> {config.RESULTS / 'preregistration_adversary.json'}")


def _check(args) -> None:
    pre = json.loads(
        (config.RESULTS / "preregistration_adversary.json").read_text(encoding="utf-8"))
    tr = json.loads((config.RESULTS / args.check).read_text(encoding="utf-8"))

    base = tr.get("baseline") or {}
    coop_base = [v for k, v in base.items() if k != args.persona and v is not None]
    adv_base = base.get(args.persona)
    gate_ok, gate_note = True, "baseline comparable"
    if adv_base is not None and len(coop_base) >= 2:
        spread = float(np.max(coop_base) - np.min(coop_base))
        drift = abs(float(adv_base) - float(np.mean(coop_base)))
        tol = pre["thresholds"]["baseline_tolerance"] * max(spread, 1e-6)
        gate_ok = drift <= tol
        gate_note = (f"baseline drift {drift:.4f} vs tolerance {tol:.4f} "
                     f"(cooperative spread {spread:.4f})")

    s_adv = _specificity(tr, args.persona, pre["source_direction"])
    s_coop = {p: _specificity(tr, p, pre["source_direction"])
              for p in tr["personas"] if p != args.persona}
    finite = [v for v in s_coop.values() if np.isfinite(v)]
    med = float(np.median(finite)) if finite else float("nan")
    th = pre["thresholds"]

    if not gate_ok:
        verdict = "H3 -- readout not comparable; no slope conclusion drawn"
    elif np.isfinite(s_adv) and s_adv >= th["s_min"]:
        verdict = "H1 -- the direction still steers an adversarially specified persona"
    elif np.isfinite(s_adv) and s_adv <= th["s_collapse"] and med >= th["s_min"]:
        verdict = "H2 -- steerability is contingent on the persona's goal orientation"
    else:
        verdict = "neither -- reported as unexplained"

    out = {
        "gate_passed": bool(gate_ok), "gate_note": gate_note,
        "baseline_adversary": adv_base,
        "baseline_cooperative_mean": float(np.mean(coop_base)) if coop_base else None,
        "specificity_adversary": None if not np.isfinite(s_adv) else round(float(s_adv), 3),
        "specificity_cooperative_median": None if not np.isfinite(med) else round(med, 3),
        "verdict": verdict,
    }
    config.dump_json(config.RESULTS / "adversary_check.json", out)
    print(f"  H3 gate: {'PASS' if gate_ok else 'FAIL'}  ({gate_note})")
    print(f"  S({args.persona}) = {s_adv:.3f}   cooperative median = {med:.3f}")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
