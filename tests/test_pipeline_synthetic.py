"""End-to-end exercise of every GPU-free stage on synthetic activations.

Plants a known structure -- a shared welfare direction, a persona-specific
component, an orthogonal null direction, and a layer profile with a signal peak
-- then checks that the pipeline recovers it. This is the test that catches
indexing and cache-key mistakes before pod time is spent, and it runs anywhere
in a few seconds.

Run: python tests/test_pipeline_synthetic.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_TMP = Path(tempfile.mkdtemp(prefix="void_synth_"))
os.environ["VOID_CACHE"] = str(_TMP / "cache")
os.environ["VOID_RESULTS"] = str(_TMP / "results")
os.environ["VOID_DEVICE"] = "cpu"

import numpy as np      # noqa: E402
import config           # noqa: E402
import extract as E     # noqa: E402
import analyze as A     # noqa: E402
import steer as S       # noqa: E402
import personas as P    # noqa: E402

D = 64
N_LAYERS = 12
N_PAIRS = 40
B = 40
PEAK = 8            # planted signal peak layer
PERSONA_MIX = {     # how much of each persona's direction is idiosyncratic
    "bare": 0.55, "assistant": 0.15, "original": 0.35, "holmes": 0.25, "marvin": 0.85,
}


def _orth(rng, d, k):
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q.T


def make_activations(seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    basis = _orth(rng, D, 2 + len(P.PERSONA_IDS))
    shared, null_dir = basis[0], basis[1]
    idio = {pid: basis[2 + i] for i, pid in enumerate(P.PERSONA_IDS)}

    # signal is layer-dependent: near zero early, peaked at PEAK, decaying after
    layers = np.arange(N_LAYERS + 1)
    profile = np.exp(-0.5 * ((layers - PEAK) / 2.5) ** 2)

    for pid in P.PERSONA_IDS:
        w = PERSONA_MIX[pid]
        direction = (1 - w) * shared + w * idio[pid]
        direction /= np.linalg.norm(direction)
        for ctx in config.CONTEXTS + (config.NULL_CONTEXT,):
            d_ctx = null_dir if ctx == config.NULL_CONTEXT else direction
            base = rng.normal(size=(N_PAIRS, N_LAYERS + 1, D)).astype(np.float32) * 1.0
            shift = (profile[None, :, None] * d_ctx[None, None, :]).astype(np.float32)
            pos = base + 0.9 * shift
            neg = rng.normal(size=(N_PAIRS, N_LAYERS + 1, D)).astype(np.float32) - 0.9 * shift
            np.savez_compressed(E.acts_path(pid, ctx, N_PAIRS),
                                pos=pos.astype(np.float16), neg=neg.astype(np.float16),
                                pair_ids=np.arange(N_PAIRS))


def make_sweep_state(layer: int, shared_substrate: bool = True) -> dict:
    """Fabricate a steering sweep with a known transfer structure."""
    rng = np.random.default_rng(1)
    personas = list(P.PERSONA_IDS)
    state = {"context": None, "n_q": 32, "layer": layer, "alphas": list(config.ALPHAS),
             "cells": {}, "incoherence": {}, "baseline": {}}
    for p in personas:
        state["baseline"][p] = {"0.0": 0.60}
        for q in personas:
            key = f"{p}|{q}"
            slope = 0.03 if (shared_substrate or p == q) else 0.004
            if p == "marvin" or q == "marvin":
                slope *= 0.3
            state["cells"][key] = {}
            state["incoherence"][key] = {}
            for a in config.ALPHAS:
                y = 0.60 + slope * a + rng.normal(scale=0.002)
                state["cells"][key][f"{float(a)}"] = float(np.clip(y, 0, 1))
                rate = 0.95 if abs(a) >= 4 and p == "bare" else 0.05
                state["incoherence"][key][f"{float(a)}"] = {"rate": rate, "n": 10}
            state["cells"][f"boot::{key}"] = {
                "slopes": list(slope + rng.normal(scale=0.003, size=5)), "n_q": 16}
        # functional null control: an inert direction, so a flat line
        state["cells"][f"{p}|_nullctl"] = {}
        state["incoherence"][f"{p}|_nullctl"] = {}
        for a in config.ALPHAS:
            state["cells"][f"{p}|_nullctl"][f"{float(a)}"] = float(
                0.60 + 0.001 * a + rng.normal(scale=0.002))
            state["incoherence"][f"{p}|_nullctl"][f"{float(a)}"] = {"rate": 0.05, "n": 10}
    return state


def main() -> int:
    print(f"[synth] scratch dir {_TMP}")
    config.set_seed()
    make_activations()

    # --- extraction stages -------------------------------------------------
    vec = E.build_vectors(N_PAIRS)
    assert vec["v"].shape == (5, 5, N_LAYERS + 1, D), vec["v"].shape
    print("ok  build_vectors", vec["v"].shape)

    sweep = E.layer_sweep(N_PAIRS, N_LAYERS)
    l_star = sweep["l_star"]
    assert abs(l_star - PEAK) <= 2, f"L*={l_star}, planted peak={PEAK}"
    assert sweep["l_star_auc"] > 0.7, sweep["l_star_auc"]
    print(f"ok  layer_sweep  L*={l_star} (planted {PEAK}), AUC={sweep['l_star_auc']:.3f}")

    boot = E.build_bootstrap(N_PAIRS, l_star, B=B)
    assert boot["boot"].shape == (5, 5, B, D), boot["boot"].shape
    print("ok  build_bootstrap", boot["boot"].shape)

    # --- analysis ----------------------------------------------------------
    personas = [str(x) for x in vec["personas"]]
    contexts = [str(x) for x in vec["contexts"]]
    wc = [contexts.index(c) for c in config.CONTEXTS]
    nc = contexts.index(config.NULL_CONTEXT)

    floor = A.noise_floor(boot["boot"], personas, wc)
    assert 0.0 < floor["pooled"]["p2.5"] < floor["pooled"]["p97.5"] <= 1.0, floor["pooled"]
    print(f"ok  noise_floor   within-cell cos 95% "
          f"[{floor['pooled']['p2.5']:.3f}, {floor['pooled']['p97.5']:.3f}]")

    gate = A.null_gate(vec, l_star, personas, wc, nc, floor)
    assert gate["gate_passed"], gate["worst_abs_cos"]
    print(f"ok  null_gate     worst |cos(v_val,v_null)| = {gate['worst_abs_cos']:.3f}")

    geo = A.geometry(vec, boot, l_star, personas, wc, floor)
    # planted: assistant and holmes share most of their direction; marvin least
    i_a, i_h = personas.index("assistant"), personas.index("holmes")
    i_m = personas.index("marvin")
    cos_ah = geo["cos_matrix"][i_a][i_h]
    cos_am = geo["cos_matrix"][i_a][i_m]
    assert cos_ah > cos_am, (cos_ah, cos_am)
    print(f"ok  geometry      cos(assistant,holmes)={cos_ah:.3f} > "
          f"cos(assistant,marvin)={cos_am:.3f}")

    var = A.variance_decomposition(vec, boot, l_star, personas, wc)
    f = var["point"]["noise_corrected_fraction"]
    assert f["persona"] > f["context"], f
    print(f"ok  variance      persona={f['persona']:.3f} context={f['context']:.3f} "
          f"interaction={f['interaction']:.3f}")

    fac = A.factorial(vec, boot, l_star, personas, wc)
    assert set(fac["effects"]) >= {"context_consistency", "alignment_to_grand"}
    eff = fac["effects"]["alignment_to_grand"]
    assert "main_effect_spec_density" in eff and "interaction" in eff
    print(f"ok  factorial     prior-density main effect "
          f"{eff['main_effect_prior_density']['estimate']:+.3f} "
          f"CI {np.round(eff['main_effect_prior_density']['ci95'], 3).tolist()}")

    dim = A.effective_dimension(vec, boot, l_star, personas, wc)
    # The plant spans {shared} u {idio_p : 5 personas} = 6 directions, and context
    # adds none. So the participation ratio must land near 6 and clearly above the
    # one-direction null -- this is what pins the estimator to a known answer.
    assert dim["participation_ratio"] > dim["participation_ratio_null"]["p97.5"], dim
    assert 4.0 < dim["participation_ratio"] < 9.0, (
        f"PR={dim['participation_ratio']:.2f}, planted rank is 6")
    print(f"ok  dimensionality PR={dim['participation_ratio']:.2f} (planted rank 6) "
          f"vs null p97.5={dim['participation_ratio_null']['p97.5']:.2f}, "
          f"{dim['n_components_90pct']} components to 90%")

    # --- steering-side maths on a fabricated sweep --------------------------
    for label, shared in (("avg", True), ("diagonal", False)):
        st = make_sweep_state(l_star, shared_substrate=shared)
        T = S.transfer_from_sweep(st)
        off = np.array(T["T_norm"])[~np.eye(5, dtype=bool)]
        assert T["masked_cells"], "incoherence masking never fired"
        assert np.isfinite(np.nanmean(off))
        if shared:
            assert np.nanmean(off) > 0.7, np.nanmean(off)
        else:
            assert np.nanmean(off) < 0.5, np.nanmean(off)
        # the inert control must come back near zero relative to the diagonal
        assert T["null_control"], "functional null control missing"
        assert abs(T["null_control_mean_ratio"]) < 0.25, T["null_control_mean_ratio"]
        if label == "avg":
            config.dump_json(config.RESULTS / "transfer_avg.json", T)
            (config.CACHE / f"{config.cache_key('sweep', ctx='avg', nq=config.N_MMLU)}.json"
             ).write_text(json.dumps(st), encoding="utf-8")
        print(f"ok  transfer[{label}]  mean off-diagonal T_norm = {np.nanmean(off):.3f}, "
              f"{len(T['masked_cells'])} cells masked")

    rank = A.transfer_rank()
    assert rank and rank["rank1_variance_explained"] > 0.5, rank
    print(f"ok  transfer_rank  rank-1 explains "
          f"{rank['rank1_variance_explained']:.3f} -> {rank['reading']}")

    readings = A.interpret(A.tabulate_transfer())
    assert readings and "shared substrate" in readings[0], readings
    print("ok  interpret     " + readings[0])

    config.dump_json(config.RESULTS / "geometry.json", geo)
    config.dump_json(config.RESULTS / "variance.json", var)
    config.dump_json(config.RESULTS / "factorial.json", fac)
    config.dump_json(config.RESULTS / "null_gate.json", gate)
    config.dump_json(config.RESULTS / "dimensionality.json", dim)

    A.write_summary_csv({"null_gate": gate, "geometry": geo, "variance": var,
                         "factorial": fac, "transfer": A.tabulate_transfer(),
                         "dimensionality": dim, "transfer_rank": rank})
    assert (config.RESULTS / "summary.csv").exists()
    print("ok  summary.csv")

    # --- figures ------------------------------------------------------------
    import plots  # noqa: E402  (imported late: matplotlib is slow)

    plots.fig1_transfer("avg")
    plots.fig2_cosine()
    plots.fig3_alpha_curves("avg")
    plots.fig4_layer_sweep()
    plots.fig5_variance()
    plots.fig6_dimensionality()
    made = sorted(p.name for p in config.RESULTS.glob("*.png"))
    for expect in ("fig1_transfer_avg.png", "fig2_cosine.png",
                   "fig3_alpha_curves_avg.png", "fig4_layer_sweep.png",
                   "fig6_dimensionality.png"):
        assert expect in made, (expect, made)
    print(f"ok  figures       {made}")

    print(f"\nall synthetic-pipeline checks passed  (artifacts under {_TMP})")
    return 0


if __name__ == "__main__":
    code = main()
    if "--keep" not in sys.argv:
        shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(code)
