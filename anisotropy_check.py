"""Which axis is the real anisotropy axis? (no GPU, reads cached activations)

Two candidate "common directions" exist, and they are not the same thing:

  m_act   mean ACTIVATION at L* -- the centre of the residual-stream cloud.
          This is anisotropy in the usual sense: every hidden state has a large
          component along it, regardless of content.

  m_diff  mean of the extracted DIFFERENCE vectors. In this design 20 of the 25
          cells are welfare contrasts and only 5 are nulls, so m_diff is mostly
          the welfare direction. Projecting it out removes the signal, which is
          what the first deflated sweep did.

This script quantifies the difference and predicts what a corrected deflation
would do, so the GPU is only spent if the correction is worth it.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

import config
import extract as E
import personas as P

ALL_CONTEXTS = config.CONTEXTS + (config.NULL_CONTEXT,)


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def _cos(a, b):
    return float(np.dot(_unit(a), _unit(b)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=config.N_PAIRS)
    ap.add_argument("--layer", type=int, default=None)
    args = ap.parse_args()

    sweep = json.loads((config.RESULTS / "layer_sweep.json").read_text(encoding="utf-8"))
    L = args.layer or sweep["l_star"]
    vec = E.build_vectors(args.n_pairs)
    personas = [str(x) for x in vec["personas"]]
    contexts = [str(x) for x in vec["contexts"]]
    wc = [contexts.index(c) for c in config.CONTEXTS]
    nc = contexts.index(config.NULL_CONTEXT)

    # --- m_act: mean activation at L*, pooled over every cell and both members
    acc, n_seen = None, 0
    for pid in personas:
        for ctx in ALL_CONTEXTS:
            pos, neg = E.load_acts(pid, ctx, args.n_pairs)
            block = np.concatenate([pos[:, L, :], neg[:, L, :]], axis=0)
            acc = block.sum(axis=0) if acc is None else acc + block.sum(axis=0)
            n_seen += block.shape[0]
            del pos, neg
    m_act = _unit(acc / n_seen)

    # --- m_diff: what the first deflated sweep actually removed
    m_diff = _unit(vec["v"][:, :, L, :].reshape(-1, vec["v"].shape[-1]).mean(axis=0))

    val = {pid: vec["v"][i, wc, L, :].mean(axis=0) for i, pid in enumerate(personas)}
    nul = {pid: vec["v"][i, nc, L, :] for i, pid in enumerate(personas)}

    def _resid_frac(v, m):
        v = _unit(v)
        return float(np.linalg.norm(v - np.dot(v, m) * m))

    out = {
        "layer": L,
        "cos_m_act_m_diff": _cos(m_act, m_diff),
        "per_persona": {
            pid: {
                "cos_val_m_act": _cos(val[pid], m_act),
                "cos_val_m_diff": _cos(val[pid], m_diff),
                "cos_null_m_act": _cos(nul[pid], m_act),
                "cos_null_m_diff": _cos(nul[pid], m_diff),
                "val_survives_m_act_deflation": _resid_frac(val[pid], m_act),
                "val_survives_m_diff_deflation": _resid_frac(val[pid], m_diff),
            }
            for pid in personas
        },
    }
    # after removing m_act, do welfare directions still agree with each other
    # more than with the nulls? That is the property the control needs.
    def _d(v):
        v = _unit(v)
        return _unit(v - np.dot(v, m_act) * m_act)

    vv, vn = [], []
    for i, a in enumerate(personas):
        for j, b in enumerate(personas):
            if i < j:
                vv.append(float(np.dot(_d(val[a]), _d(val[b]))))
            vn.append(float(np.dot(_d(val[a]), _d(nul[b]))))
    out["after_m_act_deflation"] = {
        "cos_val_val_across_personas": float(np.mean(vv)),
        "cos_val_null": float(np.mean(vn)),
        "separation": float(np.mean(vv) - np.mean(vn)),
    }
    out["verdict"] = (
        "m_act deflation preserves the welfare direction; a corrected deflated "
        "sweep is worth running"
        if np.mean([r["val_survives_m_act_deflation"] for r in out["per_persona"].values()]) > 0.7
        else "m_act and the welfare direction are strongly aligned; deflation cannot "
             "separate them and the raw sweep plus null control is the honest result")

    config.dump_json(config.RESULTS / "anisotropy_check.json", out)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
