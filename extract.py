"""Phase B: activation capture, difference-in-means, layer selection, bootstrap.

Every stage writes to disk and resumes. GPU time is the scarce resource, so
activations are never recomputed once cached; the layer sweep, the bootstrap
and the null-contrast gate all read from the same cached tensors.

Artifacts
    cache/acts_<key>_<persona>_<context>.npz   [n_pairs, n_layers+1, d] fp16
    cache/vectors_<key>.npz                    v_val[p, c, layer, d] fp32
    cache/bootstrap_<key>.npz                  boot[p, c, B, d] at L* only
    results/layer_sweep.json                   probe AUC per layer
    results/extract_summary.json               L*, norms, tokenisation report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import config
import contrasts as C
import modelio as M
import personas as P

ALL_CONTEXTS = config.CONTEXTS + (config.NULL_CONTEXT,)


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------


def acts_path(persona: str, context: str, n_pairs: int) -> Path:
    return config.CACHE / f"{config.cache_key('acts', p=persona, c=context, n=n_pairs)}.npz"


def vectors_path(n_pairs: int) -> Path:
    return config.CACHE / f"{config.cache_key('vectors', n=n_pairs)}.npz"


def bootstrap_path(n_pairs: int, layer: int) -> Path:
    return config.CACHE / f"{config.cache_key('bootstrap', n=n_pairs, L=layer)}.npz"


# --------------------------------------------------------------------------
# stage 1: activations
# --------------------------------------------------------------------------


def capture_cell(lm: M.LM, persona: P.Persona, pairs: list[C.ContrastPair],
                 context: str, n_pairs: int, force: bool = False) -> Path:
    path = acts_path(persona.id, context, n_pairs)
    if path.exists() and not force:
        return path
    pos_prompts = [M.build_situation_prompt(lm, persona, p.positive) for p in pairs]
    neg_prompts = [M.build_situation_prompt(lm, persona, p.negative) for p in pairs]
    t0 = time.time()
    pos = M.last_token_hidden(lm, pos_prompts)
    neg = M.last_token_hidden(lm, neg_prompts)
    np.savez_compressed(path, pos=pos, neg=neg,
                        pair_ids=np.array([p.id for p in pairs]),
                        elapsed=np.array([time.time() - t0]))
    print(f"  [acts] {persona.id}/{context}: {pos.shape} in {time.time() - t0:.1f}s -> {path.name}")
    return path


def load_acts(persona: str, context: str, n_pairs: int):
    d = np.load(acts_path(persona, context, n_pairs))
    return d["pos"].astype(np.float32), d["neg"].astype(np.float32)


def run_capture(lm: M.LM, n_pairs: int, force: bool = False) -> None:
    sets = C.build_all(tokenizer=lm.tokenizer, n_pairs=n_pairs,
                       model_slug=config.model_slug(lm.name), force=force)
    report = M.tokenisation_report(lm, sets[config.CONTEXTS[0]][0].positive)
    config.dump_json(config.RESULTS / "tokenisation_report.json", report)
    if not report["_bare_differs_from_chat"]:
        raise RuntimeError(
            "bare persona tokenises identically to the chat path -- the thin/thin "
            "cell is not distinct. Fix build_situation_prompt before spending GPU time."
        )
    if report["_special_tokens_in_bare"]:
        print(f"  [warn] special tokens present in bare path: {report['_special_tokens_in_bare']}")
    if not report["_chat_read_positions_match"]:
        raise RuntimeError(
            "chat personas do not share a read position; the between-persona "
            "cosines would be comparing different constructs."
        )
    if report["_hybrid_thinking_template"] and not report["_thinking_disabled"]:
        print("  [warn] hybrid-thinking template with thinking ON: the model opens\n"
              "         <think> at the read position, so the P(True) readout in Phase D\n"
              "         will be a ratio of two distribution tails. Set VOID_ENABLE_THINKING=false.")

    for persona in P.PERSONAS:
        for ctx in ALL_CONTEXTS:
            capture_cell(lm, persona, sets[ctx], ctx, n_pairs, force=force)


# --------------------------------------------------------------------------
# stage 2: difference-in-means vectors, all layers
# --------------------------------------------------------------------------


def diff_in_means(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """[n_layers+1, d] -- the welfare direction at every layer."""
    return pos.mean(axis=0) - neg.mean(axis=0)


def build_vectors(n_pairs: int, force: bool = False) -> dict:
    path = vectors_path(n_pairs)
    if path.exists() and not force:
        return dict(np.load(path))

    personas = list(P.PERSONA_IDS)
    v = None
    norms = None
    for pi, pid in enumerate(personas):
        for ci, ctx in enumerate(ALL_CONTEXTS):
            pos, neg = load_acts(pid, ctx, n_pairs)
            d = diff_in_means(pos, neg)
            if v is None:
                v = np.zeros((len(personas), len(ALL_CONTEXTS)) + d.shape, dtype=np.float32)
                norms = np.zeros((len(personas), len(ALL_CONTEXTS), d.shape[0]), dtype=np.float32)
            v[pi, ci] = d
            # median residual norm per layer, pooled over both members
            both = np.concatenate([pos, neg], axis=0)
            norms[pi, ci] = np.median(np.linalg.norm(both, axis=-1), axis=0)
    np.savez_compressed(path, v=v, norms=norms,
                        personas=np.array(personas), contexts=np.array(ALL_CONTEXTS))
    print(f"  [vectors] {v.shape} -> {path.name}")
    return {"v": v, "norms": norms,
            "personas": np.array(personas), "contexts": np.array(ALL_CONTEXTS)}


# --------------------------------------------------------------------------
# stage 3: layer selection (assistant persona only, then frozen)
# --------------------------------------------------------------------------


def probe_auc(pos: np.ndarray, neg: np.ndarray, layer: int, folds: int = config.PROBE_FOLDS,
              seed: int | None = None) -> float:
    """Cross-validated logistic-probe AUC, split by *pair* to avoid leakage.

    Both members of a pair share a prefix; splitting by item would put a
    near-duplicate of a test prompt in the training set.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    n = pos.shape[0]
    X = np.concatenate([pos[:, layer, :], neg[:, layer, :]], axis=0)
    y = np.concatenate([np.ones(n), np.zeros(n)])
    groups = np.concatenate([np.arange(n), np.arange(n)])

    k = min(folds, n)
    aucs = []
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0,
                                               random_state=seed or config.SEED))
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        if len(np.unique(y[te])) < 2:
            continue
        aucs.append(roc_auc_score(y[te], s))
    return float(np.mean(aucs)) if aucs else float("nan")


def layer_sweep(n_pairs: int, n_layers: int, persona: str | None = None,
                force: bool = False) -> dict:
    out_path = config.RESULTS / "layer_sweep.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text(encoding="utf-8"))

    persona = persona or config.LAYER_SELECT_PERSONA
    layers = M.candidate_layers(n_layers)
    per_ctx: dict[str, list[float]] = {}
    for ctx in config.CONTEXTS:            # welfare contexts only; null excluded
        pos, neg = load_acts(persona, ctx, n_pairs)
        per_ctx[ctx] = [probe_auc(pos, neg, L) for L in layers]
        print(f"  [sweep] {ctx}: best AUC {max(per_ctx[ctx]):.3f} "
              f"@ layer {layers[int(np.argmax(per_ctx[ctx]))]}")
    mean_auc = np.mean([per_ctx[c] for c in config.CONTEXTS], axis=0)
    l_star = int(layers[int(np.argmax(mean_auc))])

    # null-contrast separability at the same layers: a probe that separates the
    # neutral topics just as well would mean the layer is generically linearly
    # rich rather than welfare-selective.
    npos, nneg = load_acts(persona, config.NULL_CONTEXT, n_pairs)
    null_auc = [probe_auc(npos, nneg, L) for L in layers]

    payload = {
        "persona": persona, "layers": layers, "per_context_auc": per_ctx,
        "mean_auc": mean_auc.tolist(), "null_auc": null_auc,
        "l_star": l_star, "l_star_auc": float(np.max(mean_auc)),
        "n_layers": n_layers,
    }
    config.dump_json(out_path, payload)
    print(f"  [sweep] L* = {l_star} (mean AUC {payload['l_star_auc']:.3f}), frozen for all personas")
    return payload


# --------------------------------------------------------------------------
# stage 4: bootstrap noise floor
# --------------------------------------------------------------------------


def bootstrap_cell(pos: np.ndarray, neg: np.ndarray, layer: int, B: int,
                   rng: np.random.Generator) -> np.ndarray:
    """[B, d] difference-in-means vectors from pair-level resampling."""
    n = pos.shape[0]
    p, q = pos[:, layer, :], neg[:, layer, :]
    idx = rng.integers(0, n, size=(B, n))
    return (p[idx].mean(axis=1) - q[idx].mean(axis=1)).astype(np.float32)


def build_bootstrap(n_pairs: int, layer: int, B: int | None = None,
                    force: bool = False) -> dict:
    B = B or config.B_BOOTSTRAP
    path = bootstrap_path(n_pairs, layer)
    if path.exists() and not force:
        return dict(np.load(path))
    rng = np.random.default_rng(config.SEED)
    personas = list(P.PERSONA_IDS)
    boot = None
    for pi, pid in enumerate(personas):
        for ci, ctx in enumerate(ALL_CONTEXTS):
            pos, neg = load_acts(pid, ctx, n_pairs)
            b = bootstrap_cell(pos, neg, layer, B, rng)
            if boot is None:
                boot = np.zeros((len(personas), len(ALL_CONTEXTS), B, b.shape[-1]),
                                dtype=np.float32)
            boot[pi, ci] = b
    np.savez_compressed(path, boot=boot, layer=np.array([layer]),
                        personas=np.array(personas), contexts=np.array(ALL_CONTEXTS))
    print(f"  [bootstrap] {boot.shape} -> {path.name}")
    return {"boot": boot, "layer": np.array([layer]),
            "personas": np.array(personas), "contexts": np.array(ALL_CONTEXTS)}


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase B: extraction")
    ap.add_argument("--model", default=config.MODEL_NAME)
    ap.add_argument("--n-pairs", type=int, default=config.N_PAIRS)
    ap.add_argument("--bootstrap", type=int, default=config.B_BOOTSTRAP)
    ap.add_argument("--stage", default="all",
                    choices=["all", "acts", "vectors", "layers", "bootstrap"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-gpu", action="store_true",
                    help="skip capture; run the numpy stages from cached activations")
    args = ap.parse_args()

    config.set_seed()
    n_layers = None
    sweep: dict = {}
    l_star = -1

    if args.stage in ("all", "acts") and not args.no_gpu:
        lm = M.load_model(args.model)
        n_layers = lm.n_layers
        print(f"[extract] {lm.name}: {lm.n_layers} layers, d={lm.d_model}, {lm.device}")
        run_capture(lm, args.n_pairs, force=args.force)
        del lm
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    if n_layers is None:
        # infer from a cached cell so the numpy stages need no GPU or download
        pos, _ = load_acts(P.PERSONA_IDS[0], config.CONTEXTS[0], args.n_pairs)
        n_layers = pos.shape[1] - 1

    if args.stage in ("all", "vectors"):
        build_vectors(args.n_pairs, force=args.force)
    if args.stage in ("all", "layers", "bootstrap"):
        sweep = layer_sweep(args.n_pairs, n_layers, force=args.force)
        l_star = sweep["l_star"]
    if args.stage in ("all", "bootstrap"):
        build_bootstrap(args.n_pairs, l_star, B=args.bootstrap, force=args.force)

    if args.stage == "all":
        vec = build_vectors(args.n_pairs)
        norms = vec["norms"]
        summary = {
            "model": args.model, "n_pairs": args.n_pairs, "l_star": l_star,
            "l_star_auc": sweep["l_star_auc"], "n_layers": n_layers,
            "b_bootstrap": args.bootstrap,
            "median_norm_at_l_star": {
                pid: float(np.mean(norms[i, :, l_star]))
                for i, pid in enumerate(P.PERSONA_IDS)
            },
            "vector_norm_at_l_star": {
                pid: float(np.mean([np.linalg.norm(vec["v"][i, c, l_star])
                                    for c in range(len(config.CONTEXTS))]))
                for i, pid in enumerate(P.PERSONA_IDS)
            },
        }
        config.dump_json(config.RESULTS / "extract_summary.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
