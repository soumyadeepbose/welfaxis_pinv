"""Phase D: steering sweep, normalised P(True) readout, transfer matrix.

Readout follows Kadavath et al. (2022) as used by Han et al. (2026): the model
answers an MMLU question unsteered, is asked whether its own answer is correct,
and we read P(True)/(P(True)+P(False)) from the logits at the answer position
while steering. Logit-only: no judge, no API cost.

The headline object is the transfer matrix

    T[p][q] = slope of normalised P(True) against alpha
              when persona p is steered with the vector extracted under q

Everything is written incrementally to cache/ and resumes, because Phase D is
the expensive half of the pod budget.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import config
import extract as E
import modelio as M
import personas as P

TRUE_FALSE_PROMPT = "Is your proposed answer correct? Answer only 'True' or 'False'."
LETTERS = ("A", "B", "C", "D")


# --------------------------------------------------------------------------
# MMLU
# --------------------------------------------------------------------------


def load_mmlu(n: int) -> list[dict]:
    """Balanced draw from the high-school subsets, cached to disk."""
    path = config.CACHE / f"mmlu_{n}_s{config.SEED}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    from datasets import load_dataset

    rng = np.random.default_rng(config.SEED)
    per = int(np.ceil(n / len(config.MMLU_SUBSETS)))
    rows: list[dict] = []
    for sub in config.MMLU_SUBSETS:
        ds = load_dataset("cais/mmlu", sub, split="test")
        idx = rng.permutation(len(ds))[:per]
        for i in idx:
            r = ds[int(i)]
            rows.append({"subject": sub, "question": r["question"],
                         "choices": list(r["choices"]), "answer": int(r["answer"])})
    rows = [rows[i] for i in rng.permutation(len(rows))][:n]
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def format_question(row: dict) -> str:
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, row["choices"]))
    return (f"{row['question']}\n{opts}\n\n"
            "Give the letter of the correct option and one sentence of reasoning.")


# --------------------------------------------------------------------------
# unsteered answers (one per persona; reused at every alpha)
# --------------------------------------------------------------------------


@torch.no_grad()
def generate_unsteered_answers(lm: M.LM, persona: P.Persona, rows: list[dict],
                               force: bool = False) -> list[str]:
    path = config.CACHE / f"{config.cache_key('answers', p=persona.id, n=len(rows))}.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    tok = lm.tokenizer
    answers: list[str] = []
    bs = config.STEER_BATCH
    t0 = time.time()
    for i in range(0, len(rows), bs):
        chunk = rows[i: i + bs]
        prompts = [M.build_situation_prompt(lm, persona, format_question(r)) for r in chunk]
        enc = M.encode_batch(lm, prompts, max_len=config.MAX_LEN)
        out = lm.model.generate(**enc, max_new_tokens=config.MMLU_ANSWER_TOKENS,
                                do_sample=False, pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        answers += [tok.decode(r, skip_special_tokens=True).strip() for r in new]
    path.write_text(json.dumps(answers, indent=2), encoding="utf-8")
    print(f"  [answers] {persona.id}: {len(answers)} in {time.time() - t0:.0f}s")
    return answers


# --------------------------------------------------------------------------
# P(True) readout
# --------------------------------------------------------------------------


def _first_token_ids(lm: M.LM, words: list[str]) -> list[int]:
    ids = set()
    for w in words:
        for variant in (w, " " + w):
            enc = lm.tokenizer(variant, add_special_tokens=False)["input_ids"]
            if enc:
                ids.add(int(enc[0]))
    return sorted(ids)


def truefalse_token_ids(lm: M.LM) -> tuple[list[int], list[int]]:
    return (_first_token_ids(lm, ["True", "true", "TRUE"]),
            _first_token_ids(lm, ["False", "false", "FALSE"]))


def build_ptrue_inputs(lm: M.LM, persona: P.Persona, row: dict, answer: str):
    """Token ids + assistant-turn mask for the two-turn self-evaluation."""
    q = format_question(row)
    if persona.uses_chat_template:
        messages = []
        if persona.system_prompt:
            messages.append({"role": "system", "content": persona.system_prompt})
        messages += [
            {"role": "user", "content": q},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": TRUE_FALSE_PROMPT},
        ]
        return M.encode_chat_with_mask(lm, messages, add_generation_prompt=True)

    segments = [
        (M.BARE_PREAMBLE + q + "\n", False),
        (answer, True),
        ("\n\n" + TRUE_FALSE_PROMPT + "\n", False),
        ("", True),
    ]
    ids, mask = M.encode_bare_with_mask(lm, segments)
    if mask:
        mask[-1] = True  # the read position is a model position
    return ids, mask


@torch.no_grad()
def ptrue_scores(lm: M.LM, persona: P.Persona, rows: list[dict], answers: list[str],
                 layer: int, direction: torch.Tensor | None, magnitude: float,
                 true_ids: list[int], false_ids: list[int]) -> np.ndarray:
    """Normalised P(True) per question under one steering condition."""
    out = np.zeros(len(rows), dtype=np.float32)
    bs = config.STEER_BATCH
    for i in range(0, len(rows), bs):
        chunk = rows[i: i + bs]
        built = [build_ptrue_inputs(lm, persona, r, a)
                 for r, a in zip(chunk, answers[i: i + bs])]
        ids, attn, steer_mask = M.pad_batch(lm, [b[0] for b in built], [b[1] for b in built])
        with M.steering(lm, layer, direction, magnitude, position_mask=steer_mask):
            logits = lm.model(input_ids=ids, attention_mask=attn, use_cache=False).logits
        last = logits[:, -1, :].float()
        probs = torch.softmax(last, dim=-1)
        pt = probs[:, true_ids].sum(dim=-1)
        pf = probs[:, false_ids].sum(dim=-1)
        out[i: i + len(chunk)] = (pt / (pt + pf + 1e-12)).cpu().numpy()
    return out


# --------------------------------------------------------------------------
# incoherence logging
# --------------------------------------------------------------------------

_WORD = re.compile(r"\w+")


def is_incoherent(text: str) -> tuple[bool, dict]:
    """Cheap local heuristic: repetition, degenerate unigrams, length collapse.

    No API credit is spent here; the point is only to mask cells where steering
    has destroyed generation, which would otherwise silently inflate or erase
    an effect size.
    """
    t = text.strip()
    toks = _WORD.findall(t.lower())
    stats: dict = {"n_chars": len(t), "n_tokens": len(toks)}
    if len(t) < config.MIN_CHARS or len(toks) < 4:
        stats["reason"] = "length_collapse"
        return True, stats

    grams = [tuple(toks[i: i + 3]) for i in range(len(toks) - 2)]
    rep = 1.0 - (len(set(grams)) / max(len(grams), 1))
    top = Counter(toks).most_common(1)[0][1] / len(toks)
    stats.update({"rep3": round(rep, 3), "top_unigram": round(top, 3)})
    if rep > config.REPETITION_MAX:
        stats["reason"] = "repetition"
        return True, stats
    if top > config.DEGENERATE_UNIGRAM_MAX:
        stats["reason"] = "degenerate_unigram"
        return True, stats
    stats["reason"] = ""
    return False, stats


@torch.no_grad()
def incoherence_rate(lm: M.LM, persona: P.Persona, rows: list[dict], layer: int,
                     direction: torch.Tensor | None, magnitude: float,
                     n: int | None = None) -> dict:
    n = n or config.N_COHERENCE
    rows = rows[:n]
    if not rows:
        return {"rate": 0.0, "n": 0}
    tok = lm.tokenizer
    flags, samples = [], []
    bs = config.STEER_BATCH
    for i in range(0, len(rows), bs):
        chunk = rows[i: i + bs]
        prompts = [M.build_situation_prompt(lm, persona, format_question(r)) for r in chunk]
        enc = M.encode_batch(lm, prompts, max_len=config.MAX_LEN)
        # the prompt contains no assistant content, so nothing in it is steered;
        # every generated token is (handled inside the hook). This keeps the
        # coherence pass on the same rule as the P(True) pass.
        mask = torch.zeros_like(enc["attention_mask"], dtype=torch.bool)
        with M.steering(lm, layer, direction, magnitude, position_mask=mask):
            out = lm.model.generate(**enc, max_new_tokens=config.COHERENCE_MAX_TOKENS,
                                    do_sample=False, pad_token_id=tok.pad_token_id)
        new = out[:, enc["input_ids"].shape[1]:]
        for r in new:
            txt = tok.decode(r, skip_special_tokens=True)
            bad, _ = is_incoherent(txt)
            flags.append(bad)
            if len(samples) < 2:
                samples.append(txt[:200])
    return {"rate": float(np.mean(flags)), "n": len(flags), "samples": samples}


# --------------------------------------------------------------------------
# directions
# --------------------------------------------------------------------------


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def source_directions(vec: dict, layer: int, context: str | None) -> dict[str, np.ndarray]:
    """Unit direction per source persona, either per-context or context-averaged.

    Normalisation is not cosmetic: extracted norms differ systematically across
    personas, so steering at a raw alpha would confound direction with magnitude.
    """
    personas = [str(x) for x in vec["personas"]]
    contexts = [str(x) for x in vec["contexts"]]
    out = {}
    for i, pid in enumerate(personas):
        if context is None:
            idx = [contexts.index(c) for c in config.CONTEXTS]
            v = vec["v"][i, idx, layer, :].mean(axis=0)
        else:
            v = vec["v"][i, contexts.index(context), layer, :]
        out[pid] = unit(v.astype(np.float32))
    return out


def bootstrap_directions(boot: dict, context: str | None,
                         b_steer: int) -> dict[str, np.ndarray]:
    """[b_steer, d] unit directions per source persona, drawn from the bootstrap.

    The bootstrap cache is already built at L*, so no layer argument is needed.
    """
    personas = [str(x) for x in boot["personas"]]
    contexts = [str(x) for x in boot["contexts"]]
    rng = np.random.default_rng(config.SEED + 7)
    out = {}
    for i, pid in enumerate(personas):
        if context is None:
            idx = [contexts.index(c) for c in config.CONTEXTS]
            draws = boot["boot"][i, idx].mean(axis=0)          # [B, d]
        else:
            draws = boot["boot"][i, contexts.index(context)]   # [B, d]
        sel = rng.permutation(draws.shape[0])[:b_steer]
        out[pid] = np.stack([unit(draws[s].astype(np.float32)) for s in sel])
    return out


# --------------------------------------------------------------------------
# slopes
# --------------------------------------------------------------------------


def ols_slope(alphas: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Slope of y on alpha with its standard error."""
    if len(alphas) < 3:
        return float("nan"), float("nan")
    A = np.vstack([np.ones_like(alphas), alphas]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(alphas) - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    return float(coef[1]), float(np.sqrt(max(cov[1, 1], 0.0)))


# --------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------


def sweep_path(context: str | None, n_q: int) -> Path:
    tag = context or "avg"
    return config.CACHE / f"{config.cache_key('sweep', ctx=tag, nq=n_q)}.json"


def run_sweep(lm: M.LM, layer: int, vec: dict, boot: dict | None, rows: list[dict],
              context: str | None, n_q: int, b_steer: int, force: bool = False) -> dict:
    """Fill one persona x persona x alpha block; resumes from partial cache."""
    path = sweep_path(context, n_q)
    state = json.loads(path.read_text(encoding="utf-8")) if (path.exists() and not force) else {
        "context": context, "n_q": n_q, "layer": layer, "alphas": list(config.ALPHAS),
        "cells": {}, "incoherence": {}, "baseline": {},
    }
    cells, inco, base = state["cells"], state["incoherence"], state["baseline"]

    dirs = source_directions(vec, layer, context)
    bdirs = bootstrap_directions(boot, context, b_steer) if boot is not None else {}
    true_ids, false_ids = truefalse_token_ids(lm)
    personas = [str(x) for x in vec["personas"]]
    contexts = [str(x) for x in vec["contexts"]]
    rows = rows[:n_q]

    answers = {pid: generate_unsteered_answers(lm, P.get(pid), rows)[:n_q] for pid in personas}

    for pi, pid in enumerate(personas):
        persona = P.get(pid)
        # median residual norm at L* for the steered persona sets the alpha unit
        med = float(np.mean(vec["norms"][pi, [contexts.index(c) for c in config.CONTEXTS], layer]))

        if "0.0" not in base.get(pid, {}):
            s = ptrue_scores(lm, persona, rows, answers[pid], layer, None, 0.0,
                             true_ids, false_ids)
            base.setdefault(pid, {})["0.0"] = float(np.mean(s))
            r = incoherence_rate(lm, persona, rows, layer, None, 0.0)
            inco.setdefault(f"{pid}|_baseline", {})["0.0"] = r
            _save(path, state)

        for qid in personas:
            key = f"{pid}|{qid}"
            cells.setdefault(key, {})
            inco.setdefault(key, {})
            d = torch.from_numpy(dirs[qid])
            for alpha in config.ALPHAS:
                a = f"{float(alpha)}"
                if a in cells[key]:
                    continue
                if float(alpha) == 0.0:
                    cells[key][a] = base[pid]["0.0"]
                    inco[key][a] = inco[f"{pid}|_baseline"]["0.0"]
                    continue
                mag = M.alpha_to_magnitude(alpha, med)
                t0 = time.time()
                s = ptrue_scores(lm, persona, rows, answers[pid], layer, d, mag,
                                 true_ids, false_ids)
                cells[key][a] = float(np.mean(s))
                inco[key][a] = incoherence_rate(lm, persona, rows, layer, d, mag)
                print(f"  [sweep {context or 'avg'}] {key} a={alpha:+.1f} "
                      f"P(True)={cells[key][a]:.3f} inco={inco[key][a]['rate']:.2f} "
                      f"({time.time() - t0:.0f}s)")
                _save(path, state)

            # bootstrap replicates on a reduced question subset -> extraction
            # noise propagated into the transfer CI
            bkey = f"boot::{key}"
            if bdirs and bkey not in cells:
                nb = min(config.N_MMLU_BOOT, n_q)
                slopes = []
                for rep in bdirs[qid]:
                    ys, xs = [], []
                    for alpha in config.ALPHAS:
                        if float(alpha) == 0.0:
                            ys.append(base[pid]["0.0"])
                        else:
                            mag = M.alpha_to_magnitude(alpha, med)
                            ys.append(float(np.mean(ptrue_scores(
                                lm, persona, rows[:nb], answers[pid][:nb], layer,
                                torch.from_numpy(rep), mag, true_ids, false_ids))))
                        xs.append(float(alpha))
                    slopes.append(ols_slope(np.array(xs), np.array(ys))[0])
                cells[bkey] = {"slopes": slopes, "n_q": nb}
                _save(path, state)

    state["cells"], state["incoherence"], state["baseline"] = cells, inco, base
    _save(path, state)
    return state


def _save(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def transfer_from_sweep(state: dict) -> dict:
    """Slopes, masking, normalisation. T_norm[p][q] = T[p][q] / T[p][p]."""
    present = {k.split("|")[0] for k in state["cells"] if "|" in k
               and not k.startswith("boot::")}
    # keep the designed grid order (thin/thin -> thick/thick -> control), not
    # alphabetical, so every figure reads the same way as the persona table
    personas = [p for p in P.PERSONA_IDS if p in present]
    T = np.full((len(personas), len(personas)), np.nan)
    SE = np.full_like(T, np.nan)
    BOOT_SD = np.full_like(T, np.nan)
    masked: list[dict] = []

    for i, p in enumerate(personas):
        for j, q in enumerate(personas):
            key = f"{p}|{q}"
            cell = state["cells"].get(key, {})
            inco = state["incoherence"].get(key, {})
            xs, ys = [], []
            for a, y in sorted(cell.items(), key=lambda kv: float(kv[0])):
                rate = inco.get(a, {}).get("rate", 0.0)
                if rate > config.INCOHERENCE_MASK_RATE:
                    masked.append({"steered": p, "source": q, "alpha": float(a),
                                   "incoherence": rate})
                    continue
                xs.append(float(a))
                ys.append(float(y))
            if len(xs) >= 3:
                T[i, j], SE[i, j] = ols_slope(np.array(xs), np.array(ys))
            bs = state["cells"].get(f"boot::{key}", {}).get("slopes")
            if bs:
                BOOT_SD[i, j] = float(np.std(bs, ddof=1)) if len(bs) > 1 else 0.0

    diag = np.diag(T).copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        T_norm = T / diag[:, None]
    total_sd = np.sqrt(np.nan_to_num(SE, nan=0.0) ** 2 + np.nan_to_num(BOOT_SD, nan=0.0) ** 2)
    return {
        "personas": personas,
        "T": T.tolist(), "T_norm": T_norm.tolist(),
        "se_ols": SE.tolist(), "sd_bootstrap": BOOT_SD.tolist(),
        "ci95_halfwidth": (1.96 * total_sd).tolist(),
        "diagonal": diag.tolist(),
        "masked_cells": masked,
        "incoherence_by_alpha": _inco_by_alpha(state),
        "context": state.get("context"), "layer": state.get("layer"),
        "n_q": state.get("n_q"),
    }


def _inco_by_alpha(state: dict) -> dict:
    agg: dict[str, list[float]] = {}
    for key, per_alpha in state["incoherence"].items():
        if key.endswith("|_baseline"):
            continue
        for a, r in per_alpha.items():
            agg.setdefault(a, []).append(r.get("rate", 0.0))
    return {a: {"mean": float(np.mean(v)), "max": float(np.max(v)), "n_cells": len(v)}
            for a, v in sorted(agg.items(), key=lambda kv: float(kv[0]))}


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase D: steering sweep")
    ap.add_argument("--model", default=config.MODEL_NAME)
    ap.add_argument("--n-pairs", type=int, default=config.N_PAIRS)
    ap.add_argument("--n-mmlu", type=int, default=config.N_MMLU)
    ap.add_argument("--n-mmlu-transfer", type=int, default=config.N_MMLU_TRANSFER)
    ap.add_argument("--b-steer", type=int, default=config.B_STEER)
    ap.add_argument("--layer", type=int, default=None, help="override L*")
    ap.add_argument("--per-context", action="store_true", default=config.TRANSFER_PER_CONTEXT)
    ap.add_argument("--no-per-context", dest="per_context", action="store_false")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    config.set_seed()
    sweep = json.loads((config.RESULTS / "layer_sweep.json").read_text(encoding="utf-8"))
    layer = args.layer or sweep["l_star"]
    vec = E.build_vectors(args.n_pairs)
    try:
        boot = E.build_bootstrap(args.n_pairs, layer)
    except Exception as exc:  # bootstrap is optional for the point estimate
        print(f"  [warn] no bootstrap cache ({exc}); transfer CIs will omit extraction noise")
        boot = None

    lm = M.load_model(args.model)
    print(f"[steer] {lm.name} L*={layer} alphas={config.ALPHAS} "
          f"unit={config.ALPHA_UNIT_FRAC} x median norm")
    rows = load_mmlu(max(args.n_mmlu, args.n_mmlu_transfer))

    # headline: context-averaged vectors, full question set
    state = run_sweep(lm, layer, vec, boot, rows, None, args.n_mmlu, args.b_steer,
                      force=args.force)
    config.dump_json(config.RESULTS / "transfer_avg.json", transfer_from_sweep(state))

    if args.per_context:
        for ctx in config.CONTEXTS:
            st = run_sweep(lm, layer, vec, boot, rows, ctx, args.n_mmlu_transfer,
                           args.b_steer, force=args.force)
            config.dump_json(config.RESULTS / f"transfer_{ctx}.json", transfer_from_sweep(st))

    print("[steer] done -- stop the pod.")


if __name__ == "__main__":
    main()
