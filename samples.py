"""Extract steered generations from cached sweeps, for the appendix.

Every sweep already stores two generations per (persona, source, alpha) cell --
they were captured for the incoherence mask, not for display, which is exactly
why they are usable as evidence: nothing about their selection was influenced by
how they read. The rule is fixed by the code that wrote them (`incoherence_rate`
keeps the first two of the coherence set, in dataset order) and is restated in
the appendix so the reader can see there was no curation step.

This script does three things:

  1. dumps every cached generation with its cell metadata, so the full set is
     released alongside the paper rather than just the printed excerpt;
  2. measures whether steering changes the *content* of a generation -- the
     selected MMLU option and the text itself -- across alpha;
  3. writes the LaTeX appendix table.

The measurement in (2) is the point. The readout is a probability at one token
position, so an effect there need not surface in sampled text at all. Whether it
does is a question with an answer, and the answer belongs in the paper either
way.

    python samples.py                    # all cached sweeps for the current model
    python samples.py --cell assistant --source bare
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import defaultdict

import numpy as np

import config

# leading option marker: "C.", "**B**", "C\n", "Answer: D"
_OPTION = re.compile(r"^\W*(?:answer\s*[:\-]?\s*)?\**\s*([A-D])\b", re.I)


def selected_option(text: str) -> str | None:
    m = _OPTION.match(text.strip())
    return m.group(1).upper() if m else None


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.strip(), b.strip()).ratio()


def collect(state: dict) -> list[dict]:
    """Flatten the cached incoherence log into one row per stored generation."""
    rows = []
    for key, per_alpha in state.get("incoherence", {}).items():
        persona, _, source = key.partition("|")
        for alpha, payload in per_alpha.items():
            for idx, text in enumerate(payload.get("samples", [])):
                rows.append({
                    "persona": persona, "source": source, "alpha": float(alpha),
                    "slot": idx, "text": text,
                    "option": selected_option(text),
                    "incoherence_rate": payload.get("rate"),
                })
    return rows


def invariance(rows: list[dict]) -> dict:
    """How much does a generation change between alpha=0 and the extremes?

    Grouped by (persona, source, slot) so the comparison is always the same
    question under the same persona, differing only in the steering coefficient.
    """
    by_slot: dict[tuple, dict[float, dict]] = defaultdict(dict)
    for r in rows:
        by_slot[(r["persona"], r["source"], r["slot"])][r["alpha"]] = r

    sims, option_changed, compared = [], 0, 0
    for series in by_slot.values():
        if 0.0 not in series:
            continue
        ref = series[0.0]
        for alpha, r in series.items():
            if alpha == 0.0:
                continue
            compared += 1
            sims.append(_similarity(ref["text"], r["text"]))
            if ref["option"] and r["option"] and ref["option"] != r["option"]:
                option_changed += 1
    return {
        "n_generations": len(rows),
        "n_comparisons": compared,
        "median_similarity_to_alpha0": round(float(np.median(sims)), 4) if sims else None,
        "mean_similarity_to_alpha0": round(float(np.mean(sims)), 4) if sims else None,
        "option_changes": option_changed,
        "option_change_rate": round(option_changed / compared, 4) if compared else None,
        "max_incoherence_rate": max((r["incoherence_rate"] or 0.0) for r in rows) if rows else None,
    }


# Model output is UTF-8 and the paper is compiled with the default OT1 font
# encoding, where a stray em dash pulls in a bitmap font and makes microtype's
# font expansion a fatal error. Map the punctuation we actually see to its LaTeX
# form and drop anything else rather than letting it reach the compiler.
_UNICODE_TEX = {
    "—": "---", "–": "--", "’": "'", "‘": "`",
    "“": "``", "”": "''", "…": r"\ldots{}", " ": " ",
}


def _escape(text: str) -> str:
    out = text.strip().replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        out = out.replace(ch, "\\" + ch)
    for uni, tex in _UNICODE_TEX.items():
        out = out.replace(uni, tex)
    out = "".join(c for c in out if ord(c) < 127)
    return re.sub(r"\s+", " ", out)


def latex_table(rows: list[dict], persona: str, source: str, slot: int,
                width: int = 150) -> str:
    sel = sorted((r for r in rows if r["persona"] == persona
                  and r["source"] == source and r["slot"] == slot),
                 key=lambda r: r["alpha"])
    if not sel:
        return ""
    lines = [
        r"\begin{tabular}{@{}r p{0.82\linewidth}@{}}",
        r"\toprule",
        r"$\alpha$ & generation (first %d characters, greedy decoding) \\" % width,
        r"\midrule",
    ]
    for r in sel:
        lines.append(rf"${r['alpha']:+.0f}$ & \small {_escape(r['text'][:width])}\dots \\[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="assistant", help="persona to print")
    ap.add_argument("--source", default="bare", help="steering source to print")
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--width", type=int, default=150)
    args = ap.parse_args()

    slug = config.model_slug()
    paths = sorted(config.CACHE.glob(f"sweep_{slug}_*.json"))
    if not paths:
        print(f"no cached sweeps for {slug}")
        return

    rows: list[dict] = []
    for p in paths:
        state = json.loads(p.read_text(encoding="utf-8"))
        for r in collect(state):
            r["sweep"] = p.stem
            rows.append(r)

    stats = invariance(rows)
    payload = {"model": slug, "selection_rule":
               "first two generations of the coherence set, in dataset order; "
               "written by incoherence_rate() before any result was known",
               "invariance": stats, "generations": rows}
    config.dump_json(config.RESULTS / "generation_samples.json", payload)

    print(f"model {slug}  ({len(paths)} sweep(s), {len(rows)} stored generations)")
    for k, v in stats.items():
        print(f"  {k:32s} {v}")

    tbl = latex_table(rows, args.cell, args.source, args.slot, args.width)
    if tbl:
        out = config.RESULTS / f"table_samples_{args.cell}_{args.source}.tex"
        out.write_text(tbl + "\n", encoding="utf-8")
        print(f"  [latex] {out}")


if __name__ == "__main__":
    main()
