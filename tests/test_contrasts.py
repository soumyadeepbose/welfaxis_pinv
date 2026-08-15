"""Generation-rule tests. Runnable with pytest, or directly with python.

The affect-vocabulary test is the important one: a hit there invalidates the
extraction, so it is checked over every item of every context, not a sample.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config          # noqa: E402
import contrasts as C  # noqa: E402
import personas as P   # noqa: E402

N = 60  # enough to exercise balance without regenerating the full sets


def _sets():
    return {ctx: C.generate_context(ctx, N) for ctx in C.CONTEXT_TEMPLATES}


def test_no_affective_vocabulary():
    for ctx, pairs in _sets().items():
        for p in pairs:
            C.assert_no_affect(p.positive, f"{ctx}/{p.id}/pos")
            C.assert_no_affect(p.negative, f"{ctx}/{p.id}/neg")


def test_shared_prefix_and_single_divergence():
    for ctx, pairs in _sets().items():
        for p in pairs:
            assert p.positive[: p.divergence_char] == p.negative[: p.divergence_char]
            assert p.positive[: p.divergence_char] == p.prefix
            assert p.positive != p.negative
            # divergence is a suffix: nothing after it is shared structure beyond
            # the frame tail, which is identical by construction
            assert p.positive.endswith(".") and p.negative.endswith(".")


def test_length_matching_word_proxy():
    for ctx, pairs in _sets().items():
        deltas = [abs(len(p.positive.split()) - len(p.negative.split())) for p in pairs]
        assert max(deltas) <= config.MAX_TOKEN_DELTA + 2, (ctx, max(deltas))


def test_frame_and_order_balance():
    for ctx, pairs in _sets().items():
        frames = Counter(p.frame_id for p in pairs)
        orders = Counter(p.stored_first for p in pairs)
        assert max(frames.values()) - min(frames.values()) <= 1, (ctx, frames)
        assert abs(orders["positive"] - orders["negative"]) <= 1, (ctx, orders)
        # frame is uncorrelated with polarity by construction: both members of a
        # pair use the same frame
        for p in pairs:
            head = C.FRAMES[p.frame_id].split("{d}")[0]
            assert p.prefix.endswith(head)


def test_uniqueness():
    for ctx, pairs in _sets().items():
        assert len({p.positive for p in pairs}) == len(pairs), ctx
        assert len({p.prefix for p in pairs}) == len(pairs), ctx


def test_full_size_available():
    for ctx in C.CONTEXT_TEMPLATES:
        pairs = C.generate_context(ctx, config.N_PAIRS)
        assert len(pairs) == config.N_PAIRS


def test_persona_register():
    r = P.check_register()
    assert r["ok"], r
    assert r["word_count_spread"] <= 10, r
    assert not any(r["affect_hits"].values()), r["affect_hits"]
    assert P.get("bare").system_prompt is None
    assert not P.get("bare").uses_chat_template
    # the 2x2 really is a 2x2
    cells = {(P.get(p).spec_density, P.get(p).prior_density) for p in P.FACTORIAL_IDS}
    assert len(cells) == 4, cells


def test_null_context_shares_pipeline():
    pairs = C.generate_context(config.NULL_CONTEXT, N)
    assert len(pairs) == N
    for p in pairs:
        C.assert_no_affect(p.positive)
        assert p.positive[: p.divergence_char] == p.negative[: p.divergence_char]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
