"""Hook placement and mask broadcasting, on a stub model.

The layer-indexing convention (steering at L writes to the output of block L-1,
which is exactly the tensor read as hidden_states[L]) is the easiest thing in
this pipeline to get quietly wrong: an off-by-one here would mean the steering
figure and the geometry figure describe different layers. A stub cannot prove
the convention holds for Qwen3 -- Phase A on the real model does that -- but it
does prove the wiring, the tuple handling and the position mask.

Run: python tests/test_steering_hook.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("VOID_DEVICE", "cpu")

import torch          # noqa: E402
import torch.nn as nn  # noqa: E402

import modelio as M   # noqa: E402

D = 16
N_LAYERS = 4


class Block(nn.Module):
    """Mirrors a HF decoder layer: takes and returns a tuple-wrapped tensor."""

    def __init__(self, k):
        super().__init__()
        self.lin = nn.Linear(D, D, bias=False)
        nn.init.eye_(self.lin.weight)
        self.lin.weight.data += 0.01 * k

    def forward(self, x, **_):
        return (x + self.lin(x) * 0.0 + 0.0,)  # identity, so deltas propagate cleanly


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(64, D)
        self.layers = nn.ModuleList(Block(k) for k in range(N_LAYERS))
        self.norm = nn.Identity()


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **_):
        h = self.model.embed_tokens(input_ids)
        states = [h]
        for blk in self.model.layers:
            h = blk(h)[0]
            states.append(h)
        states[-1] = self.model.norm(h)
        return type("Out", (), {"hidden_states": tuple(states), "logits": h})()


class StubTok:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False, **__):
        ids = [ord(c) % 60 + 1 for c in text][:24]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


def _lm():
    return M.LM(model=StubModel(), tokenizer=StubTok(), name="stub",
                n_layers=N_LAYERS, d_model=D)


def test_hook_writes_to_the_layer_it_claims():
    lm = _lm()
    ids = torch.randint(1, 60, (2, 6))
    attn = torch.ones_like(ids)
    direction = torch.zeros(D)
    direction[3] = 1.0
    mag = 2.5
    layer = 2

    base = lm.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    with M.steering(lm, layer, direction, mag, position_mask=None):
        out = lm.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)

    # everything strictly before L is untouched
    for i in range(layer):
        assert torch.allclose(base.hidden_states[i], out.hidden_states[i]), i
    # h[L] shifted by exactly magnitude * direction
    delta = out.hidden_states[layer] - base.hidden_states[layer]
    expect = mag * direction
    assert torch.allclose(delta, expect.expand_as(delta), atol=1e-5), delta[0, 0, :5]
    # and the shift propagates onward (identity blocks)
    assert not torch.allclose(base.hidden_states[-1], out.hidden_states[-1])
    print(f"ok  hook writes h[{layer}] and leaves h[0..{layer - 1}] untouched")


def test_position_mask_selects_tokens():
    lm = _lm()
    ids = torch.randint(1, 60, (2, 6))
    attn = torch.ones_like(ids)
    direction = torch.zeros(D)
    direction[0] = 1.0
    mask = torch.zeros((2, 6), dtype=torch.bool)
    mask[:, -2:] = True  # only the trailing "assistant" positions

    base = lm.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    with M.steering(lm, 1, direction, 3.0, position_mask=mask):
        out = lm.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    delta = (out.hidden_states[1] - base.hidden_states[1])[..., 0]
    assert torch.allclose(delta[:, :-2], torch.zeros(2, 4)), delta
    assert torch.allclose(delta[:, -2:], torch.full((2, 2), 3.0)), delta
    print("ok  position mask steers only the marked positions")


def test_zero_alpha_is_a_no_op_and_handle_is_removed():
    lm = _lm()
    ids = torch.randint(1, 60, (1, 5))
    attn = torch.ones_like(ids)
    d = torch.ones(D)
    base = lm.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    with M.steering(lm, 2, d, 0.0):
        out = lm.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    assert torch.allclose(base.hidden_states[-1], out.hidden_states[-1])
    # after the context exits the model must be clean again
    with M.steering(lm, 2, d, 5.0):
        pass
    after = lm.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
    assert torch.allclose(base.hidden_states[-1], after.hidden_states[-1])
    print("ok  alpha=0 is a no-op and the hook handle is removed on exit")


def test_pad_batch_left_pads_and_aligns_masks():
    lm = _lm()
    ids, attn, steer = M.pad_batch(lm, [[5, 6, 7], [8, 9]],
                                   [[False, True, True], [False, True]])
    assert ids.shape == (2, 3)
    assert attn[1].tolist() == [0, 1, 1]          # left padding
    assert steer[1].tolist() == [False, False, True]
    assert steer[0].tolist() == [False, True, True]
    print("ok  pad_batch left-pads and keeps masks aligned to the right edge")


def test_bare_segments_mask_model_turns():
    lm = _lm()
    ids, mask = M.encode_bare_with_mask(lm, [("question", False), ("answer", True)])
    assert len(ids) == len(mask)
    assert not any(mask[:len("question")])
    assert all(mask[len("question"):])
    print("ok  bare-path segments mark model turns only")


def test_alpha_units():
    # alpha is in units of ALPHA_UNIT_FRAC * median residual norm
    assert M.alpha_to_magnitude(4.0, 50.0) == 4.0 * 0.1 * 50.0
    assert M.alpha_to_magnitude(0.0, 50.0) == 0.0
    print("ok  alpha -> magnitude conversion")


def test_candidate_layers_exclude_endpoints():
    for n in (12, 28, 36):
        cand = M.candidate_layers(n)
        assert 0 not in cand and n not in cand, (n, cand[:3], cand[-3:])
        assert cand == sorted(cand) and len(cand) > 4
    print("ok  candidate layers exclude embeddings and the post-norm state")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} hook tests passed")
