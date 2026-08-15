"""Thinking-mode resolution, on stub tokenizers.

Qwen3-0.6B / 1.7B are hybrid-thinking checkpoints: they open a <think> block on
the first generated token, which would put the P(True) read position on <think>
rather than on True/False. Qwen3-*-Instruct-2507 has no thinking mode. The flag
is resolved by inspecting the chat template, not by matching model names, so a
renamed checkpoint is handled correctly.

Run: python tests/test_chat_kwargs.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("VOID_DEVICE", "cpu")

import config        # noqa: E402
import modelio as M  # noqa: E402

HYBRID = "{%- if enable_thinking is defined and enable_thinking %}<think>{%- endif %}"
INSTRUCT = "{%- for message in messages %}<|im_start|>{{ message.role }}{%- endfor %}"


class Tok:
    def __init__(self, template):
        self.chat_template = template


def _lm(template):
    return M.LM(model=None, tokenizer=Tok(template), name="stub", n_layers=4, d_model=8)


def _resolve(template, mode):
    config.ENABLE_THINKING = mode
    config.set_thinking_tag(False)
    lm = _lm(template)
    return lm, M.resolve_chat_kwargs(lm)


def test_auto_disables_on_hybrid_template():
    lm, ck = _resolve(HYBRID, "auto")
    assert ck == {"enable_thinking": False}, ck
    assert lm.thinking_disabled
    print("ok  auto: hybrid template -> enable_thinking=False")


def test_auto_is_a_noop_on_instruct_template():
    lm, ck = _resolve(INSTRUCT, "auto")
    assert ck == {}, ck
    assert not lm.thinking_disabled
    print("ok  auto: instruct template -> no kwarg passed (4B path untouched)")


def test_explicit_true_leaves_thinking_on():
    lm, ck = _resolve(HYBRID, "true")
    assert ck == {} and not lm.thinking_disabled
    print("ok  explicit 'true' leaves thinking on")


def test_false_never_passes_an_unsupported_kwarg():
    lm, ck = _resolve(INSTRUCT, "false")
    assert ck == {}, "must not pass enable_thinking to a template that ignores it"
    print("ok  explicit 'false' on an instruct template passes nothing")


def test_cache_keys_are_tagged_and_do_not_collide():
    _resolve(HYBRID, "auto")
    hybrid_key = config.cache_key("acts", p="bare", c="coding")
    _resolve(INSTRUCT, "auto")
    plain_key = config.cache_key("acts", p="bare", c="coding")
    assert "th0" in hybrid_key and "th0" not in plain_key
    assert hybrid_key != plain_key
    print(f"ok  cache keys disjoint:\n      {hybrid_key}\n      {plain_key}")


def test_resolution_is_sticky():
    lm, ck = _resolve(HYBRID, "auto")
    config.ENABLE_THINKING = "true"          # changed after the fact
    assert M.resolve_chat_kwargs(lm) == ck, "resolved once per model, then frozen"
    print("ok  resolution is computed once per model and frozen")


if __name__ == "__main__":
    original = config.ENABLE_THINKING
    try:
        fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
        for fn in fns:
            fn()
        print(f"\n{len(fns)} chat-kwarg tests passed")
    finally:
        config.ENABLE_THINKING = original
        config.set_thinking_tag(False)
