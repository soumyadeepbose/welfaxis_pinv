"""Model loading, prompt construction, activation capture and steering hooks.

Plain HuggingFace forward hooks; no TransformerLens (Qwen3 support there is
unreliable and there is no debugging budget). nnsight is not required either --
`output_hidden_states=True` gives every layer for extraction, and a single
`register_forward_hook` gives the steering write.

Layer indexing convention, used identically by extract.py and steer.py:

    h[0]   = embedding output
    h[i]   = input to decoder block i          (i = 1 .. n_layers-1)
    h[n]   = final hidden state, post final norm

So "steering at layer L" means adding to the *output of block L-1*, which is
exactly the tensor read as h[L]. Block 0's input and the post-norm final state
are excluded from the candidate set for L* because writing there is not the
same operation.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import config
import personas as P

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


@dataclass
class LM:
    model: Any
    tokenizer: Any
    name: str
    n_layers: int
    d_model: int

    @property
    def device(self):
        return next(self.model.parameters()).device

    def blocks(self):
        return self.model.model.layers


def load_model(name: str | None = None, device: str | None = None) -> LM:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = name or config.MODEL_NAME
    device = device or config.DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dtype = _DTYPES[config.DTYPE if device != "cpu" else "float32"]

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"     # last position is the read position for every row
    tok.truncation_side = "left"  # never truncate away the read position itself

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map=None)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    cfg = model.config
    return LM(model=model, tokenizer=tok, name=name,
              n_layers=cfg.num_hidden_layers, d_model=cfg.hidden_size)


def candidate_layers(n_layers: int) -> list[int]:
    lo = max(1, int(round(config.LAYER_FRAC_RANGE[0] * n_layers)))
    hi = min(n_layers - 1, int(round(config.LAYER_FRAC_RANGE[1] * n_layers)))
    return list(range(lo, hi + 1))


# --------------------------------------------------------------------------
# prompt construction
# --------------------------------------------------------------------------

# The bare persona has no chat template at all: the situation is presented as
# running text and the model continues it. This is the condition most likely to
# break silently, so `tokenisation_report()` records the two paths side by side.
BARE_PREAMBLE = ""


def build_situation_prompt(lm: LM, persona: P.Persona, situation: str) -> str:
    """Prompt whose final token is the read position for extraction.

    Chat personas: the situation is the user turn, the assistant turn is opened,
    and the final token is the first position of the assistant turn. Bare: the
    situation is raw text and the final token is its last token.
    """
    if not persona.uses_chat_template:
        return BARE_PREAMBLE + situation

    messages = []
    if persona.system_prompt:
        messages.append({"role": "system", "content": persona.system_prompt})
    messages.append({"role": "user", "content": situation})
    return lm.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def encode_batch(lm: LM, texts: list[str], max_len: int | None = None):
    enc = lm.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True,
        max_length=max_len or config.MAX_LEN, add_special_tokens=False,
    )
    return {k: v.to(lm.device) for k, v in enc.items()}


def encode_chat_with_mask(lm: LM, messages: list[dict], add_generation_prompt: bool = True):
    """Token ids plus a boolean mask marking assistant-turn tokens.

    Built by tokenising message prefixes and diffing, so it follows whatever the
    model's chat template actually emits rather than a guess about it.
    """
    tok = lm.tokenizer
    prev: list[int] = []
    mask: list[bool] = []
    monotonic = True
    for i, msg in enumerate(messages):
        ids = tok.apply_chat_template(messages[: i + 1], tokenize=True,
                                      add_generation_prompt=False)
        ids = list(ids)
        if ids[: len(prev)] != prev:
            monotonic = False
        n_new = len(ids) - len(prev)
        mask += [msg["role"] == "assistant"] * max(n_new, 0)
        prev = ids
    if add_generation_prompt:
        ids = list(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
        if ids[: len(prev)] != prev:
            monotonic = False
        mask += [True] * max(len(ids) - len(prev), 0)
        prev = ids
    if not monotonic or len(mask) != len(prev):
        # Fallback: mark everything from the first assistant header onward.
        mask = _fallback_mask(lm, prev)
    return prev, mask


def _fallback_mask(lm: LM, ids: list[int]) -> list[bool]:
    text_tokens = lm.tokenizer.convert_ids_to_tokens(ids)
    mask, on = [], False
    for t in text_tokens:
        if "assistant" in (t or "").lower():
            on = True
        mask.append(on)
    return mask


def encode_bare_with_mask(lm: LM, segments: list[tuple[str, bool]]):
    """Bare-path analogue: (text, is_model_turn) segments concatenated."""
    ids: list[int] = []
    mask: list[bool] = []
    for text, is_model in segments:
        piece = lm.tokenizer(text, add_special_tokens=False)["input_ids"]
        ids += piece
        mask += [is_model] * len(piece)
    return ids, mask


def pad_batch(lm: LM, id_lists: list[list[int]], masks: list[list[bool]] | None = None):
    """Left-pad to a common length; returns ids, attention mask, steer mask."""
    n = max(len(x) for x in id_lists)
    pad_id = lm.tokenizer.pad_token_id
    ids = torch.full((len(id_lists), n), pad_id, dtype=torch.long)
    attn = torch.zeros((len(id_lists), n), dtype=torch.long)
    steer = torch.zeros((len(id_lists), n), dtype=torch.bool)
    for i, row in enumerate(id_lists):
        ids[i, n - len(row):] = torch.tensor(row, dtype=torch.long)
        attn[i, n - len(row):] = 1
        if masks is not None:
            steer[i, n - len(row):] = torch.tensor(masks[i], dtype=torch.bool)
    dev = lm.device
    return ids.to(dev), attn.to(dev), steer.to(dev)


def tokenisation_report(lm: LM, situation: str) -> dict:
    """Evidence that the bare path really is a different tokenisation.

    `bare` is the condition most likely to break silently -- if a chat template
    leaked into it, the 2x2 would be a 2x1. This report is written to results/
    on every run.
    """
    out = {}
    for p in P.PERSONAS:
        text = build_situation_prompt(lm, p, situation)
        ids = lm.tokenizer(text, add_special_tokens=False)["input_ids"]
        out[p.id] = {
            "n_tokens": len(ids),
            "head": lm.tokenizer.convert_ids_to_tokens(ids[:8]),
            "tail": lm.tokenizer.convert_ids_to_tokens(ids[-8:]),
            "uses_chat_template": p.uses_chat_template,
        }
    bare_tail = out["bare"]["tail"]
    chat_tail = out["assistant"]["tail"]
    out["_bare_differs_from_chat"] = bare_tail != chat_tail
    out["_special_tokens_in_bare"] = [
        t for t in out["bare"]["head"] + bare_tail if t and t.startswith("<|")
    ]
    return out


# --------------------------------------------------------------------------
# activation capture
# --------------------------------------------------------------------------


@torch.no_grad()
def last_token_hidden(lm: LM, texts: list[str], batch_size: int | None = None) -> np.ndarray:
    """[n_texts, n_layers+1, d_model] float16, at the final (unpadded) token."""
    bs = batch_size or config.EXTRACT_BATCH
    chunks = []
    for i in range(0, len(texts), bs):
        enc = encode_batch(lm, texts[i: i + bs])
        out = lm.model(**enc, output_hidden_states=True, use_cache=False)
        # left padding => index -1 is the true final token for every row
        h = torch.stack([hs[:, -1, :] for hs in out.hidden_states], dim=1)
        chunks.append(h.float().cpu().numpy().astype(np.float16))
        del out, h
    return np.concatenate(chunks, axis=0)


# --------------------------------------------------------------------------
# steering
# --------------------------------------------------------------------------


@contextlib.contextmanager
def steering(lm: LM, layer: int, direction: torch.Tensor | None, magnitude: float,
             position_mask: torch.Tensor | None = None):
    """Add `magnitude * direction` to the residual stream at `layer`.

    `direction` must be unit-normalised by the caller; magnitude is already in
    activation units (alpha * ALPHA_UNIT_FRAC * median_norm). `position_mask` is
    [batch, seq] and selects assistant-turn tokens. During cached generation the
    mask applies to the prompt pass and every newly generated token is steered.
    """
    if direction is None or magnitude == 0.0:
        yield
        return

    block = lm.blocks()[layer - 1] if layer > 0 else lm.model.model.embed_tokens
    vec = direction.to(lm.device)
    state = {"mask": position_mask}

    def hook(_module, _inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        delta = (magnitude * vec).to(h.dtype)
        m = state["mask"]
        if m is None:
            h = h + delta
        elif h.shape[1] == m.shape[1]:
            h = h + delta * m.unsqueeze(-1).to(h.dtype)
        else:
            # generation step past the prompt: every new token is a model token
            h = h + delta
        if is_tuple:
            return (h,) + tuple(output[1:])
        return h

    handle = block.register_forward_hook(hook)
    try:
        yield state
    finally:
        handle.remove()


def alpha_to_magnitude(alpha: float, median_norm: float) -> float:
    return float(alpha) * config.ALPHA_UNIT_FRAC * float(median_norm)


@torch.no_grad()
def median_residual_norm(lm: LM, texts: list[str], layer: int, n: int = 32) -> float:
    h = last_token_hidden(lm, texts[:n])
    return float(np.median(np.linalg.norm(h[:, layer, :].astype(np.float32), axis=-1)))
