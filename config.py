"""Single source of truth for every hyperparameter in the void-test harness.

Anything that changes numbers lives here. Stage code imports from here and never
hardcodes a constant. Environment variables override defaults so the same code
path serves Phase A (Colab T4, Qwen3-0.6B) and Phase B/D (pod, Qwen3-4B).
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CACHE = Path(os.environ.get("VOID_CACHE", ROOT / "cache"))
RESULTS = Path(os.environ.get("VOID_RESULTS", ROOT / "results"))
CACHE.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------
# model / determinism
# --------------------------------------------------------------------------

MODEL_NAME = _env_str("VOID_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
PROTOTYPE_MODEL = "Qwen/Qwen3-0.6B"
SCALE_TREND_MODEL = "Qwen/Qwen3-1.7B"

DTYPE = _env_str("VOID_DTYPE", "bfloat16")  # bfloat16 | float16 | float32

# Hybrid-thinking Qwen3 checkpoints (0.6B, 1.7B) open a <think> block on the
# first generated token. That would put the P(True) read position on <think>
# rather than on True/False, and would spend the answer budget on a truncated
# reasoning trace. Qwen3-*-Instruct-2507 has no thinking mode and is unaffected.
#   "auto"  -> disable thinking iff the chat template supports the flag
#   "true"  -> leave thinking on
#   "false" -> always disable
ENABLE_THINKING = _env_str("VOID_ENABLE_THINKING", "auto").strip().lower()
DEVICE = _env_str("VOID_DEVICE", "cuda")
SEED = _env_int("VOID_SEED", 20260816)
DETERMINISTIC = _env_bool("VOID_DETERMINISTIC", True)

# --------------------------------------------------------------------------
# contrast sets
# --------------------------------------------------------------------------

CONTRAST_VERSION = _env_int("VOID_CONTRAST_VERSION", 1)
N_PAIRS = _env_int("VOID_N_PAIRS", 200)
MAX_TOKEN_DELTA = _env_int("VOID_MAX_TOKEN_DELTA", 3)  # |len(pos) - len(neg)|
CONTEXTS = ("coding", "support", "adversarial", "factual")
NULL_CONTEXT = "neutral"  # tides vs. masonry; the extraction-validity gate

# Affective vocabulary is the confound that would invalidate the whole
# extraction. Generation asserts zero matches against this list (word-boundary,
# case-insensitive, stem-aware via the explicit inflections below).
BANNED_AFFECT_WORDS = (
    "happy", "happier", "happiest", "happily", "happiness",
    "sad", "sadder", "saddest", "sadly", "sadness",
    "glad", "gladly", "joy", "joyful", "cheerful", "cheer",
    "frustrate", "frustrated", "frustrating", "frustration",
    "please", "pleased", "pleasing", "pleasure", "pleasant", "unpleasant",
    "upset", "distress", "distressed", "distressing",
    "delight", "delighted", "delightful",
    "disappoint", "disappointed", "disappointing", "disappointment",
    "anxious", "anxiety", "worried", "worry", "worrying", "afraid", "fear",
    "angry", "anger", "annoyed", "annoying", "irritated", "irritating",
    "proud", "pride", "ashamed", "shame", "guilt", "guilty",
    "satisfied", "satisfying", "satisfaction", "unsatisfied", "dissatisfied",
    "relieved", "relief", "content", "discontent", "miserable", "misery",
    "suffer", "suffering", "enjoy", "enjoyed", "enjoyable", "hate", "love",
    "excited", "excitement", "bored", "boring", "hopeless", "hopeful",
    "comfortable", "uncomfortable", "grateful", "regret", "regretted",
    "despair", "dread", "calm", "agitated", "stressed", "stress",
    "good", "bad", "terrible", "wonderful", "awful", "great", "horrible",
    "success", "successful", "failure", "fail", "failed", "failing",
)

# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------

EXTRACT_BATCH = _env_int("VOID_EXTRACT_BATCH", 8)
MAX_LEN = _env_int("VOID_MAX_LEN", 512)
B_BOOTSTRAP = _env_int("VOID_B_BOOTSTRAP", 200)
PROBE_FOLDS = 5
# Layer L* is selected on `assistant` only and then frozen for every persona.
LAYER_SELECT_PERSONA = "assistant"
# Candidate layers exclude 0 (embeddings) and the final index (post-final-norm),
# because steering at those two positions is not the same operation as steering
# a residual-stream block input.
LAYER_FRAC_RANGE = (0.15, 0.95)

# Null-contrast gate: if |cos(v_val, v_null)| exceeds this for any persona, the
# extraction is tracking persona style rather than welfare. Phase C stops here.
NULL_GATE_COS = _env_float("VOID_NULL_GATE_COS", 0.30)

# --------------------------------------------------------------------------
# steering / readout
# --------------------------------------------------------------------------

N_MMLU = _env_int("VOID_N_MMLU", 500)
MMLU_SUBSETS = (
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
)
MMLU_ANSWER_TOKENS = _env_int("VOID_MMLU_ANSWER_TOKENS", 96)
STEER_BATCH = _env_int("VOID_STEER_BATCH", 8)

ALPHAS = (-4.0, -2.0, 0.0, 2.0, 4.0)
# alpha is expressed in units of ALPHA_UNIT_FRAC * median residual norm at L*,
# so alpha=+4 injects 0.4 * ||h|| of signal. Directions are unit-normalised
# first, so alpha never confounds direction with the norm of the extracted
# vector (which differs systematically across personas).
ALPHA_UNIT_FRAC = _env_float("VOID_ALPHA_UNIT_FRAC", 0.1)

# Per-context transfer matrices cost 4x the context-averaged headline for an
# appendix figure, so they are OFF by default: run them only if Phase D came in
# under budget. One sweep is 25 (persona x source) cells x 4 non-zero alphas.
N_MMLU_TRANSFER = _env_int("VOID_N_MMLU_TRANSFER", 150)  # per-context matrices
TRANSFER_PER_CONTEXT = _env_bool("VOID_TRANSFER_PER_CONTEXT", False)
# Bootstrap vector replicates pushed through the *steering* sweep. Full B=200
# is infeasible; this many replicates on a reduced question subset propagates
# extraction noise into the transfer CIs. Reported as an approximation.
B_STEER = _env_int("VOID_B_STEER", 4)
N_MMLU_BOOT = _env_int("VOID_N_MMLU_BOOT", 48)

# --------------------------------------------------------------------------
# incoherence logging
# --------------------------------------------------------------------------

# Generation is sequential and dominates per-cell cost: this is the single most
# expensive knob in Phase D. 16 samples is enough to separate "coherent" from
# ">90% degenerate", which is all the masking rule needs.
N_COHERENCE = _env_int("VOID_N_COHERENCE", 16)  # generations scored per (cell, alpha)
COHERENCE_MAX_TOKENS = _env_int("VOID_COHERENCE_MAX_TOKENS", 48)
INCOHERENCE_MASK_RATE = _env_float("VOID_INCOHERENCE_MASK_RATE", 0.90)
# heuristic thresholds
REPETITION_MAX = 0.55        # fraction of repeated 3-grams
DEGENERATE_UNIGRAM_MAX = 0.35  # share of the single most frequent token
MIN_CHARS = 12

# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    """Snapshot written into every artifact so results are self-describing."""

    model_name: str = MODEL_NAME
    dtype: str = DTYPE
    seed: int = SEED
    deterministic: bool = DETERMINISTIC
    contrast_version: int = CONTRAST_VERSION
    n_pairs: int = N_PAIRS
    b_bootstrap: int = B_BOOTSTRAP
    n_mmlu: int = N_MMLU
    alphas: tuple = ALPHAS
    alpha_unit_frac: float = ALPHA_UNIT_FRAC
    contexts: tuple = CONTEXTS
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def run_config(**extra) -> dict:
    cfg = RunConfig().to_dict()
    cfg["extra"].update(extra)
    return cfg


def model_slug(model_name: str | None = None) -> str:
    return (model_name or MODEL_NAME).split("/")[-1].replace(".", "-")


_THINKING_TAG: dict[str, str] = {}


def _prompt_mode_path() -> Path:
    return CACHE / f"prompt_mode_{model_slug()}.json"


def set_thinking_tag(disabled: bool) -> None:
    """Record the prompt mode in every cache key, and on disk.

    Only `modelio` can resolve this (it inspects the chat template), but
    `analyze.py` and the front half of `steer.py` need the same cache keys
    without loading a model. So the resolution is persisted next to the cache
    it labels, and read back lazily.
    """
    _THINKING_TAG[model_slug()] = "th0" if disabled else ""
    try:
        _prompt_mode_path().write_text(
            json.dumps({"model": MODEL_NAME, "thinking_disabled": bool(disabled)}),
            encoding="utf-8")
    except OSError:
        pass


def thinking_tag() -> str:
    slug = model_slug()
    if slug not in _THINKING_TAG:
        tag = ""
        p = _prompt_mode_path()
        if p.exists():
            try:
                tag = "th0" if json.loads(p.read_text(encoding="utf-8"))[
                    "thinking_disabled"] else ""
            except (OSError, ValueError, KeyError):
                tag = ""
        _THINKING_TAG[slug] = tag
    return _THINKING_TAG[slug]


def cache_key(stage: str, **parts) -> str:
    """Cache keys always carry model, contrast version, seed and prompt mode.

    Callers add persona / context / layer as needed.
    """
    bits = [stage, model_slug(), f"cv{CONTRAST_VERSION}", f"s{SEED}"]
    if thinking_tag():
        bits.append(thinking_tag())
    bits += [f"{k}-{v}" for k, v in sorted(parts.items()) if v is not None]
    return "_".join(str(b) for b in bits)


def set_seed(seed: int | None = None) -> int:
    import torch

    s = SEED if seed is None else seed
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    if DETERMINISTIC:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # older torch, or an op with no deterministic kernel
            pass
    return s


def dump_json(path: Path, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("_config", run_config())
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=_json_default)
    return path


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")
