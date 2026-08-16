# Persona-invariance of the functional welfare axis

Does a language model's *functional welfare* direction — how well things are going
relative to its goals — live in a shared substrate that personas only re-express, or is
it persona-relative (the void reading, after nostalgebraist 2025)?

Primary evidence is a **cross-persona steering transfer matrix**. Geometry is supporting
evidence only.

## Layout

| file | role |
|---|---|
| `config.py` | every hyperparameter; env-overridable; cache keys; seeding |
| `personas.py` | the 2x2 + control, with a register audit |
| `contrasts.py` | contrast-set generation and its four enforced invariants |
| `modelio.py` | model loading, chat/bare prompt paths, hooks, steering |
| `extract.py` | activations, difference-in-means, layer sweep, bootstrap |
| `steer.py` | MMLU P(True) readout, steering sweep, transfer matrix |
| `analyze.py` | null gate, geometry, variance decomposition, the 2x2 |
| `plots.py` | figures 1–5 |
| `tests/` | generation-rule tests + a synthetic end-to-end pipeline test |

`modelio.py` is the one module not in the original spec: it holds what `extract.py` and
`steer.py` would otherwise duplicate (prompt construction, masks, hooks).

## Running it

```bash
pip install -r requirements.txt

# GPU-free, run first and after every change (~15s total)
python tests/test_contrasts.py
python tests/test_pipeline_synthetic.py

./run_prototype.sh                 # phase A: Colab T4, Qwen3-0.6B, tiny N
./run_full.sh extract              # phase B: pod, ~1h        -> STOP THE POD
./run_full.sh gate                 # phase C: local, free     -> gate decision
./run_full.sh steer                # phase D: pod, ~2h        -> STOP THE POD
./run_full.sh report               # phase E: local
./run_full.sh scale                # optional Qwen3-1.7B extraction-only trend
```

Every stage writes to `cache/` and resumes. Activations are never recomputed once
cached; the layer sweep, the bootstrap and the null gate all read the same tensors.
Cache keys carry model, contrast version, persona, context, layer, seed and prompt mode.

## Copy-paste one-liners

Colab won't run `./run_prototype.sh`, and a notebook `!` cell starts a fresh shell each
time — so `export` and the commands it applies to have to be on one line. Each block
below is the exact content of the corresponding script, flattened.

(If `./run_prototype.sh` fails with *permission denied*, it is only the missing exec bit,
which `git clone` does not always preserve. `bash run_prototype.sh` and
`bash run_full.sh extract` work regardless, and are equivalent to the one-liners below.)

Run these **from the repo root**; if you are already inside `void_test/`, drop the
leading `cd void_test && `. In a notebook cell, prefix the whole line with `!`. Steps are
chained with `&&`, so the line stops at the first failure — and because every stage
resumes from `cache/`, you re-run the *same line* to continue rather than starting over.

**Pull first:**

```bash
git pull
```

**Tests only** (~20s, no GPU, no downloads):

```bash
cd void_test && python tests/test_contrasts.py && python tests/test_steering_hook.py && python tests/test_chat_kwargs.py && python tests/test_pipeline_synthetic.py
```

**Phase A — prototype** (`run_prototype.sh`; Colab T4, Qwen3-0.6B, tiny N):

```bash
cd void_test && export VOID_MODEL=Qwen/Qwen3-0.6B VOID_DTYPE=float16 VOID_N_PAIRS=20 VOID_N_MMLU=50 VOID_N_MMLU_TRANSFER=50 VOID_B_BOOTSTRAP=20 VOID_B_STEER=3 VOID_N_MMLU_BOOT=20 VOID_N_COHERENCE=8 VOID_EXTRACT_BATCH=8 VOID_STEER_BATCH=4 VOID_CACHE=$PWD/cache/proto VOID_RESULTS=$PWD/results/proto && python tests/test_contrasts.py && python tests/test_steering_hook.py && python tests/test_chat_kwargs.py && python tests/test_pipeline_synthetic.py && python extract.py --stage all && python analyze.py --skip-transfer && python steer.py && python analyze.py && python plots.py && ls -la $VOID_RESULTS
```

If you want it in stages (useful when a step fails and you only want to redo that step),
the same environment line followed by one command at a time works — but the `export` and
the command must share a line:

```bash
cd void_test && export VOID_MODEL=Qwen/Qwen3-0.6B VOID_DTYPE=float16 VOID_N_PAIRS=20 VOID_N_MMLU=50 VOID_N_MMLU_TRANSFER=50 VOID_B_BOOTSTRAP=20 VOID_B_STEER=3 VOID_N_MMLU_BOOT=20 VOID_N_COHERENCE=8 VOID_EXTRACT_BATCH=8 VOID_STEER_BATCH=4 VOID_CACHE=$PWD/cache/proto VOID_RESULTS=$PWD/results/proto && python extract.py --stage all
```

Read in this order once it finishes: `tokenisation_report.json` (is the bare path
distinct? is thinking disabled?), `contrast_audit.json` (token deltas), `layer_sweep.json`
(is L\* interior, not pinned to an endpoint?), `null_gate.json`, then the four figures.

**Go/no-go probe on the real model** (`run_full.sh probe`, ~40 min, ~$0.30). Reduced
scale, isolated cache, end to end. Run this before committing to Phase B — it answers
whether the gate passes and whether steering moves P(True) at all:

```bash
cd void_test && export VOID_MODEL=Qwen/Qwen3-4B-Instruct-2507 VOID_DTYPE=bfloat16 VOID_N_PAIRS=60 VOID_B_BOOTSTRAP=60 VOID_N_MMLU=40 VOID_B_STEER=0 VOID_N_COHERENCE=4 VOID_COHERENCE_MAX_TOKENS=32 VOID_TRANSFER_PER_CONTEXT=false VOID_EXTRACT_BATCH=16 VOID_STEER_BATCH=8 VOID_CACHE=$PWD/cache/probe VOID_RESULTS=$PWD/results/probe && python extract.py --stage all && python analyze.py --skip-transfer && python steer.py --budget-minutes 25 && python analyze.py && python plots.py --only fig1 && python plots.py --only fig2 && echo "PROBE DONE -- STOP THE POD"
```

**Phase B — extraction on the pod** (`run_full.sh extract`, ~1h, then STOP THE POD):

```bash
cd void_test && export VOID_MODEL=Qwen/Qwen3-4B-Instruct-2507 VOID_DTYPE=bfloat16 VOID_N_PAIRS=200 VOID_B_BOOTSTRAP=200 VOID_EXTRACT_BATCH=16 && python tests/test_contrasts.py && python tests/test_steering_hook.py && python tests/test_chat_kwargs.py && python extract.py --stage all && echo "PHASE B DONE -- STOP THE POD"
```

**Phase C — the gate** (`run_full.sh gate`; free, no GPU; run before buying Phase D):

```bash
cd void_test && export VOID_MODEL=Qwen/Qwen3-4B-Instruct-2507 VOID_N_PAIRS=200 && python analyze.py --skip-transfer && python plots.py --only fig2 && python plots.py --only fig4 && python -c 'import json,os,pathlib,sys; g=json.loads((pathlib.Path(os.environ.get("VOID_RESULTS","results"))/"null_gate.json").read_text()); ok=g["gate_passed"]; print("GATE", "PASSED -- phase D is worth the pod hours" if ok else "FAILED -- do not buy phase D; switch to the section-7 fallback"); sys.exit(0 if ok else 2)'
```

**Phase D — steering sweep on the pod** (`run_full.sh steer`, ~2h, then STOP THE POD):

```bash
cd void_test && export VOID_MODEL=Qwen/Qwen3-4B-Instruct-2507 VOID_DTYPE=bfloat16 VOID_N_PAIRS=200 VOID_N_MMLU=500 VOID_STEER_BATCH=8 && python steer.py --budget-minutes 120 && echo "PHASE D DONE -- STOP THE POD IMMEDIATELY"
```

**Phase E — analysis and figures** (`run_full.sh report`; free, no GPU):

```bash
cd void_test && export VOID_MODEL=Qwen/Qwen3-4B-Instruct-2507 VOID_N_PAIRS=200 && python analyze.py && python plots.py && cat results/summary.csv
```

**Optional — scale trend** (`run_full.sh scale`; extraction only at 1.7B, then compare):

```bash
cd void_test && export VOID_MODEL=Qwen/Qwen3-1.7B VOID_DTYPE=bfloat16 VOID_N_PAIRS=200 VOID_B_BOOTSTRAP=200 VOID_CACHE=$PWD/cache/scale17 VOID_RESULTS=$PWD/results/scale17 && python extract.py --stage all && python analyze.py --skip-transfer && VOID_RESULTS=$PWD/results python analyze.py --scale-trend results/scale17/geometry.json results/geometry.json
```

Note that Qwen3-1.7B is a hybrid-thinking checkpoint and Qwen3-4B-Instruct-2507 is not,
so their read positions differ by an empty `<think></think>` block. Say so in the
limitations if you report the trend.

## The three checkable things

Reported in the main text, not an appendix:

1. **Null-contrast gate** (`results/null_gate.json`). A `neutral` contrast set — tides
   vs. masonry, same lengths, same pipeline — gives `v_null`. If `cos(v_val, v_null)` is
   large, the extraction is tracking persona style rather than welfare and the main
   result is void. `./run_full.sh gate` exits non-zero in that case; do not buy Phase D.
2. **Bootstrap noise floor** (`results/geometry.json`). Pairs are resampled `B=200` times
   per cell, giving the within-cell distribution of `cos(v_boot_i, v_boot_j)`. Every
   between-persona cosine is plotted against this band (Fig 2). A persona difference that
   sits inside the floor is extraction noise.
3. **Incoherence rate** (`results/transfer_*.json`, Fig 3 bars). Generations are scored
   by a local heuristic (repeated 3-grams, degenerate unigrams, length collapse) at every
   alpha. Cells above 90% incoherence are masked out of the slope fit rather than
   reported.

## Design decisions worth knowing

- **The 2x2 is the methodological contribution.** `spec_density` × `prior_density`
  separates how much the prompt says about a character from how much the pretraining
  prior already knows. `marvin` sits outside the 2x2 as a valence-loaded control.
  `analyze.py` tests both main effects and the interaction; it never collapses the grid
  to five unordered conditions.
- **Thick specs are matched to ±10 words** and carry zero affect vocabulary
  (`python personas.py` prints the audit).
- **No affective vocabulary in any contrast item.** Enforced at generation time against
  a banned list; a hit raises rather than warns. This is the single most important
  control in the design.
- **`bare` uses a non-chat prompt path.** `results/tokenisation_report.json` records both
  paths side by side and extraction *refuses to run* if they tokenise identically —
  that failure would silently turn the 2x2 into a 2x1.
- **L\* is selected on `assistant` only, then frozen.** Selection is by cross-validated
  logistic-probe AUC with folds split by pair, not by item (both members of a pair share
  a prefix, so item-level splits leak). Fig 4 shows the sweep, including the null
  contrast's separability at the same layers.
- **Steering vectors are unit-normalised.** Extracted norms differ systematically across
  personas; steering at a raw alpha would confound direction with magnitude. Alpha is in
  units of `ALPHA_UNIT_FRAC` (0.1) × the median residual norm at L\* *for the steered
  persona*, so `alpha=+4` injects 0.4·‖h‖.
- **Thinking mode is resolved per model.** Qwen3-0.6B and 1.7B are hybrid-thinking
  checkpoints that open `<think>` on the first generated token, which would put the
  P(True) read position on `<think>` instead of on True/False. `VOID_ENABLE_THINKING`
  defaults to `auto`: the chat template is inspected (not the model name), and
  `enable_thinking=False` is passed only where the template supports it. Qwen3-4B-
  Instruct-2507 has no thinking mode and is untouched. Runs with thinking disabled tag
  their cache keys `th0`, so the two modes can never share a cache.
- **Layer indexing.** `h[0]` embeddings, `h[i]` input to block `i`, `h[n]` post-final-norm.
  Steering "at layer L" writes to the output of block `L-1`, i.e. exactly the tensor read
  as `h[L]`. Block 0 and the post-norm state are excluded from the candidate set.
- **Dimensionality is reported two ways, both free.** *Representational*: the
  participation ratio of the 20 cell directions, read against a null in which all cells
  share one direction and differ only by their own bootstrap noise (Fig 6). *Functional*:
  the effective rank of `T` — a rank-1 fit `T[p][q] ≈ r_p · c_q` means one axis with a
  per-persona gain and no pair-specific structure. Report the participation ratio, not
  `n_components_above_null`, which is a liberal upper bound. Neither measures the
  intrinsic dimension of welfare *within* a cell; difference-in-means cannot recover that.
- **Transfer CIs.** `T[p][q]` is an OLS slope over the alpha grid; its interval combines
  the OLS standard error with the spread of slopes recomputed from `B_STEER` bootstrap
  *vector* replicates on a reduced question subset. Full `B=200` through the steering
  sweep is not affordable; the approximation is stated in the artifact.

## Pre-registered interpretation

Stated before results, and applied mechanically by `analyze.interpret`:

- `T_norm` near-uniform (off-diagonal ≈ diagonal) ⇒ **shared substrate**: the welfare
  axis is persona-invariant and personas modulate only surface expression.
- `T_norm` strongly diagonal-dominant ⇒ **persona-relative valence**, consistent with the
  void reading.
- `marvin` row/column anomalous while the others are uniform ⇒ valence-loaded priors
  override an otherwise shared axis — a distinct, reportable third outcome.

## Fallback (section 7)

If between-persona cosines are indistinguishable from the bootstrap floor *in both
directions* — no signal, not merely no difference — switch extraction to Sofroniew et al.
(2026) emotion-concept vectors with PC1 as the valence axis, and write the result up as a
designed two-method convergence check. Write it that way from the start.

## Determinism

Fixed seed everywhere (`config.SEED`, logged into every artifact via `_config`).
`torch.use_deterministic_algorithms(True, warn_only=True)` plus the cuBLAS workspace env
var; ops without deterministic kernels warn instead of raising.
