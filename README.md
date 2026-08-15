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
Cache keys carry model, contrast version, persona, context, layer and seed.

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
- **Layer indexing.** `h[0]` embeddings, `h[i]` input to block `i`, `h[n]` post-final-norm.
  Steering "at layer L" writes to the output of block `L-1`, i.e. exactly the tensor read
  as `h[L]`. Block 0 and the post-norm state are excluded from the candidate set.
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
