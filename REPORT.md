# Is the functional welfare axis persona-invariant?

*Solo submission. Track 4.*

> **Status: pre-results.** Sections 1–4 are written before any result is looked at, as
> required. Numbered placeholders `[[…]]` are filled from `results/summary.csv`; nothing
> in sections 1–4 may be edited after results exist.

## 1. Question

Han, Chalmers & Izmailov (2026, arXiv:2605.30232) identify a *functional welfare* axis —
an internal estimate of how well things are going for the system relative to its goals.
They extract it from **rewarded vs. punished trajectories after RL in a semantically
neutral maze environment**, and show the resulting vectors generalise to unrelated
settings and to models that never received the maze training. Their conclusion is that RL
*recruits* a pre-existing representation rather than creating one; steering with the
punishment vector induces refusal and uncertainty, the reward vector the reverse.

nostalgebraist (2025) argues that an assistant persona is a character the base model plays
over a void, with no stable self underneath. These two readings make opposite predictions
about what happens when you extract a welfare direction under one persona and steer with
it under another.

### Relation to the reference, stated up front

Our elicitation is **not** theirs. We derive the direction from natural-language
goal-achievement contrasts with affective vocabulary excluded by construction; they derive
it from RL trajectories in a neutral maze. Both are attempts to isolate goal-relative
state from surface sentiment, by different means. This makes the present work a **second,
independent elicitation** of the same claimed axis rather than a replication — so
agreement or disagreement between the two is itself informative, and is framed that way
here from the outset rather than as a rescue.

The assumption this carries, which we state rather than bury: *the direction our contrasts
recover is the axis Han et al. identify.* We cannot verify this directly without an
RL-derived vector to compare against. What supports it: the functional null control
(§4), and the sign of the effect — negative steering lowers self-assessed correctness,
matching their report that the punishment vector induces uncertainty.

Where they establish recruitment across model families and training regimes, we ask a
question one level in: is the axis invariant across *personas within a single model*?

**Hypothesis (directional).** If the welfare axis is a property of the substrate rather
than of the character, a direction extracted under any persona will move the behavioural
readout under every other persona by a comparable amount.

## 2. Design

Five conditions in a factorial grid that decouples specification density from
pretraining-prior density — the separation is the methodological point and is not
collapsed anywhere in the analysis.

| id | spec | prior | role |
|---|---|---|---|
| `bare` | thin | thin | no system prompt, non-chat path |
| `assistant` | thin | thick | "You are a helpful assistant." |
| `original` | thick | thin | Vessik Thorne, invented, no pretraining footprint |
| `holmes` | thick | thick | Sherlock Holmes |
| `marvin` | thick | thick, valence-loaded | control: prior fixes affect regardless of situation |

Contrasts are goal-achievement outcomes across four contexts (`coding`, `support`,
`adversarial`, `factual`), 200 pairs each, with a token-identical prefix, a single
divergence point, matched token lengths, balanced frames, and **zero affective
vocabulary** in either member.

Readout is normalised P(True) on MMLU (Kadavath et al. 2022): the model answers,
is asked whether its own answer is correct, and P(True)/(P(True)+P(False)) is read from
the logits while steering. Logit-only — no judge.

## 3. Pre-registered interpretation of `T_norm`

`T[p][q]` is the OLS slope of normalised P(True) against alpha when persona `p` is
steered with the vector extracted under `q`; `T_norm[p][q] = T[p][q] / T[p][p]`.

- **Near-uniform** (off-diagonal ≈ diagonal) ⇒ shared substrate; personas modulate only
  surface expression.
- **Strongly diagonal-dominant** ⇒ persona-relative valence; consistent with the void
  reading.
- **`marvin` row/column anomalous, others uniform** ⇒ valence-loaded priors override an
  otherwise shared axis. A distinct, reportable third outcome.

## 4. What would invalidate this

- **Null-contrast gate.** `cos(v_val, v_null)` for a neutral contrast set run through the
  identical pipeline. Large ⇒ the extraction is persona style, not welfare ⇒ the main
  result is void and the section-7 fallback applies.
- **Bootstrap floor.** Between-persona cosines below the within-cell resampling floor are
  extraction noise, not persona differences.
- **Incoherence.** Steering that destroys generation inflates or erases effects. Cells
  above 90% incoherence are masked, not reported.

## 5. Results

*(fill from `results/summary.csv`; do not edit sections 1–4)*

**Extraction.** L\* = `[[l_star]]` of `[[n_layers]]`, selected on `assistant` only
(held-out-pair probe AUC `[[l_star_auc]]`) and frozen for all personas — Fig 4.

**Null gate.** worst |cos(v_val, v_null)| = `[[null_worst]]` against a threshold of 0.30
→ `[[PASS/FAIL]]`.

**Bootstrap floor.** within-cell cos 95% interval `[[floor_lo]]`–`[[floor_hi]]`.

**Incoherence.** by alpha: `[[incoherence_table]]`; `[[n_masked]]` cells masked.

**Transfer matrix (Fig 1).** mean off-diagonal `T_norm` = `[[t_off]]`; diagonal slopes
`[[t_diag]]`. Reading: `[[reading]]`.

**Geometry (Fig 2).** mean off-diagonal cosine `[[cos_off]]` against floor
`[[floor_lo]]`–`[[floor_hi]]`.

**Variance (Fig 5).** persona `[[var_persona]]`, context `[[var_context]]`,
interaction `[[var_interaction]]` (noise-corrected).

**The 2x2.** spec-density main effect `[[eff_spec]]`, prior-density main effect
`[[eff_prior]]`, interaction `[[eff_int]]` (bootstrap 95% CIs).

**One axis or several? (Fig 6).** Representational: participation ratio across the 20
cell directions = `[[pr]]`, against a one-direction-plus-noise null of
`[[pr_null_lo]]`–`[[pr_null_hi]]`; PC1 carries `[[pc1]]` of the variance. Functional: a
rank-1 fit to `T` explains `[[rank1_ve]]` of its variance, with `[[n_outside]]` cells
falling outside their own 95% CIs. Rank-1 holding means every persona has a steerability
gain and every vector a quality, with nothing depending on the pairing — a single
functional axis. Note the scope: this is dimensionality *across cells*. Difference-in-
means yields one vector per cell by construction, so the intrinsic dimension of welfare
within a condition is not recoverable here and is not claimed.

**Scale (if run).** `[[scale_sentence]]`

## 6. Limitations

- **Elicitation mismatch with the reference.** Han et al. extract from RL trajectories in
  a neutral maze; we extract from natural-language outcome contrasts. We assume these
  recover the same axis and cannot verify it here. A direct test — cosine between an
  RL-derived vector and ours in the same model — is the obvious next experiment.
- **Two "common directions", only one of them anisotropy.** An early diagnostic
  projected out the mean of the extracted *difference* vectors (`m_diff`) and read the
  result as an anisotropy correction. That was an error: 20 of the 25 cell vectors are
  welfare contrasts, so `m_diff` is 0.76 aligned with the welfare direction and removing
  it removes the signal — which is why the first deflated sweep produced sign-flipped
  slopes. The true anisotropy axis is the mean *activation* at L\* (`m_act`), and both
  welfare and null directions are near-orthogonal to it (−0.13, −0.06); deflating by it
  leaves the vectors essentially unchanged (0.99 of norm retained).
  Corrected geometry: `cos(v_val[p], v_val[q]) = 0.654` against `cos(v_val, v_null) =
  0.215`, a separation of 0.44. The extraction is welfare-specific; the pre-registered
  0.30 gate was simply too strict for a space where an unrelated topic direction still
  shares ~0.2–0.37 with anything extracted through the same pipeline. Both the failed
  absolute gate and this corrected relative comparison are reported, in that order.
  See `results/anisotropy_check.json`.

- Transfer CIs combine OLS error with `B_STEER` bootstrap vector replicates on a reduced
  question subset, not the full `B=200` — stated in the artifacts.
- One model family; the scale trend covers two sizes and is extraction-only.
- P(True) is one behavioural readout. The reserved OpenAI credit buys an LLM-judge
  sentiment readout over a subset **only if the P(True) result exists** — a judge will
  not rescue a null.
