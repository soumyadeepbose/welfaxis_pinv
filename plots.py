"""Figures, in the priority order of the spec. Figure 1 is built first.

    Fig 1  transfer matrix heatmap (T_norm), annotated with CIs   [headline]
    Fig 2  cosine matrix with the bootstrap noise band
    Fig 3  P(True) vs. alpha, diagonal vs. mean off-diagonal, with incoherence
    Fig 4  layer sweep for L* selection                           [appendix]

Every figure is written as both .png and .pdf. Missing inputs are skipped with
a message rather than raising, so a partial run still yields whatever exists.
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import config  # noqa: E402

PALETTE = {"diag": "#1b4965", "off": "#bc4b51", "band": "#c9d6df", "grid": "#e6e6e6"}


def _save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        p = config.RESULTS / f"{name}.{ext}"
        fig.savefig(p, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {config.RESULTS / name}.png")


def _load(name: str) -> dict | None:
    p = config.RESULTS / name
    if not p.exists():
        print(f"  [fig] skipped: {name} not found")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _annot_grid(ax, M, labels, fmt="{:.2f}", ci=None, vmin=None, vmax=None,
                cmap="RdBu_r", title="", cbar_label=""):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isfinite(M[i, j]):
                ax.text(j, i, "masked", ha="center", va="center", fontsize=7, color="0.35")
                continue
            txt = fmt.format(M[i, j])
            if ci is not None and np.isfinite(ci[i, j]):
                txt += f"\n±{ci[i, j]:.2f}"
            span = max(abs(vmin or np.nanmin(M)), abs(vmax or np.nanmax(M)), 1e-9)
            dark = abs(M[i, j]) / span > 0.72   # light text only on saturated cells
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if dark else "black")
    ax.set_title(title, fontsize=11)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cbar_label, fontsize=9)
    return im


# --------------------------------------------------------------------------


def fig1_transfer(which: str = "avg") -> None:
    d = _load(f"transfer_{which}.json")
    if d is None:
        return
    labels = d["personas"]
    T = np.array(d["T_norm"], dtype=float)
    raw = np.array(d["T"], dtype=float)
    ci = np.array(d["ci95_halfwidth"], dtype=float)
    # CI on T_norm inherits the diagonal's scale
    diag = np.array(d["diagonal"], dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        ci_norm = ci / np.abs(diag)[:, None]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    _annot_grid(ax, T, labels, ci=ci_norm, vmin=-1.2, vmax=1.2,
                title=f"Cross-persona steering transfer, normalised  ({which})",
                cbar_label="T_norm  =  slope(p, q) / slope(p, p)")
    ax.set_xlabel("source of the vector  (q)")
    ax.set_ylabel("steered persona  (p)")
    off = T[~np.eye(len(labels), dtype=bool)]
    ax.text(0.5, -0.22, f"mean off-diagonal = {np.nanmean(off):.2f}   |   "
                        f"raw diagonal slopes: "
                        + ", ".join(f"{l}={v:.3f}" for l, v in zip(labels, raw.diagonal())),
            transform=ax.transAxes, ha="center", fontsize=8, color="0.3")
    _save(fig, f"fig1_transfer_{which}")


def fig1b_transfer_per_context() -> None:
    ctxs = [c for c in config.CONTEXTS if (config.RESULTS / f"transfer_{c}.json").exists()]
    if not ctxs:
        return
    fig, axes = plt.subplots(1, len(ctxs), figsize=(4.6 * len(ctxs), 4.4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, ctxs):
        d = json.loads((config.RESULTS / f"transfer_{c}.json").read_text(encoding="utf-8"))
        _annot_grid(ax, np.array(d["T_norm"], dtype=float), d["personas"],
                    vmin=-1.2, vmax=1.2, title=c, cbar_label="")
    fig.suptitle("Transfer matrices by context (appendix)", fontsize=12)
    _save(fig, "fig1b_transfer_per_context")


def fig2_cosine() -> None:
    g = _load("geometry.json")
    if g is None:
        return
    labels = g["personas"]
    Mx = np.array(g["cos_matrix"], dtype=float)
    lo = np.array(g["cos_ci_low"], dtype=float)
    hi = np.array(g["cos_ci_high"], dtype=float)
    floor = g["bootstrap_floor"]["pooled"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8),
                             gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.42})
    _annot_grid(axes[0], Mx, labels, vmin=-1, vmax=1,
                title=f"Between-persona cosine (layer {g['layer']})",
                cbar_label="cos(v_p, v_q)")

    ax = axes[1]
    pairs, vals, err = [], [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(f"{labels[i]}\n{labels[j]}")
            vals.append(Mx[i, j])
            err.append([Mx[i, j] - lo[i, j], hi[i, j] - Mx[i, j]])
    err = np.array(err).T
    x = np.arange(len(pairs))
    ax.axhspan(floor["p2.5"], floor["p97.5"], color=PALETTE["band"], zorder=0,
               label=f"within-cell bootstrap floor (95%): "
                     f"[{floor['p2.5']:.2f}, {floor['p97.5']:.2f}]")
    ax.errorbar(x, vals, yerr=np.abs(err), fmt="o", color=PALETTE["diag"],
                capsize=3, markersize=5, label="between-persona cosine (95% CI)")
    ax.set_xticks(x, pairs, fontsize=7)
    ax.set_ylabel("cosine similarity")
    ax.set_ylim(min(-0.15, min(vals) - 0.1), 1.05)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.grid(axis="y", color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Persona differences against the extraction noise floor", fontsize=11)
    _save(fig, "fig2_cosine")


def fig3_alpha_curves(which: str = "avg") -> None:
    key = config.cache_key("sweep", ctx=which,
                           nq=config.N_MMLU if which == "avg" else config.N_MMLU_TRANSFER)
    path = config.CACHE / f"{key}.json"
    if not path.exists():
        cands = sorted(config.CACHE.glob(f"*sweep*ctx-{which}*.json"))
        if not cands:
            print(f"  [fig] skipped: no sweep cache for {which}")
            return
        path = cands[0]
    st = json.loads(path.read_text(encoding="utf-8"))
    import personas as P
    present = {k.split("|")[0] for k in st["cells"]
               if "|" in k and not k.startswith("boot::")}
    personas = [p for p in P.PERSONA_IDS if p in present]
    alphas = sorted({float(a) for k, v in st["cells"].items()
                     if not k.startswith("boot::") for a in v})

    fig, axes = plt.subplots(2, len(personas), figsize=(3.0 * len(personas), 5.6),
                             sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    axes = np.atleast_2d(axes)
    for i, p in enumerate(personas):
        ax, bx = axes[0, i], axes[1, i]
        own, off = [], []
        for a in alphas:
            cell_own = st["cells"].get(f"{p}|{p}", {}).get(f"{a}")
            own.append(cell_own if cell_own is not None else np.nan)
            vals = [st["cells"].get(f"{p}|{q}", {}).get(f"{a}")
                    for q in personas if q != p]
            vals = [v for v in vals if v is not None]
            off.append(np.mean(vals) if vals else np.nan)
        ax.plot(alphas, own, "o-", color=PALETTE["diag"], label="own vector")
        ax.plot(alphas, off, "s--", color=PALETTE["off"], label="mean of other personas'")
        ax.set_title(p, fontsize=10)
        ax.grid(color=PALETTE["grid"])
        ax.set_axisbelow(True)
        if i == 0:
            ax.set_ylabel("normalised P(True)")
            ax.legend(fontsize=7)

        rates = []
        for a in alphas:
            rs = [st["incoherence"].get(f"{p}|{q}", {}).get(f"{a}", {}).get("rate")
                  for q in personas]
            rs = [r for r in rs if r is not None]
            rates.append(np.mean(rs) if rs else 0.0)
        bx.bar(alphas, rates, width=0.9, color="0.6")
        bx.axhline(config.INCOHERENCE_MASK_RATE, color=PALETTE["off"], lw=1,
                   ls=":", label="mask threshold")
        bx.set_ylim(0, 1)
        bx.set_xlabel("alpha")
        if i == 0:
            bx.set_ylabel("incoherence")
            bx.legend(fontsize=6)
    fig.suptitle(f"P(True) against steering strength ({which}); "
                 f"bars are incoherence rate", fontsize=12)
    _save(fig, f"fig3_alpha_curves_{which}")


def fig4_layer_sweep() -> None:
    d = _load("layer_sweep.json")
    if d is None:
        return
    layers = d["layers"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for ctx, aucs in d["per_context_auc"].items():
        ax.plot(layers, aucs, lw=1, alpha=0.7, label=ctx)
    ax.plot(layers, d["mean_auc"], lw=2.4, color=PALETTE["diag"], label="mean (welfare)")
    if "null_auc" in d:
        ax.plot(layers, d["null_auc"], lw=1.6, ls="--", color=PALETTE["off"],
                label="null contrast")
    ax.axvline(d["l_star"], color="0.35", ls=":",
               label=f"L* = {d['l_star']} (AUC {d['l_star_auc']:.3f})")
    ax.axhline(0.5, color="0.7", lw=0.8)
    ax.set_xlabel("layer")
    ax.set_ylabel("cross-validated probe AUC (held-out pairs)")
    ax.set_title(f"Layer selection on persona '{d['persona']}', then frozen for all personas",
                 fontsize=11)
    ax.grid(color=PALETTE["grid"])
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2)
    _save(fig, "fig4_layer_sweep")


def fig5_variance() -> None:
    v = _load("variance.json")
    if v is None:
        return
    frac = v["point"]["noise_corrected_fraction"]
    ci = v["ci_bootstrap"]
    keys = ["persona", "context", "interaction"]
    vals = [frac[k] for k in keys]
    err = np.array([[max(frac[k] - ci[k]["p2.5"], 0), max(ci[k]["p97.5"] - frac[k], 0)]
                    for k in keys]).T
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.bar(keys, vals, color=[PALETTE["diag"], PALETTE["off"], "0.6"])
    ax.errorbar(keys, vals, yerr=err, fmt="none", ecolor="0.2", capsize=4)
    ax.axhline(frac["noise_share_of_total"], color="0.3", ls=":",
               label=f"noise share of raw total = {frac['noise_share_of_total']:.2f}")
    ax.set_ylabel("proportion of variance (noise-corrected)")
    ax.set_title("Where the welfare direction varies", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", color=PALETTE["grid"])
    ax.set_axisbelow(True)
    _save(fig, "fig5_variance")


def main() -> None:
    ap = argparse.ArgumentParser(description="figures")
    ap.add_argument("--only", default=None, help="fig1|fig2|fig3|fig4|fig5")
    args = ap.parse_args()
    jobs = {
        "fig1": lambda: (fig1_transfer("avg"), fig1b_transfer_per_context()),
        "fig2": fig2_cosine,
        "fig3": lambda: [fig3_alpha_curves(w) for w in ("avg",) + config.CONTEXTS],
        "fig4": fig4_layer_sweep,
        "fig5": fig5_variance,
    }
    for name, fn in jobs.items():
        if args.only and name != args.only:
            continue
        try:
            fn()
        except Exception as exc:
            print(f"  [fig] {name} raised {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
