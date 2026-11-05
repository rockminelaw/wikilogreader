"""Compare every model on the same test edits: metrics with bootstrap
confidence intervals, a slice by editor type, and the report charts.

Usage:
    python -m training.compare --exports DIR --teacher RUN_DIR --baselines RUN_DIR \
        [--liftwing FILE.jsonl] --out DIR
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from training.data import make_split
from training.metrics import at_alert_rate, evaluate, fit_platt, sigmoid

NAMES = {
    "teacher": "Teacher (DeBERTa-v3-base, Platt-calibrated)",
    "gbdt_meta_diffstats": "GBDT, metadata + diff statistics",
    "gbdt_meta": "GBDT, metadata only",
    "liftwing": "Wikimedia revert-risk (Lift Wing)",
    "rule_points": "Rules, points score",
    "anon_big_removal": "Rule: anonymous + removal of 500+ bytes",
    "profanity": "Rule: profanity added",
}
ORDER = list(NAMES)
REL_CUTS = [0, .5, .8, .9, .95, .98, .99, .995, 1]

# Reference palette (dataviz skill), light mode, fixed slot order.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLOUR = {"teacher": SLOTS[0], "gbdt_meta_diffstats": SLOTS[1], "liftwing": SLOTS[2],
          "gbdt_meta": SLOTS[3], "rule_points": SLOTS[4]}


def load_scores(args, test: pd.DataFrame) -> pd.DataFrame:
    df = test[["rev_id", "label", "is_anon", "is_temp"]].copy()
    # Platt calibration fitted on the validation day (training/metrics.py).
    allp = pd.read_parquet(os.path.join(args.teacher, "predictions.parquet"))
    v = allp[allp.split == "val"]
    a, b = fit_platt(v.logit.to_numpy(), v.label.to_numpy())
    tp = allp[allp.split == "test"]
    tp = tp.assign(teacher=sigmoid(a * tp.logit + b))[["rev_id", "teacher"]]
    df = df.merge(tp, on="rev_id", how="left")
    bp = pd.read_parquet(os.path.join(args.baselines, "predictions.parquet")).query("split == 'test'")
    df = df.merge(bp.pivot_table(index="rev_id", columns="model", values="score").reset_index(),
                  on="rev_id", how="left")
    if args.liftwing and os.path.exists(args.liftwing):
        rows = [json.loads(x) for x in open(args.liftwing) if x.strip()]
        lw = pd.DataFrame(rows)
        lw = lw[lw.status == "ok"][["rev_id", "p"]].rename(columns={"p": "liftwing"})
        df = df.merge(lw.drop_duplicates("rev_id"), on="rev_id", how="left")
    return df


def bootstrap(y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = {"pr_auc": [], "p2": [], "r2": [], "brier": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yy, pp = y[i], p[i]
        out["pr_auc"].append(average_precision_score(yy, pp))
        a = at_alert_rate(yy, pp, 0.02)
        out["p2"].append(a["precision"])
        out["r2"].append(a["recall"])
        out["brier"].append(float(np.mean((pp - yy) ** 2)))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def paired_diff(y, a, b, n=1000, seed=1):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(average_precision_score(y[i], a[i]) - average_precision_score(y[i], b[i]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def table(df: pd.DataFrame, models, boot=True) -> pd.DataFrame:
    y = df.label.to_numpy()
    rows = []
    for m in models:
        p = df[m].to_numpy()
        e = evaluate(y, p)
        row = {"model": m, "name": NAMES[m], "pr_auc": e["pr_auc"], "roc_auc": e["roc_auc"],
               "p_at_2": e["at_2pct"]["precision"], "r_at_2": e["at_2pct"]["recall"],
               "p_at_5": e["at_5pct"]["precision"], "r_at_5": e["at_5pct"]["recall"],
               "brier": e["brier"], "ece": e["ece"]}
        if boot:
            ci = bootstrap(y, p)
            row.update({f"{k}_ci": v for k, v in ci.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def charts(df: pd.DataFrame, models, out: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})
    y = df.label.to_numpy()
    files = []
    show = [m for m in models if m in COLOUR]

    # Precision-recall curves.
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for m in show:
        pr, rc, _ = precision_recall_curve(y, df[m].to_numpy())
        ax.plot(rc, pr, color=COLOUR[m], linewidth=2, label=f"{NAMES[m]} ({average_precision_score(y, df[m]):.2f})")
    ax.axhline(y.mean(), color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.01, y.mean() + 0.015, f"base rate {y.mean():.1%}", ha="left", va="bottom", color=INK2, fontsize=8)
    ax.set_xlabel("Recall (share of reverted edits caught)")
    ax.set_ylabel("Precision (share of flags that were reverted)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision and recall on the test day, 29 October 2026", loc="left", fontsize=11, color=INK)
    ax.legend(title="Model (PR-AUC)", fontsize=8, title_fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    files.append(os.path.join(out, "pr_curves.png"))
    fig.savefig(files[-1], facecolor=SURFACE)
    plt.close(fig)

    # Reliability: groups by score percentile, finer at the top where flags are.
    rel = [m for m in ("teacher", "gbdt_meta_diffstats", "liftwing") if m in models]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=(0, (3, 3)), label="Perfectly calibrated")
    for m in rel:
        p = df[m].to_numpy()
        q = np.quantile(p, REL_CUTS)
        idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
        xs = [p[idx == b].mean() for b in range(len(q) - 1) if (idx == b).any()]
        ys = [y[idx == b].mean() for b in range(len(q) - 1) if (idx == b).any()]
        ax.plot(xs, ys, color=COLOUR[m], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{NAMES[m]} (Brier {np.mean((p - y) ** 2):.4f})")
    ax.set_xlabel("Predicted probability of being reverted (mean per group)")
    ax.set_ylabel("Share actually reverted")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration on the test day: edits grouped by score percentile",
                 loc="left", fontsize=11, color=INK)
    ax.text(0.0, -0.16, "Groups: bottom 50%, 50-80th, 80-90th, 90-95th, 95-98th, 98-99th, 99-99.5th "
            "percentile, top 0.5%", transform=ax.transAxes, fontsize=7.5, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    files.append(os.path.join(out, "reliability.png"))
    fig.savefig(files[-1], facecolor=SURFACE)
    plt.close(fig)
    return files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--liftwing", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    _, _, test = make_split(args.exports, with_text=False)
    df = load_scores(args, test)
    models = [m for m in ORDER if m in df.columns]
    result = {"n_test": len(df), "positives": int(df.label.sum())}

    full = [m for m in models if m != "liftwing"]
    result["full_test"] = table(df, full).to_dict("records")
    if "liftwing" in df:
        both = df[df.liftwing.notna()]
        result["liftwing_coverage"] = {"scored": int(len(both)), "of": int(len(df))}
        result["with_liftwing"] = table(both, models).to_dict("records")
        result["teacher_minus_liftwing_pr_auc"] = paired_diff(
            both.label.to_numpy(), both.teacher.to_numpy(), both.liftwing.to_numpy())
        df_chart = both
    else:
        df_chart = df
    result["teacher_minus_gbdt_pr_auc"] = paired_diff(
        df.label.to_numpy(), df.teacher.to_numpy(), df.gbdt_meta_diffstats.to_numpy())

    # Slices by editor type, with a 2% alert budget chosen over all edits.
    slices = {}
    for m in [x for x in ("teacher", "gbdt_meta_diffstats", "liftwing") if x in df_chart]:
        d = df_chart
        n = max(1, int(round(len(d) * 0.02)))
        top = d.nlargest(n, m)
        slices[m] = {
            "share_of_flags_anonymous": float(top.is_anon.mean()),
            "share_of_edits_anonymous": float(d.is_anon.mean()),
            "share_of_reverts_anonymous": float(d[d.label == 1].is_anon.mean()),
            "recall_anonymous": float(top[top.is_anon == 1].label.sum() / max(1, d[(d.label == 1) & (d.is_anon == 1)].shape[0])),
            "recall_registered": float(top[top.is_anon == 0].label.sum() / max(1, d[(d.label == 1) & (d.is_anon == 0)].shape[0])),
            "precision_anonymous_flags": float(top[top.is_anon == 1].label.mean()) if (top.is_anon == 1).any() else None,
            "precision_registered_flags": float(top[top.is_anon == 0].label.mean()) if (top.is_anon == 0).any() else None,
            "pr_auc_anonymous": float(average_precision_score(d[d.is_anon == 1].label, d[d.is_anon == 1][m])),
            "pr_auc_registered": float(average_precision_score(d[d.is_anon == 0].label, d[d.is_anon == 0][m])),
        }
    result["slices"] = slices
    result["charts"] = charts(df_chart, models, args.out)
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k not in ("full_test", "with_liftwing")}, indent=2))
    for r in result.get("with_liftwing", result["full_test"]):
        print(f"{r['name']:<42} PR-AUC {r['pr_auc']:.3f} [{r['pr_auc_ci'][0]:.3f}-{r['pr_auc_ci'][1]:.3f}]  "
              f"P@2% {r['p_at_2']:.3f}  R@2% {r['r_at_2']:.3f}  Brier {r['brier']:.4f}  ECE {r['ece']:.4f}")


if __name__ == "__main__":
    main()
