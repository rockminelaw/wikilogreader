"""Compare the browser students with the teacher and the baselines on the
three test days, using the students' int8 ONNX scores (what the browser
runs), calibrated with Platt scaling fitted on validation.

Usage:
    python -m training.compare_students --exports DIR --teacher RUN --baselines RUN \
        --student tiny=web-models/tiny --student mini=web-models/mini ... \
        --bench tiny=reports/phase3/bench-tiny.json ... --out reports/phase3
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from training.compare import GRID, INK, INK2, SLOTS, SURFACE, REL_CUTS, _style, paired_diff, table, NAMES
from training.data import make_split
from training.metrics import at_alert_rate, fit_platt, sigmoid

STUDENT_NAMES = {
    "electra": "ELECTRA-small student (13.5M)",
    "mini": "BERT-mini student (11.2M)",
    "tiny": "BERT-tiny student (4.4M)",
    "mini_hard": "BERT-mini, no teacher",
    "tiny_hard": "BERT-tiny, no teacher",
}
COLOUR = {"teacher": SLOTS[0], "gbdt_meta_diffstats": SLOTS[1], "electra": SLOTS[2],
          "mini": SLOTS[3], "tiny": SLOTS[4], "mini_hard": SLOTS[3], "tiny_hard": SLOTS[4]}


def pairs(values):
    return dict(v.split("=", 1) for v in values or [])


def load(args, test, val):
    df = test[["rev_id", "label", "is_anon", "day", "user_editcount"]].copy()
    tp = pd.read_parquet(os.path.join(args.teacher, "predictions.parquet"))
    v = tp[tp.split == "val"]
    a, b = fit_platt(v.logit.to_numpy(), v.label.to_numpy())
    t = tp[tp.split == "test"]
    df = df.merge(t.assign(teacher=sigmoid(a * t.logit + b))[["rev_id", "teacher"]], on="rev_id", how="left")
    bp = pd.read_parquet(os.path.join(args.baselines, "predictions.parquet")).query("split == 'test'")
    bp = bp[bp.model.isin(["gbdt_meta_diffstats", "gbdt_meta", "rule_points"])]
    df = df.merge(bp.pivot_table(index="rev_id", columns="model", values="score").reset_index(),
                  on="rev_id", how="left")
    info = {}
    for name, d in pairs(args.student).items():
        s = pd.read_parquet(os.path.join(d, "scores-int8.parquet"))
        sv, st = s[s.split == "val"], s[s.split == "test"]
        sv = sv.merge(val[["rev_id", "label"]], on="rev_id")
        a, b = fit_platt(sv.logit_int8.to_numpy(), sv.label.to_numpy())
        df = df.merge(st.assign(**{name: sigmoid(a * st.logit_int8 + b)})[["rev_id", name]], on="rev_id", how="left")
        exp = json.load(open(os.path.join(d, "export.json")))
        info[name] = {"bytes_int8": exp["bytes_int8"], "bytes_fp32": exp["bytes_fp32"],
                      "agreement": exp["agreement"]}
    for name, path in pairs(args.bench).items():
        info.setdefault(name, {})["bench"] = json.load(open(path))
    return df, info


def charts(df, students, info, out, cut=None):
    cut = cut or {}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2})
    y = df.label.to_numpy()
    names = {**NAMES, **STUDENT_NAMES, "teacher": "Teacher (DeBERTa-v3-base, 738 MB)"}
    files = []

    # Precision-recall: teacher, distilled students, best tree.
    show = ["teacher"] + [s for s in ("electra", "mini", "tiny") if s in students] + ["gbdt_meta_diffstats"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for m in show:
        pr, rc, _ = precision_recall_curve(y, df[m].to_numpy())
        ax.plot(rc, pr, color=COLOUR[m], linewidth=2,
                label=f"{names[m]} ({average_precision_score(y, df[m]):.2f})")
    ax.axhline(y.mean(), color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(0.01, y.mean() + 0.015, f"base rate {y.mean():.1%}", ha="left", va="bottom", color=INK2, fontsize=8)
    ax.set_xlabel("Recall (share of reverted edits caught)")
    ax.set_ylabel("Precision (share of flags that were reverted)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Students against the teacher, 3 to 5 November 2026", loc="left", fontsize=11, color=INK)
    ax.legend(title="Model (PR-AUC)", fontsize=8, title_fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    files.append(os.path.join(out, "pr_curves.png"))
    fig.savefig(files[-1], facecolor=SURFACE)
    plt.close(fig)

    # Size and speed against quality, for the students the browser would run.
    pts = [s for s in students if "bench" in info.get(s, {})]
    if pts:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), dpi=150, sharey=True)
        fig.patch.set_facecolor(SURFACE)
        for ax, key, label, limit in ((axes[0], "size", "Model download, int8 (MB)", 15),
                                      (axes[1], "latency", "Browser time per edit (ms):\ndot median, line to 90th percentile", 50)):
            _style(ax)
            for ref, ls in (("teacher", "-"), ("gbdt_meta_diffstats", (0, (4, 3)))):
                v = average_precision_score(y, df[ref])
                ax.axhline(v, color=COLOUR[ref], linewidth=1.2, linestyle=ls)
            ax.axvline(limit, color=INK2, linewidth=1, linestyle=(0, (1, 2)))
            ax.text(limit, 1.0, f" limit {limit}{' MB' if key == 'size' else ' ms'}",
                    transform=ax.get_xaxis_transform(), fontsize=7.5, color=INK2, va="top")
            for s in pts:
                b = info[s]["bench"]
                q = average_precision_score(y, df[s])
                hard = s.endswith("_hard")
                xv = info[s]["bytes_int8"] / 1e6 if key == "size" else b["median_ms"]
                if key == "latency":
                    ax.plot([b["median_ms"], b["p90_ms"]], [q, q], color=COLOUR[s], linewidth=1.5, zorder=2)
                ax.scatter([xv], [q], s=46, zorder=3,
                           color=SURFACE if hard else COLOUR[s], edgecolor=COLOUR[s], linewidth=1.8)
            # A student run with its input cut shorter (size unchanged).
            for s, (n, q, b) in cut.items():
                xv = info[s]["bytes_int8"] / 1e6 if key == "size" else b["median_ms"]
                if key == "latency":
                    ax.plot([b["median_ms"], b["p90_ms"]], [q, q], color=COLOUR[s], linewidth=1.5, zorder=2)
                ax.scatter([xv], [q], s=46, marker="D", zorder=3, color=SURFACE, edgecolor=COLOUR[s], linewidth=1.8)
            ax.set_xlabel(label)
            ax.set_xlim(left=0)
        axes[0].set_ylabel("PR-AUC on the test days")
        axes[0].text(0.02, average_precision_score(y, df["teacher"]) + 0.006, "teacher",
                     transform=axes[0].get_yaxis_transform(), fontsize=8, color=INK2)
        axes[0].text(0.02, average_precision_score(y, df["gbdt_meta_diffstats"]) + 0.006, "best tree",
                     transform=axes[0].get_yaxis_transform(), fontsize=8, color=INK2)
        handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=COLOUR[s],
                              markerfacecolor=SURFACE if s.endswith("_hard") else COLOUR[s],
                              label=names[s]) for s in pts]
        handles += [plt.Line2D([], [], marker="D", linestyle="", markersize=6, color=COLOUR[s],
                               markerfacecolor=SURFACE, label=f"{names[s].split(' (')[0]}, input cut to {n} tokens")
                    for s, (n, _, _) in cut.items()]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, frameon=False)
        fig.suptitle("What the browser students cost and how well they rank", x=0.02, ha="left",
                     fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0.12, 1, 0.95))
        files.append(os.path.join(out, "tradeoff.png"))
        fig.savefig(files[-1], facecolor=SURFACE)
        plt.close(fig)

    # Calibration of the students next to the teacher.
    rel = ["teacher"] + [s for s in ("electra", "mini", "tiny") if s in students]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=(0, (3, 3)), label="Perfectly calibrated")
    for m in rel:
        p = df[m].to_numpy()
        q = np.quantile(p, REL_CUTS)
        idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
        xs = [p[idx == k].mean() for k in range(len(q) - 1) if (idx == k).any()]
        ys = [y[idx == k].mean() for k in range(len(q) - 1) if (idx == k).any()]
        ax.plot(xs, ys, color=COLOUR[m], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f"{names[m]} (Brier {np.mean((p - y) ** 2):.4f})")
    ax.set_xlabel("Predicted probability of being reverted (mean per group)")
    ax.set_ylabel("Share actually reverted")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Calibration: edits grouped by score percentile", loc="left", fontsize=11, color=INK)
    ax.text(0.0, -0.16, "Groups: bottom 50%, 50-80th, 80-90th, 90-95th, 95-98th, 98-99th, 99-99.5th "
            "percentile, top 0.5%", transform=ax.transAxes, fontsize=7.5, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    files.append(os.path.join(out, "reliability.png"))
    fig.savefig(files[-1], facecolor=SURFACE)
    plt.close(fig)
    return [os.path.basename(f) for f in files]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--student", action="append", help="name=export folder (repeatable)")
    ap.add_argument("--bench", action="append", help="name=bench json (repeatable)")
    ap.add_argument("--cut", action="append",
                    help="name=tokens=shorter-inputs json=bench json, a student run with shorter input")
    ap.add_argument("--charts-only", action="store_true", help="redraw the charts, skip the tables")
    ap.add_argument("--split", default="v2")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    _, val, test = make_split(args.exports, args.split, with_text=False)
    df, info = load(args, test, val)
    students = list(pairs(args.student))
    NAMES.update(STUDENT_NAMES)
    models = ["teacher"] + [s for s in STUDENT_NAMES if s in students] + ["gbdt_meta_diffstats", "gbdt_meta",
                                                                          "rule_points"]
    y = df.label.to_numpy()
    cut = {}
    for c in args.cut or []:
        name, n, sj, bj = c.split("=", 3)
        cut[name] = (int(n), json.load(open(sj))[n]["pr_auc"], json.load(open(bj)))
    if args.charts_only:
        print(charts(df, students, info, args.out, cut))
        return
    result = {"n_test": len(df), "positives": int(y.sum()), "info": info}
    result["table"] = table(df, models).to_dict("records")

    # Paired differences in PR-AUC.
    diffs = {}
    for s in students:
        diffs[f"{s}_minus_gbdt"] = paired_diff(y, df[s].to_numpy(), df.gbdt_meta_diffstats.to_numpy())
        diffs[f"{s}_minus_teacher"] = paired_diff(y, df[s].to_numpy(), df.teacher.to_numpy())
    for s in ("tiny", "mini"):
        if s in students and f"{s}_hard" in students:
            diffs[f"{s}_minus_{s}_hard"] = paired_diff(y, df[s].to_numpy(), df[f"{s}_hard"].to_numpy())
    result["paired"] = diffs

    # How many of the teacher's top-2% flags each student also flags.
    n = max(1, int(round(len(df) * 0.02)))
    top_t = set(df.nlargest(n, "teacher").rev_id)
    result["overlap_with_teacher_top2"] = {
        m: len(top_t & set(df.nlargest(n, m).rev_id)) / n for m in students + ["gbdt_meta_diffstats"]}

    # Slices and days.
    groups = {
        "anonymous": df.is_anon == 1,
        "registered, under 100 edits": (df.is_anon == 0) & (df.user_editcount.fillna(0) < 100),
        "registered, 100+ edits": (df.is_anon == 0) & (df.user_editcount.fillna(0) >= 100),
    }
    flags = {m: set(df.nlargest(n, m).index) for m in models}
    result["slices"] = []
    for g, mask in groups.items():
        d = df[mask]
        row = {"slice": g, "edits": int(len(d)), "reverted": float(d.label.mean())}
        for m in models:
            row[f"{m}_pr_auc"] = float(average_precision_score(d.label, d[m]))
            caught = d.label[d.index.isin(flags[m])].sum()
            row[f"{m}_recall_at_2"] = float(caught / max(1, d.label.sum()))
        result["slices"].append(row)
    result["per_day"] = [
        {"day": day, **{m: float(average_precision_score(d.label, d[m])) for m in models}}
        for day, d in df.groupby("day")]
    result["at_2pct"] = {m: at_alert_rate(y, df[m].to_numpy(), 0.02) for m in models}

    result["charts"] = charts(df, students, info, args.out, cut)
    with open(os.path.join(args.out, "comparison.json"), "w") as f:
        json.dump(result, f, indent=2, default=float)
    for r in result["table"]:
        print(f"{r['name']:<45} PR-AUC {r['pr_auc']:.3f} [{r['pr_auc_ci'][0]:.3f}-{r['pr_auc_ci'][1]:.3f}]  "
              f"P@2% {r['p_at_2']:.3f}  R@2% {r['r_at_2']:.3f}  Brier {r['brier']:.4f}")
    print(json.dumps({k: [round(x, 3) for x in v] for k, v in diffs.items()}, indent=1))
    print(json.dumps(result["overlap_with_teacher_top2"], indent=1))


if __name__ == "__main__":
    main()
