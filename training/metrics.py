"""The metrics every model is reported with, so results compare like for like.

- PR-AUC (average precision) and ROC-AUC.
- Precision and recall at a fixed alert rate: flag the top 2% (and 5%) of
  edits by score, as the demo would.
- Calibration: Brier score, expected calibration error over 10 equal-width
  bins, and the bins themselves for a reliability plot.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def at_alert_rate(y: np.ndarray, score: np.ndarray, rate: float) -> dict:
    n = max(1, int(round(len(y) * rate)))
    top = np.argsort(-score, kind="stable")[:n]
    hits = int(y[top].sum())
    return {"precision": hits / n, "recall": hits / max(1, int(y.sum())), "flagged": n}


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                        "mean_pred": float(p[m].mean()), "frac_pos": float(y[m].mean())})
    return out


def evaluate(y, p, rates=(0.02, 0.05)) -> dict:
    """y: 0/1 labels. p: predicted probability of being reverted."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    rel = reliability(y, p)
    ece = sum(b["n"] * abs(b["mean_pred"] - b["frac_pos"]) for b in rel) / len(y)
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece),
        "reliability": rel,
    }
    for r in rates:
        out[f"at_{int(r * 100)}pct"] = at_alert_rate(y, p, r)
    return out


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """One-parameter calibration on held-out data: p = sigmoid(logit / T)."""
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y, dtype=float)

    def nll(t):
        z = logits / t
        return float(np.mean(np.logaddexp(0, z) - y * z))

    lo, hi = 0.05, 20.0
    for _ in range(100):  # golden-section search; nll is unimodal in T
        a = hi - (hi - lo) / 1.618
        b = lo + (hi - lo) / 1.618
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return (lo + hi) / 2


def fit_platt(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-parameter calibration on held-out data: p = sigmoid(a * logit + b).
    Unlike a temperature alone, the offset b can correct a model that is
    overconfident across its middle range; unlike isotonic regression it keeps
    the ranking, so PR-AUC is unchanged."""
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(logits, dtype=float)[:, None], y)
    return float(lr.coef_.ravel()[0]), float(lr.intercept_[0])


def sigmoid(z):
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def summary_line(name: str, m: dict) -> str:
    return (f"{name:<28} PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  "
            f"P@2% {m['at_2pct']['precision']:.3f}  R@2% {m['at_2pct']['recall']:.3f}  "
            f"Brier {m['brier']:.4f}  ECE {m['ece']:.4f}")
