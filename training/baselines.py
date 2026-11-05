"""Baselines on the same split as the teacher.

Rules (no training, apart from turning their scores into probabilities):
  anon_big_removal   anonymous editor and the page shrank by 500 bytes or more
  profanity          added text contains a word from a short profanity list
                     that the removed text doesn't
  rule_points        a points score combining these with a few other signs
                     (blanking most of a page, shouting in capitals, no edit
                     summary from a new account)

Rules give few distinct scores, so their probabilities come from the
validation day: each score level gets the revert rate seen at that level.

Gradient-boosted trees (LightGBM), trained on the train days with early
stopping on the validation day:
  gbdt_meta          metadata only: editor, sizes, summary, tags, time of day
  gbdt_meta_diffstats metadata plus cheap statistics of the diff text

Usage:
    python -m training.baselines --exports DIR --out DIR
Writes predictions.parquet (rev_id, split, model, score) and metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

from training.data import make_split
from training.metrics import evaluate, summary_line

# Deliberately short and mild: the point is to show what a word list catches,
# not to build a good one.
PROFANITY = [
    "fuck", "fucking", "fucker", "shit", "shitty", "bitch", "bastard", "crap", "damn",
    "dick", "cock", "penis", "vagina", "boobs", "tits", "ass", "asshole", "arse",
    "butt", "poop", "pee", "fart", "sex", "sexy", "porn", "gay", "suck", "sucks",
    "stupid", "idiot", "dumb", "loser", "lol", "lmao", "haha", "hahaha", "hehe",
    "poo", "wanker", "twat", "piss", "pissed", "nerd", "ugly", "fat", "smelly",
]
REPEAT_RE = re.compile(r"(.)\1{4,}")
PROF_RE = re.compile(r"\b(" + "|".join(map(re.escape, PROFANITY)) + r")\b", re.IGNORECASE)


def _text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def profanity_added(df: pd.DataFrame) -> np.ndarray:
    added = _text(df.added_text).map(lambda x: len(PROF_RE.findall(x)))
    removed = _text(df.removed_text).map(lambda x: len(PROF_RE.findall(x)))
    return (added > removed).to_numpy().astype(int)


def rule_scores(df: pd.DataFrame) -> pd.DataFrame:
    delta = df.size_delta.fillna(0)
    old = df.size_old.fillna(0).clip(lower=1)
    added = _text(df.added_text)
    letters = added.str.count(r"[A-Za-z]")
    caps = added.str.count(r"[A-Z]")
    shouting = ((letters >= 20) & (caps / letters.clip(lower=1) > 0.7)).astype(int)
    blanking = ((delta < 0) & (df.size_new.fillna(0) < 0.1 * old) & (old > 1000)).astype(int)
    new_account = (df.user_editcount.fillna(0) < 10).astype(int)
    no_summary = _text(df.comment).str.strip().eq("").astype(int)
    anon = df.is_anon.astype(int)
    big_removal = (delta <= -500).astype(int)
    prof = profanity_added(df)
    out = pd.DataFrame({
        "anon_big_removal": (anon & big_removal).astype(float),
        "profanity": prof.astype(float),
        "rule_points": (
            2 * prof + 2 * blanking + 1.5 * (anon & big_removal) + shouting
            + anon + 0.5 * (new_account & no_summary)
        ).astype(float),
    })
    return out


def calibrate_levels(val_score: np.ndarray, val_y: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Map each distinct score to the validation revert rate at that score."""
    rates = pd.Series(val_y).groupby(val_score).mean()
    base = float(val_y.mean())
    return pd.Series(score).map(rates).fillna(base).to_numpy()


def gbdt_features(df: pd.DataFrame, tags: list[str], with_diff: bool) -> pd.DataFrame:
    comment = _text(df.comment)
    X = pd.DataFrame({
        "anon": df.is_anon.astype(int),
        "temp": df.is_temp.astype(int),
        "minor": df.minor.astype(int),
        "size_delta": df.size_delta,
        "size_old": df.size_old,
        "log_editcount": np.log1p(df.user_editcount.fillna(0)),
        "editcount_known": df.user_editcount.notna().astype(int),
        "account_age_days": (df.ts - df.user_registration) / 86400,
        "comment_len": comment.str.len(),
        "no_comment": comment.str.strip().eq("").astype(int),
        "section_comment": comment.str.startswith("/*").astype(int),
        "hour_utc": (df.ts % 86400) // 3600,
        "weekday": ((df.ts // 86400) + 4) % 7,
    })
    groups = _text(df.user_groups)
    for g in ("autoconfirmed", "extendedconfirmed", "sysop", "rollbacker", "patroller",
              "reviewer", "extendedmover"):
        X[f"group_{g}"] = groups.str.contains(rf"\b{g}\b").astype(int)
    tag_lists = _text(df.tags_at_save).str.split(",")
    for t in tags:
        X["tag_" + re.sub(r"\W+", "_", t)] = tag_lists.apply(lambda xs, t=t: t in xs).astype(int)
    if with_diff:
        added, removed = _text(df.added_text), _text(df.removed_text)
        letters = added.str.count(r"[A-Za-z]").clip(lower=1)
        X["lines_added"] = df.lines_added
        X["lines_removed"] = df.lines_removed
        X["lines_changed"] = df.lines_changed
        X["added_chars"] = added.str.len()
        X["removed_chars"] = removed.str.len()
        X["added_caps_ratio"] = added.str.count(r"[A-Z]") / letters
        X["added_digits"] = added.str.count(r"\d")
        X["added_links"] = added.str.count(r"\[\[")
        X["added_refs"] = added.str.count(r"<ref")
        X["added_templates"] = added.str.count(r"\{\{")
        X["added_urls"] = added.str.count(r"https?://")
        X["added_exclaims"] = added.str.count("!")
        X["repeated_chars"] = added.map(lambda x: len(REPEAT_RE.findall(x)))
        X["profanity_added"] = profanity_added(df)
        X["removed_refs"] = removed.str.count(r"<ref")
    return X


def train_gbdt(tr, va, te, with_diff: bool, seed: int = 13):
    import lightgbm as lgb

    tags = [t for t in _text(tr.tags_at_save).str.split(",").explode().value_counts().index[:40] if t]
    Xtr, Xva, Xte = (gbdt_features(d, tags, with_diff) for d in (tr, va, te))
    model = lgb.LGBMClassifier(
        n_estimators=3000, learning_rate=0.03, num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=seed, verbose=-1,
    )
    model.fit(Xtr, tr.label, eval_set=[(Xva, va.label)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(150, verbose=False)])
    t0 = time.perf_counter()
    p_te = model.predict_proba(Xte)[:, 1]
    per_edit_ms = (time.perf_counter() - t0) * 1000 / len(Xte)
    one = Xte.iloc[[0]]
    t0 = time.perf_counter()
    for _ in range(200):
        model.predict_proba(one)
    single_ms = (time.perf_counter() - t0) * 1000 / 200
    imp = pd.Series(model.booster_.feature_importance("gain"), Xtr.columns)
    return {
        "val": model.predict_proba(Xva)[:, 1], "test": p_te,
        "trees": int(model.best_iteration_ or model.n_estimators),
        "latency_ms": {"batched_per_edit": per_edit_ms, "single_edit": single_ms},
        "importance": (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    tr, va, te = make_split(args.exports)
    print(f"train {len(tr)}, val {len(va)}, test {len(te)}", flush=True)
    preds, metrics = [], {}

    rv, rt = rule_scores(va), rule_scores(te)
    for name in rv.columns:
        p = calibrate_levels(rv[name].to_numpy(), va.label.to_numpy(), rt[name].to_numpy())
        # Break ties between equal rule scores by the size of the removal, so
        # "top 2%" is well defined; probabilities are unchanged.
        rank = p + 1e-9 * (-te.size_delta.fillna(0).clip(upper=0)).to_numpy() / 1e6
        m = evaluate(te.label, p)
        flagged = rt[name].to_numpy() > 0
        m["as_flag"] = {"flag_rate": float(flagged.mean()),
                        "precision": float(te.label[flagged].mean()) if flagged.any() else 0.0,
                        "recall": float(te.label[flagged].sum() / te.label.sum())}
        from training.metrics import at_alert_rate
        for r in (0.02, 0.05):
            m[f"at_{int(r * 100)}pct"] = at_alert_rate(te.label.to_numpy(), rank, r)
        metrics[name] = m
        preds.append(pd.DataFrame({"rev_id": te.rev_id, "split": "test", "model": name, "score": p}))
        print(summary_line(name, m), "| as a flag:", json.dumps(m["as_flag"]), flush=True)

    for name, with_diff in (("gbdt_meta", False), ("gbdt_meta_diffstats", True)):
        res = train_gbdt(tr, va, te, with_diff)
        m = evaluate(te.label, res["test"])
        m.update(trees=res["trees"], latency_ms=res["latency_ms"], importance=res["importance"])
        metrics[name] = m
        preds.append(pd.DataFrame({"rev_id": te.rev_id, "split": "test", "model": name, "score": res["test"]}))
        preds.append(pd.DataFrame({"rev_id": va.rev_id, "split": "val", "model": name, "score": res["val"]}))
        print(summary_line(name, m), f"| {res['trees']} trees", flush=True)

    pd.concat(preds).to_parquet(os.path.join(args.out, "predictions.parquet"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
