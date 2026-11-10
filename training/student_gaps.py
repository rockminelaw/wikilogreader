"""Where the browser students fall behind the teacher: PR-AUC by input
length, by how much of the text the students' vocabulary can't read, and by
editor type, on the three test days.

PR-AUC only depends on the order of the scores, so raw logits are used.

Usage:
    python -m training.student_gaps --splits runs/cache/splits-v2-maxchars1400-student \
        --teacher runs/teacher-v2-256 --baselines runs/baselines-v2 --exports DIR \
        --student tiny=web-models/tiny --student mini=web-models/mini --out reports/phase3/gaps.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from training.data import make_split


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--baselines", required=True)
    ap.add_argument("--exports", required=True)
    ap.add_argument("--student", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    students = dict(s.split("=", 1) for s in args.student)

    from transformers import AutoTokenizer

    df = pd.read_parquet(os.path.join(args.splits, "test.parquet"), columns=["rev_id", "label", "text"])
    _, _, test = make_split(args.exports, "v2", with_text=False)
    df = df.merge(test[["rev_id", "is_anon", "user_editcount"]], on="rev_id")
    t = pd.read_parquet(os.path.join(args.teacher, "predictions.parquet")).query("split == 'test'")
    df = df.merge(t[["rev_id", "logit"]].rename(columns={"logit": "teacher"}), on="rev_id")
    b = pd.read_parquet(os.path.join(args.baselines, "predictions.parquet"))
    b = b[(b.split == "test") & (b.model == "gbdt_meta_diffstats")]
    df = df.merge(b[["rev_id", "score"]].rename(columns={"score": "gbdt"}), on="rev_id")
    for name, d in students.items():
        s = pd.read_parquet(os.path.join(d, "scores-int8.parquet")).query("split == 'test'")
        df = df.merge(s[["rev_id", "logit_int8"]].rename(columns={"logit_int8": name}), on="rev_id")

    # Token counts and [UNK] share with the students' tokeniser (all the
    # students here share BERT's uncased vocabulary).
    tok = AutoTokenizer.from_pretrained(os.path.join(next(iter(students.values())), "hf-tokenizer"))
    n_tok, n_unk = [], []
    texts = df.text.tolist()
    for i in range(0, len(texts), 5000):
        for ids in tok(texts[i:i + 5000], add_special_tokens=False)["input_ids"]:
            n_tok.append(len(ids))
            n_unk.append(sum(1 for x in ids if x == tok.unk_token_id))
    df["tokens"] = np.array(n_tok)
    df["unk_share"] = np.array(n_unk) / np.maximum(1, df.tokens)

    models = ["teacher", *students, "gbdt"]
    groups = {
        "tokens": [("up to 64", df.tokens <= 64), ("65-128", (df.tokens > 64) & (df.tokens <= 128)),
                   ("129-254", (df.tokens > 128) & (df.tokens <= 254)), ("cut at 256", df.tokens > 254)],
        "unreadable": [("no [UNK] tokens", df.unk_share == 0),
                       ("under 5% [UNK]", (df.unk_share > 0) & (df.unk_share < 0.05)),
                       ("5% or more [UNK]", df.unk_share >= 0.05)],
        "editor": [("anonymous", df.is_anon == 1),
                   ("registered, under 100 edits", (df.is_anon == 0) & (df.user_editcount.fillna(0) < 100)),
                   ("registered, 100+ edits", (df.is_anon == 0) & (df.user_editcount.fillna(0) >= 100))],
    }
    result = {"n": len(df)}
    for key, rows in groups.items():
        result[key] = []
        for label, mask in rows:
            d = df[mask]
            r = {"group": label, "edits": int(len(d)), "share_of_edits": float(mask.mean()),
                 "reverted": float(d.label.mean())}
            for m in models:
                r[m] = float(average_precision_score(d.label, d[m]))
            result[key].append(r)
            print(key, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}))

    # Reverted edits in the teacher's top 2% that a student leaves out of its
    # own top 2%: how they differ from the ones it keeps.
    n = int(round(len(df) * 0.02))
    top_t = df.nlargest(n, "teacher")
    hits = top_t[top_t.label == 1]
    result["teacher_top2_hits"] = int(len(hits))
    for name in students:
        mine = set(df.nlargest(n, name).rev_id)
        kept, lost = hits[hits.rev_id.isin(mine)], hits[~hits.rev_id.isin(mine)]
        result[f"{name}_missed_teacher_hits"] = {
            "missed": int(len(lost)), "share_missed": float(len(lost) / max(1, len(hits))),
            "missed_anonymous": float(lost.is_anon.mean()), "kept_anonymous": float(kept.is_anon.mean()),
            "missed_median_tokens": float(lost.tokens.median()), "kept_median_tokens": float(kept.tokens.median()),
            "missed_with_unk": float((lost.unk_share > 0).mean()), "kept_with_unk": float((kept.unk_share > 0).mean()),
        }
        print(name, json.dumps(result[f"{name}_missed_teacher_hits"]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
