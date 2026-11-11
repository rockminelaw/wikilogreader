"""Build the files the website demo loads: the int8 model, its vocabulary
and a settings file with the calibration and the flag threshold, all for one
input length.

The settings file also records how the threshold did on the test days, so
the demo's note can quote measured numbers.

Usage:
    python -m training.web_bundle --model web-models/mini --splits runs/cache/splits-v2-maxchars1400-student \
        --max-len 128 --threshold 0.5 --out web-bundle/mini-128
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from training.export_onnx import score
from training.metrics import fit_platt, sigmoid


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="export folder (model.onnx, vocab.txt, hf-tokenizer/)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--max-len", type=int, required=True)
    ap.add_argument("--threshold", type=float, default=0.5, help="calibrated score at or above which an edit is flagged")
    ap.add_argument("--diff-chars", type=int, default=1400)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(args.model, "hf-tokenizer"))
    os.makedirs(args.out, exist_ok=True)
    scores_path = os.path.join(args.out, f"scores-{args.max_len}.parquet")
    data = {s: pd.read_parquet(os.path.join(args.splits, f"{s}.parquet"), columns=["rev_id", "day", "label", "text"])
            for s in ("val", "test")}
    if os.path.exists(scores_path):
        saved = pd.read_parquet(scores_path)
        logits = {s: saved[saved.split == s].set_index("rev_id").loc[d.rev_id, "logit"].to_numpy() for s, d in data.items()}
    else:
        logits = {s: score(os.path.join(args.model, "model.onnx"), tok, d.text.tolist(), args.max_len)
                  for s, d in data.items()}
        pd.concat([pd.DataFrame({"rev_id": d.rev_id, "split": s, "logit": logits[s]}) for s, d in data.items()]
                  ).to_parquet(scores_path, index=False)

    a, b = fit_platt(logits["val"], data["val"].label.to_numpy())
    y = data["test"].label.to_numpy()
    p = sigmoid(a * logits["test"] + b)
    flag = p >= args.threshold
    days = {}
    for day, d in data["test"].assign(flag=flag).groupby("day"):
        days[day] = {"edits": int(len(d)), "flagged": int(d.flag.sum()),
                     "flagged_share": float(d.flag.mean()),
                     "precision": float(d.label[d.flag].mean()) if d.flag.any() else None}
    test = {
        "edits": int(len(y)),
        "reverted_share": float(y.mean()),
        "pr_auc": float(average_precision_score(y, p)),
        "flagged": int(flag.sum()),
        "flagged_share": float(flag.mean()),
        "precision": float(y[flag].mean()),
        "recall": float(y[flag].sum() / y.sum()),
        "by_day": days,
    }
    print(json.dumps(test, indent=2))

    for f in ("model.onnx", "vocab.txt"):
        shutil.copy(os.path.join(args.model, f), os.path.join(args.out, f))
    settings = json.load(open(os.path.join(args.model, "tokenizer.json")))
    config = {
        "max_len": args.max_len,
        "lowercase": settings["lowercase"],
        "diff_chars": args.diff_chars,
        "platt": {"a": float(a), "b": float(b)},
        "threshold": args.threshold,
        "model_bytes": os.path.getsize(os.path.join(args.out, "model.onnx")),
        "test": test,
    }
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
