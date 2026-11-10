"""Score an exported int8 student with its input cut shorter than it was
trained on, to see how much ranking quality a faster model would give up.

Time per edit grows with the number of tokens, so cutting long edits from 256
tokens to 128 or 192 is the simplest way to make a model faster without
retraining. Only edits longer than the cut are affected (about one in seven
are at the full 256).

Usage:
    python -m training.shorter_inputs --model web-models/mini --splits runs/cache/splits-v2-maxchars1400-student \
        --lengths 128 192 --out reports/phase3/shorter-inputs-mini.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from training.export_onnx import score
from training.metrics import evaluate, fit_platt, sigmoid


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="export folder (model.onnx, hf-tokenizer/)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(args.model, "hf-tokenizer"))
    data = {s: pd.read_parquet(os.path.join(args.splits, f"{s}.parquet"), columns=["rev_id", "label", "text"])
            for s in ("val", "test")}
    result = {}
    for n in args.lengths:
        logits = {s: score(os.path.join(args.model, "model.onnx"), tok, d.text.tolist(), n)
                  for s, d in data.items()}
        a, b = fit_platt(logits["val"], data["val"].label.to_numpy())
        result[str(n)] = evaluate(data["test"].label.to_numpy(), sigmoid(a * logits["test"] + b))
        print(n, json.dumps({k: result[str(n)][k] for k in ("pr_auc", "brier")}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=float)


if __name__ == "__main__":
    main()
