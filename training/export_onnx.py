"""Export a trained student to ONNX, quantise it to int8, and check that the
browser-sized model still scores like the PyTorch one.

Steps:
  1. export the Hugging Face checkpoint (best/) to ONNX in float32, with
     inputs input_ids and attention_mask and one output, the logit;
  2. quantise the weights to int8 (dynamic quantisation: weights stored as
     8-bit integers, activations quantised on the fly at run time);
  3. score the validation and test edits with both files in onnxruntime and
     compare them with the PyTorch scores in predictions.parquet.

Usage:
    python -m training.export_onnx --run runs/student-mini --splits runs/cache/splits-v2-maxchars1400-student --out web-models/mini

Writes to --out: model.onnx (int8), model-fp32.onnx, vocab.txt,
tokenizer.json (the browser tokeniser's settings) and export.json (sizes,
agreement with PyTorch, metrics of the int8 model, Platt parameters fitted on
validation).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd


def export(run_dir: str, out: str, opset: int = 17) -> str:
    import torch
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(run_dir, "best"), dtype=torch.float32).eval()

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            # token_type_ids default to zeros, which is what training used.
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits[:, 0]

    ids = torch.ones(2, 16, dtype=torch.long)
    mask = torch.ones(2, 16, dtype=torch.long)
    path = os.path.join(out, "model-fp32.onnx")
    # The TorchScript exporter: its graphs pass onnxruntime's quantiser and
    # optimiser cleanly, which the newer dynamo exporter's didn't.
    torch.onnx.export(
        Wrapped(model), (ids, mask), path, input_names=["input_ids", "attention_mask"],
        output_names=["logit"], opset_version=opset, dynamo=False,
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logit": {0: "batch"}},
    )
    return path


def quantise(fp32: str, out: str) -> str:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    path = os.path.join(out, "model.onnx")
    # MatMul and Gather cover the dense layers and the embedding table, which
    # is most of the file in these small models.
    quantize_dynamic(fp32, path, weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gather"])
    return path


def score(path: str, tok, texts, max_len: int, batch_size: int = 64) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros(len(texts), dtype=np.float32)
    for s in range(0, len(order), batch_size):
        g = order[s:s + batch_size]
        enc = tok([texts[i] for i in g], truncation=True, max_length=max_len, padding=True, return_tensors="np")
        out[g] = sess.run(["logit"], {"input_ids": enc["input_ids"].astype(np.int64),
                                      "attention_mask": enc["attention_mask"].astype(np.int64)})[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="student run folder (best/, predictions.parquet, metrics.json)")
    ap.add_argument("--splits", required=True, help="cached student splits folder (val.parquet, test.parquet)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="score only this many edits per split")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoTokenizer

    from training.metrics import evaluate, fit_platt, sigmoid

    run_args = json.load(open(os.path.join(args.run, "metrics.json")))["args"] \
        if os.path.exists(os.path.join(args.run, "metrics.json")) else {"max_len": 256}
    max_len = int(run_args.get("max_len", 256))
    tok = AutoTokenizer.from_pretrained(os.path.join(args.run, "best"))
    # The browser tokeniser needs the WordPiece vocabulary, one token per line
    # in id order, and the normalisation settings.
    vocab = sorted(tok.get_vocab().items(), key=lambda kv: kv[1])
    assert [i for _, i in vocab] == list(range(len(vocab))), "vocabulary ids are not contiguous"
    with open(os.path.join(args.out, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(t for t, _ in vocab) + "\n")
    tok.save_pretrained(os.path.join(args.out, "hf-tokenizer"))  # the reference, for checks
    with open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"lowercase": bool(getattr(tok, "do_lower_case", True)), "max_len": max_len,
                   "cls": tok.cls_token, "sep": tok.sep_token, "unk": tok.unk_token,
                   "pad": tok.pad_token}, f, indent=2)

    t0 = time.time()
    fp32 = export(args.run, args.out)
    int8 = quantise(fp32, args.out)
    info = {"max_len": max_len, "export_seconds": round(time.time() - t0, 1),
            "bytes_fp32": os.path.getsize(fp32), "bytes_int8": os.path.getsize(int8),
            "bytes_vocab": os.path.getsize(os.path.join(args.out, "vocab.txt"))}
    print(json.dumps(info), flush=True)

    preds = pd.read_parquet(os.path.join(args.run, "predictions.parquet"))
    result = {}
    scored = {}
    for split in ("val", "test"):
        d = pd.read_parquet(os.path.join(args.splits, f"{split}.parquet"), columns=["rev_id", "label", "text"])
        if args.limit:
            d = d.sample(min(len(d), args.limit), random_state=1)
        d = d.merge(preds[preds.split == split][["rev_id", "logit"]], on="rev_id")
        texts = d.text.tolist()
        s32 = score(fp32, tok, texts, max_len)
        s8 = score(int8, tok, texts, max_len)
        scored[split] = (d, s8)
        result[split] = {
            "n": len(d),
            "fp32_vs_pytorch_max_abs_logit_diff": float(np.max(np.abs(s32 - d.logit.to_numpy()))),
            "int8_vs_pytorch_mean_abs_logit_diff": float(np.mean(np.abs(s8 - d.logit.to_numpy()))),
            "int8_vs_pytorch_p99_abs_logit_diff": float(np.percentile(np.abs(s8 - d.logit.to_numpy()), 99)),
            "pytorch_pr_auc": evaluate(d.label.to_numpy(), sigmoid(d.logit.to_numpy()))["pr_auc"],
            "int8_pr_auc": evaluate(d.label.to_numpy(), sigmoid(s8))["pr_auc"],
        }
        print(split, json.dumps(result[split]), flush=True)

    # Calibration for the int8 model itself, fitted on validation.
    dv, sv = scored["val"]
    a, b = fit_platt(sv, dv.label.to_numpy())
    dt, st = scored["test"]
    info["platt_int8"] = [a, b]
    info["test_int8_calibrated"] = {k: v for k, v in evaluate(dt.label.to_numpy(), sigmoid(a * st + b)).items()
                                    if k != "reliability"}
    info["agreement"] = result
    pd.DataFrame({"rev_id": np.concatenate([dv.rev_id, dt.rev_id]),
                  "split": ["val"] * len(dv) + ["test"] * len(dt),
                  "logit_int8": np.concatenate([sv, st])}).to_parquet(os.path.join(args.out, "scores-int8.parquet"))
    with open(os.path.join(args.out, "export.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
