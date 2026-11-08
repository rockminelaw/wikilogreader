"""Fine-tune the teacher: a transformer that reads the edit as text
(training/data.py:edit_text) and predicts whether it will be reverted.

Class imbalance: about 5.5% of edits are positive. The model trains on the
natural distribution with plain cross-entropy rather than reweighting, which
keeps its probabilities meaningful; results are judged on PR-AUC and the
precision at a fixed alert rate, not accuracy; and a one-parameter
two-parameter (Platt) calibration is fitted on the validation day before the
scores are reported on the test day.

Usage (from the folder that contains training/):
    python -m training.train_teacher --exports ..\\wiki-export --out ..\\runs\\teacher-v1

Writes to --out:
    train.log          everything printed
    best/              the best checkpoint by validation PR-AUC (Hugging Face format)
    predictions.parquet  rev_id, split, label, logit for validation and test
    metrics.json       test metrics, raw and calibrated, latency, settings
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time

import numpy as np
import pandas as pd


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8", buffering=1)

    def __call__(self, *args):
        line = " ".join(str(a) for a in args)
        stamp = time.strftime("%H:%M:%S")
        print(f"{stamp} {line}", flush=True)
        self.f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")


def keep_awake(on: bool) -> None:
    """Stop Windows from sleeping while training runs."""
    if platform.system() != "Windows":
        return
    import ctypes

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
    ctypes.windll.kernel32.SetThreadExecutionState(flags)


def load_splits(args, log):
    from training.data import add_text, make_split

    cache = os.path.join(args.cache, f"splits-{args.split}-maxchars{args.diff_chars}")
    paths = {s: os.path.join(cache, f"{s}.parquet") for s in ("train", "val", "test")}
    if all(os.path.exists(p) for p in paths.values()):
        log(f"Loading cached splits from {cache}")
        return [pd.read_parquet(paths[s]) for s in ("train", "val", "test")]
    log(f"Building splits from {args.exports} (a few minutes the first time)")
    os.makedirs(cache, exist_ok=True)
    out = []
    for name, df in zip(("train", "val", "test"), make_split(args.exports, args.split)):
        df = add_text(df, args.diff_chars)[["rev_id", "day", "page", "label", "text"]]
        df.to_parquet(paths[name])
        out.append(df)
    return out


def batches(texts, labels, batch_size, shuffle, rng):
    """Group edits of similar length to cut padding, then shuffle the groups."""
    idx = np.arange(len(texts))
    if shuffle:
        rng.shuffle(idx)
    lengths = np.array([len(texts[i]) for i in idx])
    chunk = batch_size * 50
    groups = []
    for s in range(0, len(idx), chunk):
        part = idx[s:s + chunk]
        part = part[np.argsort(lengths[s:s + chunk], kind="stable")]
        groups += [part[i:i + batch_size] for i in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(groups)
    for g in groups:
        yield [texts[i] for i in g], labels[g], g


def predict(model, tok, texts, args, device, dtype, batch_size=64):
    import torch

    model.eval()
    out = np.zeros(len(texts), dtype=np.float32)
    with torch.no_grad():
        for bt, _, g in batches(texts, np.zeros(len(texts)), batch_size, False, None):
            enc = tok(bt, truncation=True, max_length=args.max_len, padding=True,
                      return_tensors="pt").to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(**enc).logits.squeeze(-1)
            out[g] = logits.float().cpu().numpy()
    model.train()
    return out


def latency(model, tok, texts, args, device, n):
    """Milliseconds per edit at batch size 1, tokenisation included."""
    import torch

    model.eval()
    times = []
    with torch.no_grad():
        for i, t in enumerate(texts[: n + 5]):
            t0 = time.perf_counter()
            enc = tok([t], truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            model(**enc).logits.float().cpu()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i >= 5:  # skip warm-up
                times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90)), "n": len(times)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--exports", required=True, help="folder with edits-YYYY-MM-DD files")
    p.add_argument("--out", required=True, help="folder for this run")
    p.add_argument("--cache", default=None, help="where to keep built splits (default: next to --out)")
    p.add_argument("--split", default="v2", help="which split in training/data.py (v1 or v2)")
    p.add_argument("--score-train", action="store_true",
                   help="also save the best model's logits on the training set (for distillation)")
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--diff-chars", type=int, default=1400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--limit", type=int, default=0, help="use only this many edits per split (testing)")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    args.cache = args.cache or os.path.join(os.path.dirname(os.path.abspath(args.out)), "cache")

    os.makedirs(args.out, exist_ok=True)
    log = Tee(os.path.join(args.out, "train.log"))
    log("Args:", json.dumps(vars(args)))

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from training.metrics import evaluate, fit_platt, fit_temperature, sigmoid, summary_line

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + f", bf16: {dtype is not None}, torch {torch.__version__}")

    train, val, test = load_splits(args, log)
    if args.limit:
        train, val, test = (d.sample(min(len(d), args.limit), random_state=args.seed).reset_index(drop=True)
                            for d in (train, val, test))
    for name, d in (("train", train), ("val", val), ("test", test)):
        log(f"{name}: {len(d)} edits, {int(d.label.sum())} reverted ({d.label.mean():.2%})")

    tok = AutoTokenizer.from_pretrained(args.model)
    # Load in float32: the checkpoint is stored in float16, and recent transformers
    # versions keep that dtype, which makes training overflow to NaN.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.float32).to(device)
    # Start the output at the base rate instead of 50/50, so early steps aren't
    # spent learning that most edits are fine.
    base = float(train.label.mean())
    head = getattr(model, "classifier", None)
    if isinstance(head, torch.nn.Linear) and 0 < base < 1:
        with torch.no_grad():
            head.bias.fill_(math.log(base / (1 - base)))
    lens = [len(x) for x in tok(list(train.text.sample(min(2000, len(train)), random_state=1)))["input_ids"]]
    log(f"Tokens per edit: median {int(np.median(lens))}, 95th percentile {int(np.percentile(lens, 95))}, "
        f"cut at {args.max_len} for {np.mean(np.array(lens) > args.max_len):.1%} of edits")

    texts, labels = train.text.tolist(), train.label.to_numpy().astype(np.float32)
    # Whole optimiser steps per epoch; a final partial accumulation is dropped.
    steps_per_epoch = max(1, math.ceil(len(texts) / args.batch_size) // args.accum)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    opt = torch.optim.AdamW(
        [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm"))],
          "weight_decay": 0.01},
         {"params": [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm"))],
          "weight_decay": 0.0}],
        lr=args.lr,
    )
    warm = max(1, int(total_steps * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total_steps - s) / max(1, total_steps - warm))))
    loss_fn = torch.nn.BCEWithLogitsLoss()

    log(f"Training for {args.epochs} epochs: {total_steps} optimiser steps of "
        f"{args.batch_size * args.accum} edits")
    keep_awake(True)
    rng = np.random.default_rng(args.seed)
    step, micro, best_ap, t_start, seen = 0, 0, -1.0, time.time(), 0
    running = []
    epoch = 0
    model.train()
    try:
        while step < total_steps:
            epoch += 1
            micro = 0
            opt.zero_grad(set_to_none=True)
            for bt, by, _ in batches(texts, labels, args.batch_size, True, rng):
                if micro >= steps_per_epoch * args.accum:
                    break
                enc = tok(bt, truncation=True, max_length=args.max_len, padding=True,
                          return_tensors="pt").to(device)
                target = torch.tensor(by, device=device)
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    logits = model(**enc).logits.squeeze(-1)
                loss = loss_fn(logits.float(), target) / args.accum
                if not torch.isfinite(loss):
                    raise RuntimeError(f"loss became {loss.item()} at step {step}; stopping")
                loss.backward()
                running.append(loss.item() * args.accum)
                micro += 1
                seen += len(bt)
                if micro % args.accum:
                    continue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0 or step == total_steps:
                    el = time.time() - t_start
                    rate = seen / el
                    eta = (total_steps - step) * args.batch_size * args.accum / rate
                    mem = (f", GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB"
                           if device.type == "cuda" else "")
                    log(f"epoch {epoch} step {step}/{total_steps}  loss {np.mean(running[-200:]):.4f}  "
                        f"{rate:.0f} edits/s  about {eta / 60:.0f} min left{mem}")
                if step >= total_steps:
                    break
            # Validate at the end of each epoch (and when stopping part-way).
            vlog = predict(model, tok, val.text.tolist(), args, device, dtype)
            m = evaluate(val.label.to_numpy(), sigmoid(vlog))
            log(summary_line(f"validation after epoch {epoch}", m))
            if m["pr_auc"] > best_ap:
                best_ap = m["pr_auc"]
                model.save_pretrained(os.path.join(args.out, "best"))
                tok.save_pretrained(os.path.join(args.out, "best"))
                log(f"Saved best checkpoint (validation PR-AUC {best_ap:.4f})")
    finally:
        keep_awake(False)

    log("Scoring validation and test with the best checkpoint")
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(args.out, "best"), dtype=torch.float32).to(device)
    vlog = predict(model, tok, val.text.tolist(), args, device, dtype)
    tlog = predict(model, tok, test.text.tolist(), args, device, dtype)
    temp = fit_temperature(vlog, val.label.to_numpy())
    a, b = fit_platt(vlog, val.label.to_numpy())
    raw = evaluate(test.label.to_numpy(), sigmoid(tlog))
    cal_t = evaluate(test.label.to_numpy(), sigmoid(tlog / temp))
    cal = evaluate(test.label.to_numpy(), sigmoid(a * tlog + b))
    log(f"Fitted on validation: temperature {temp:.3f}; Platt a={a:.3f} b={b:.3f}")
    log(summary_line("test, raw", raw))
    log(summary_line("test, temperature", cal_t))
    log(summary_line("test, Platt", cal))

    parts = [
        pd.DataFrame({"rev_id": val.rev_id, "split": "val", "label": val.label, "logit": vlog}),
        pd.DataFrame({"rev_id": test.rev_id, "split": "test", "label": test.label, "logit": tlog}),
    ]
    pd.concat(parts).to_parquet(os.path.join(args.out, "predictions.parquet"))
    if args.score_train:
        log("Scoring the training set for distillation")
        model.to(device)
        trlog = predict(model, tok, train.text.tolist(), args, device, dtype)
        parts.append(pd.DataFrame({"rev_id": train.rev_id, "split": "train", "label": train.label,
                                   "logit": trlog}))
        pd.concat(parts).to_parquet(os.path.join(args.out, "predictions.parquet"))
    lat = {"gpu" if device.type == "cuda" else "cpu": latency(model, tok, test.text.tolist(), args, device, 200)}
    if device.type == "cuda":
        torch.set_num_threads(os.cpu_count() or 4)
        lat["cpu"] = latency(model.to("cpu").float(), tok, test.text.tolist(), args, torch.device("cpu"), 40)
    log("Latency per edit (batch of 1, tokenising included):", json.dumps(lat))

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "temperature": temp, "platt": [a, b], "test_raw": raw,
                   "test_temperature": cal_t, "test_calibrated": cal,
                   "best_val_pr_auc": best_ap, "latency": lat,
                   "train_edits": len(train), "train_minutes": (time.time() - t_start) / 60,
                   "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                   "torch": torch.__version__}, f, indent=2)
    log(f"Done. Results in {args.out}")


if __name__ == "__main__":
    sys.exit(main())
