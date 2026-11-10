"""Distil the teacher into a small student that can run in a browser.

The student reads the same edit text as the teacher plus one line about
capitals (training/data.py:student_text), because the small models lowercase
everything. It learns from two targets at once:

    loss = alpha * BCE(student, label)
         + (1 - alpha) * T^2 * BCE(student / T, sigmoid(teacher / T))

The hard label is what actually happened; the soft label is the teacher's
probability, softened by the temperature T so the student also learns how
sure the teacher was. With --alpha 1 the teacher is ignored, which gives the
"same student, no teacher" comparison.

The teacher's logits on the training edits come from its run with
--score-train (predictions.parquet, split "train").

Usage (from the folder that contains training/):
    python -m training.distil --teacher ..\\runs\\teacher-v2-256 --exports ..\\wiki-export \\
        --model google/bert_uncased_L-4_H-256_A-4 --out ..\\runs\\student-mini

Writes the same files as train_teacher: train.log, best/, predictions.parquet,
metrics.json.
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

from training.train_teacher import Tee, batches, keep_awake, latency, predict


def load_student_splits(args, log):
    from training.data import add_text, make_split, student_text

    cache = os.path.join(args.cache, f"splits-{args.split}-maxchars{args.diff_chars}-student")
    paths = {s: os.path.join(cache, f"{s}.parquet") for s in ("train", "val", "test")}
    if all(os.path.exists(p) for p in paths.values()):
        log(f"Loading cached student splits from {cache}")
        return [pd.read_parquet(paths[s]) for s in ("train", "val", "test")]
    log(f"Building student splits from {args.exports} (a few minutes the first time)")
    os.makedirs(cache, exist_ok=True)
    out = []
    for name, df in zip(("train", "val", "test"), make_split(args.exports, args.split)):
        df = add_text(df, args.diff_chars, student_text)[["rev_id", "day", "page", "label", "text"]]
        df.to_parquet(paths[name])
        out.append(df)
    return out


def output_layer(model):
    """The final Linear layer of the classification head (BERT has one Linear,
    ELECTRA a small MLP ending in out_proj)."""
    import torch

    head = getattr(model, "classifier", None)
    if isinstance(head, torch.nn.Linear):
        return head
    return getattr(head, "out_proj", None)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, help="teacher run folder (with train logits)")
    p.add_argument("--exports", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--split", default="v2")
    p.add_argument("--model", default="google/bert_uncased_L-4_H-256_A-4")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--diff-chars", type=int, default=1400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=float, default=0.06)
    p.add_argument("--alpha", type=float, default=0.3, help="weight of the hard label (1 = no teacher)")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    args.cache = args.cache or os.path.join(os.path.dirname(os.path.abspath(args.out)), "cache")

    os.makedirs(args.out, exist_ok=True)
    log = Tee(os.path.join(args.out, "train.log"))
    log("Args:", json.dumps(vars(args)))

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from training.metrics import evaluate, fit_platt, sigmoid, summary_line

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    log(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
        + f", bf16: {dtype is not None}, torch {torch.__version__}")

    train, val, test = load_student_splits(args, log)
    tp = pd.read_parquet(os.path.join(args.teacher, "predictions.parquet"), columns=["rev_id", "split", "logit"])
    tp = tp[tp.split == "train"][["rev_id", "logit"]].rename(columns={"logit": "teacher_logit"})
    train = train.merge(tp, on="rev_id", how="left")
    missing = int(train.teacher_logit.isna().sum())
    if missing and args.alpha < 1:
        raise SystemExit(f"{missing} training edits have no teacher logit; was the teacher run with --score-train?")
    train["teacher_logit"] = train.teacher_logit.fillna(0.0)
    if args.limit:
        train, val, test = (d.sample(min(len(d), args.limit), random_state=args.seed).reset_index(drop=True)
                            for d in (train, val, test))
    for name, d in (("train", train), ("val", val), ("test", test)):
        log(f"{name}: {len(d)} edits, {int(d.label.sum())} reverted ({d.label.mean():.2%})")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=1, dtype=torch.float32).to(device)
    n_params = sum(x.numel() for x in model.parameters())
    log(f"Student {args.model}: {n_params / 1e6:.1f}M parameters")
    base = float(train.label.mean())
    out_layer = output_layer(model)
    if out_layer is not None and 0 < base < 1:
        with torch.no_grad():
            out_layer.bias.fill_(math.log(base / (1 - base)))
    lens = [len(x) for x in tok(list(train.text.sample(min(2000, len(train)), random_state=1)))["input_ids"]]
    log(f"Tokens per edit: median {int(np.median(lens))}, 95th percentile {int(np.percentile(lens, 95))}, "
        f"cut at {args.max_len} for {np.mean(np.array(lens) > args.max_len):.1%} of edits")

    texts = train.text.tolist()
    # Column 0 is the hard label, column 1 the teacher's logit.
    targets = np.stack([train.label.to_numpy().astype(np.float32),
                        train.teacher_logit.to_numpy().astype(np.float32)], axis=1)
    steps_per_epoch = math.ceil(len(texts) / args.batch_size)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm"))]
    no_decay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    warm = max(1, int(total_steps * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total_steps - s) / max(1, total_steps - warm))))
    bce = torch.nn.BCEWithLogitsLoss()
    T, alpha = args.temperature, args.alpha

    log(f"Training for {args.epochs} epochs: {total_steps} steps of {args.batch_size} edits, "
        f"alpha {alpha} (hard label weight), temperature {T}")
    keep_awake(True)
    rng = np.random.default_rng(args.seed)
    step, best_ap, t_start, seen, epoch = 0, -1.0, time.time(), 0, 0
    running = []
    model.train()
    try:
        while step < total_steps:
            epoch += 1
            for i, (bt, by, _) in enumerate(batches(texts, targets, args.batch_size, True, rng)):
                if i >= steps_per_epoch or step >= total_steps:
                    break
                enc = tok(bt, truncation=True, max_length=args.max_len, padding=True,
                          return_tensors="pt").to(device)
                y = torch.tensor(by[:, 0], device=device)
                t_logit = torch.tensor(by[:, 1], device=device)
                with torch.autocast(device.type, dtype=dtype, enabled=dtype is not None):
                    s = model(**enc).logits.squeeze(-1)
                s = s.float()
                loss = alpha * bce(s, y)
                if alpha < 1:
                    loss = loss + (1 - alpha) * T * T * bce(s / T, torch.sigmoid(t_logit / T))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"loss became {loss.item()} at step {step}; stopping")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                running.append(loss.item())
                seen += len(bt)
                step += 1
                if step % 200 == 0 or step == total_steps:
                    el = time.time() - t_start
                    rate = seen / el
                    eta = (total_steps - step) * args.batch_size / rate
                    log(f"epoch {epoch} step {step}/{total_steps}  loss {np.mean(running[-400:]):.4f}  "
                        f"{rate:.0f} edits/s  about {eta / 60:.0f} min left")
            vlog = predict(model, tok, val.text.tolist(), args, device, dtype, batch_size=256)
            m = evaluate(val.label.to_numpy(), sigmoid(vlog))
            log(summary_line(f"validation after epoch {epoch}", m))
            if m["pr_auc"] > best_ap:
                best_ap = m["pr_auc"]
                model.save_pretrained(os.path.join(args.out, "best"))
                tok.save_pretrained(os.path.join(args.out, "best"))
                log(f"Saved best checkpoint (validation PR-AUC {best_ap:.4f})")
    finally:
        keep_awake(False)
    train_minutes = (time.time() - t_start) / 60

    log("Scoring validation and test with the best checkpoint")
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(args.out, "best"), dtype=torch.float32).to(device)
    vlog = predict(model, tok, val.text.tolist(), args, device, dtype, batch_size=256)
    tlog = predict(model, tok, test.text.tolist(), args, device, dtype, batch_size=256)
    a, b = fit_platt(vlog, val.label.to_numpy())
    raw = evaluate(test.label.to_numpy(), sigmoid(tlog))
    cal = evaluate(test.label.to_numpy(), sigmoid(a * tlog + b))
    log(f"Fitted on validation: Platt a={a:.3f} b={b:.3f}")
    log(summary_line("test, raw", raw))
    log(summary_line("test, Platt", cal))
    pd.concat([
        pd.DataFrame({"rev_id": val.rev_id, "split": "val", "label": val.label, "logit": vlog}),
        pd.DataFrame({"rev_id": test.rev_id, "split": "test", "label": test.label, "logit": tlog}),
    ]).to_parquet(os.path.join(args.out, "predictions.parquet"))

    lat = {"gpu" if device.type == "cuda" else "cpu": latency(model, tok, test.text.tolist(), args, device, 200)}
    if device.type == "cuda":
        torch.set_num_threads(os.cpu_count() or 4)
        lat["cpu"] = latency(model.to("cpu").float(), tok, test.text.tolist(), args, torch.device("cpu"), 100)
    log("Latency per edit in PyTorch (batch of 1, tokenising included):", json.dumps(lat))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"args": vars(args), "parameters": n_params, "platt": [a, b], "test_raw": raw,
                   "test_calibrated": cal, "best_val_pr_auc": best_ap, "latency_pytorch": lat,
                   "train_edits": len(train), "train_minutes": train_minutes,
                   "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
                   "torch": torch.__version__}, f, indent=2)
    log(f"Done. Results in {args.out}")


if __name__ == "__main__":
    sys.exit(main())
