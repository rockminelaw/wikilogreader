"""Run web/bench.html in headless Chromium against an exported student.

Picks random test edits, scores them with onnxruntime in Python (the
reference), serves the page, the model and onnxruntime-web from a local
folder, and records the browser's latency and agreement.

Usage:
    python web/bench.py --model web-models/mini --splits runs/cache/splits-v2-maxchars1400-student \
        --ort node_modules/onnxruntime-web/dist --out reports/phase3/bench-mini.json [--n 500]
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import tempfile
import threading

import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # so "training" imports from the repo


def reference_cases(model_dir: str, splits: str, n: int, seed: int) -> list[dict]:
    """Random test edits with the int8 model's logits from onnxruntime in
    Python, scored one at a time like the browser does."""
    from transformers import AutoTokenizer

    from training.export_onnx import score

    settings = json.load(open(os.path.join(model_dir, "tokenizer.json")))
    d = pd.read_parquet(os.path.join(splits, "test.parquet"), columns=["rev_id", "text"]).sample(n, random_state=seed)
    # The tokeniser saved at export (building one from vocab.txt alone gives
    # wrong ids in transformers 5).
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "hf-tokenizer"))
    logits = score(os.path.join(model_dir, "model.onnx"), tok, d.text.tolist(), settings["max_len"], batch_size=1)
    return [{"rev_id": int(r), "text": t, "logit": float(x)} for r, t, x in zip(d.rev_id, d.text, logits)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--ort", required=True, help="onnxruntime-web dist folder")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--debug", action="store_true", help="also save every edit's two scores")
    args = ap.parse_args(argv)

    site = tempfile.mkdtemp(prefix="bench-")
    for f in ("bench.html", "wordpiece.mjs"):
        shutil.copy(os.path.join(HERE, f), site)
    os.makedirs(os.path.join(site, "ort"))
    for f in ("ort.wasm.min.mjs", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"):
        shutil.copy(os.path.join(args.ort, f), os.path.join(site, "ort"))
    os.makedirs(os.path.join(site, "model"))
    for f in ("model.onnx", "vocab.txt", "tokenizer.json"):
        shutil.copy(os.path.join(args.model, f), os.path.join(site, "model"))
    cases = reference_cases(args.model, args.splits, args.n, args.seed)
    with open(os.path.join(site, "model", "bench-cases.json"), "w") as f:
        json.dump(cases, f)

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    handler = functools.partial(Quiet, directory=site)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = (f"http://127.0.0.1:{srv.server_port}/bench.html?model=model&threads={args.threads}"
           + ("&debug=1" if args.debug else ""))

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.wait_for_function("window.benchResult || window.benchError", timeout=30 * 60 * 1000)
        err = page.evaluate("window.benchError")
        if err:
            raise SystemExit(err)
        result = page.evaluate("window.benchResult")
        if args.debug:
            result["rows"] = page.evaluate("window.debugRows")
        browser.close()
    srv.shutdown()
    result["model_bytes"] = os.path.getsize(os.path.join(args.model, "model.onnx"))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    shutil.rmtree(site, ignore_errors=True)


if __name__ == "__main__":
    main()
