"""Score edits with Wikimedia's revert-risk model on Lift Wing, as an outside
comparison.

Model: revertrisk-language-agnostic (gradient-boosted trees over article
quality features of the old and new revision, plus account age, number of
previous edits and number of user groups; trained on 2022 edits with a random
split). See https://meta.wikimedia.org/wiki/Machine_learning_models/Production/Language-agnostic_revert_risk

Caveat for the comparison: the service works out its features when it is
called, so editor details (edit count, groups) are as they are on the day of
scoring, not as they were when the edit was made. Scoring a week later gives
it slightly more information than a live system would have.

Requests go out in a random order, so a run that stops early still leaves a
uniform random sample. Results are appended to a JSON-lines file and a rerun
skips edits already scored.

Usage:
    python -m training.liftwing --rev-ids FILE.parquet --out FILE.jsonl [--rate 8]
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

URL = "https://api.wikimedia.org/service/lw/inference/v1/models/revertrisk-language-agnostic:predict"
UA = ("WikiLogReader/0.1.0 (https://github.com/rockminelaw/wikilogreader; celarry1@gmail.com) "
      f"python-requests/{requests.__version__}")


class Pacer:
    """At most `rate` request starts a second across all threads, and a shared
    pause after a 429."""

    def __init__(self, rate: float):
        self.gap = 1.0 / rate
        self.next = time.monotonic()
        self.lock = threading.Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if self.next > now:
                time.sleep(self.next - now)
            self.next = max(self.next, now) + self.gap

    def pause(self, seconds: float):
        with self.lock:
            self.next = max(self.next, time.monotonic() + seconds)


local = threading.local()


def session():
    s = getattr(local, "s", None)
    if s is None:
        s = local.s = requests.Session()
        s.headers["User-Agent"] = UA
    return s


def score(rev_id: int, pacer: Pacer) -> dict:
    backoff = 5.0
    for _ in range(8):
        pacer.wait()
        try:
            r = session().post(URL, json={"rev_id": int(rev_id), "lang": "en"}, timeout=30)
        except requests.RequestException as exc:
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
            last = f"network:{exc.__class__.__name__}"
            continue
        if r.status_code == 200:
            out = r.json()["output"]
            return {"rev_id": int(rev_id), "p": out["probabilities"]["true"], "status": "ok"}
        if r.status_code in (429, 503) or r.status_code >= 500:
            try:
                wait = float(r.headers.get("Retry-After", backoff))
            except ValueError:
                wait = backoff
            pacer.pause(wait)
            backoff = min(backoff * 2, 300)
            last = f"http{r.status_code}"
            continue
        return {"rev_id": int(rev_id), "p": None, "status": f"http{r.status_code}", "body": r.text[:200]}
    return {"rev_id": int(rev_id), "p": None, "status": f"gave_up:{last}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev-ids", required=True, help="parquet file with a rev_id column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", type=float, default=8.0, help="requests per second (default 8)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    ids = pd.read_parquet(args.rev_ids, columns=["rev_id"]).rev_id.astype(int).unique()
    ids = np.random.default_rng(args.seed).permutation(ids)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = {json.loads(line)["rev_id"] for line in f if line.strip()}
    todo = [int(i) for i in ids if int(i) not in done]
    print(f"{len(ids)} edits, {len(done)} already scored, {len(todo)} to go", flush=True)

    pacer, lock = Pacer(args.rate), threading.Lock()
    t0, n, errors = time.time(), 0, 0
    with open(args.out, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        for res in pool.map(lambda r: score(r, pacer), todo):
            with lock:
                f.write(json.dumps(res) + "\n")
                n += 1
                errors += res["status"] != "ok"
                if n % 1000 == 0:
                    f.flush()
                    rate = n / (time.time() - t0)
                    print(f"{n}/{len(todo)} scored, {errors} errors, {rate:.1f}/s, "
                          f"about {(len(todo) - n) / rate / 60:.0f} min left", flush=True)
    print(f"Done: {n} scored, {errors} errors", flush=True)


if __name__ == "__main__":
    main()
