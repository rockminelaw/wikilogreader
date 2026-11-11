"""Check web/edit-text.mjs's diff extraction against wikirevert/diffs.py on
real compare responses, using headless Chromium's HTML parser.

Usage: python web/test-diffs.py DIFFS.jsonl   (one {"rev_id", "body"} per line)
"""

import functools
import http.server
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from wikirevert.diffs import DiffResult, extract, to_row  # noqa: E402

FIELDS = ["added_text", "removed_text", "lines_added", "lines_removed", "lines_changed", "moved_paras"]


def main():
    cases = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    want = []
    for c in cases:
        row = to_row(c["rev_id"], DiffResult(None, None, c["body"], extract(c["body"])), 0.0, 50_000)
        want.append({k: row[k] for k in FIELDS})

    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=HERE))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(f"http://127.0.0.1:{srv.server_port}/bench.html?model=none")
        got = pg.evaluate("""async (bodies) => {
            const { extractDiff } = await import('./edit-text.mjs');
            const parse = (h) => new DOMParser().parseFromString(h, 'text/html');
            return bodies.map((b) => extractDiff(b, parse));
        }""", [c["body"] for c in cases])
        b.close()
    bad = [(c["rev_id"], k) for c, w, g in zip(cases, want, got) for k in FIELDS if w[k] != g[k]]
    print(f"{len(cases) - len({r for r, _ in bad})}/{len(cases)} diffs match")
    for rev, k in bad[:5]:
        print("mismatch", rev, k)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
