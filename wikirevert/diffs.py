"""Fetching an edit's diff with action=compare and pulling text out of it.

The diff comes back as HTML table rows (difftype=table). Each row is a context
line, a wholly added line, a wholly removed line, or a changed line with the
changed words marked with <ins>/<del>. Paragraphs that only moved are marked
with mw-diff-movedpara links; they are counted but kept out of the added and
removed text, because moving text is not adding it.

Changed words on one line are joined with spaces, which can look odd: turning
"Valence" into a link inserts "[[" and ", Drôme|Valence]]" around the unchanged
word, stored as "[[ , Drôme|Valence]]". The full old and new lines are stored
too, and so is the raw HTML, so the model input can be rebuilt differently.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import lxml.html

from .http import ApiClient


@dataclass
class DiffText:
    added_text: list[str] = field(default_factory=list)
    removed_text: list[str] = field(default_factory=list)
    added_lines: list[str] = field(default_factory=list)
    removed_lines: list[str] = field(default_factory=list)
    lines_added: int = 0
    lines_removed: int = 0
    lines_changed: int = 0
    moved_paras: int = 0


@dataclass
class DiffResult:
    page_id: int | None
    title: str | None
    html: str
    text: DiffText


def _cell_text(cell) -> str:
    return cell.text_content().strip("\n")


def _is_moved(cell) -> bool:
    return bool(cell.xpath(".//a[contains(@class, 'mw-diff-movedpara')]"))


def extract(body: str) -> DiffText:
    out = DiffText()
    if not body or not body.strip():
        return out
    # The body is a run of <tr> elements; wrap it so lxml keeps the table cells.
    root = lxml.html.fromstring(f"<table>{body}</table>")
    for row in root.iter("tr"):
        deleted = row.xpath("./td[contains(@class, 'diff-deletedline')]")
        added = row.xpath("./td[contains(@class, 'diff-addedline')]")
        if not deleted and not added:
            continue
        if any(_is_moved(c) for c in deleted + added):
            out.moved_paras += 1
            continue
        if deleted and added:
            out.lines_changed += 1
            d, a = deleted[0], added[0]
            # Changed words on one line are joined with spaces, one line per entry.
            dels = [x.text_content() for x in d.iter("del") if x.text_content().strip()]
            ins = [x.text_content() for x in a.iter("ins") if x.text_content().strip()]
            if dels:
                out.removed_text.append(" ".join(dels))
            if ins:
                out.added_text.append(" ".join(ins))
            out.removed_lines.append(_cell_text(d))
            out.added_lines.append(_cell_text(a))
        elif added:
            out.lines_added += 1
            line = _cell_text(added[0])
            out.added_text.append(line)
            out.added_lines.append(line)
        else:
            out.lines_removed += 1
            line = _cell_text(deleted[0])
            out.removed_text.append(line)
            out.removed_lines.append(line)
    return out


def fetch(api: ApiClient, old_rev: int, new_rev: int) -> DiffResult:
    data = api.get(
        {
            "action": "compare",
            "fromrev": str(old_rev),
            "torev": str(new_rev),
            "prop": "diff|ids|title",
            "difftype": "table",
        }
    )
    cmp = data.get("compare") or {}
    body = cmp.get("body") or ""
    return DiffResult(
        page_id=cmp.get("toid") or cmp.get("fromid"),
        title=cmp.get("totitle"),
        html=body,
        text=extract(body),
    )


def _join(parts: list[str], sep: str, limit: int) -> str:
    s = sep.join(parts)
    return s if len(s) <= limit else s[:limit]


def to_row(rev_id: int, result: DiffResult, now: float, max_chars: int) -> dict:
    t = result.text
    html_bytes = result.html.encode("utf-8")
    return {
        "rev_id": rev_id,
        "status": "ok",
        "fetched_at": now,
        "diff_size": len(html_bytes),
        "added_text": _join(t.added_text, "\n", max_chars),
        "removed_text": _join(t.removed_text, "\n", max_chars),
        "added_lines": _join(t.added_lines, "\n", max_chars),
        "removed_lines": _join(t.removed_lines, "\n", max_chars),
        "lines_added": t.lines_added,
        "lines_removed": t.lines_removed,
        "lines_changed": t.lines_changed,
        "moved_paras": t.moved_paras,
        "html_z": zlib.compress(html_bytes, 6),
    }
