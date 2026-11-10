"""Loading the daily exports, labelling, splitting by time, and turning an
edit into the text the transformer models read.

Scope: edits by accounts without the bot flag. Bot edits are 30-50% of the
stream but only about 0.2% of them are reverted, so the demo will show them
without a score.

Label: reverted by someone else within 48 hours. Self-reverts (an editor
undoing their own edit) count as not reverted, because nobody else had to
step in.

Dropped rows: edits with no label (deleted pages or revisions) and edits
whose diff could not be fetched. The second group is reverted far more
often than average (about 26% against 6%), so keeping them without a diff
would let a model learn "no diff means bad edit", which is a collection
artefact, not something about the edit.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from dataclasses import dataclass

import pandas as pd

META_COLUMNS = [
    "rev_id", "page_id", "title", "ts", "user_hash", "is_bot", "is_anon", "is_temp",
    "minor", "comment", "size_old", "size_new", "size_delta", "user_registration",
    "user_editcount", "user_groups", "tags_at_save", "diff_status",
    "lines_added", "lines_removed", "lines_changed", "moved_paras",
    "label_status", "reverted", "reverted_48h", "revert_seconds", "reverter_is_self",
]
# The model input (edit_text) and the baselines only read these two. The full
# changed lines are in the export too, but loading them doubles the memory.
TEXT_COLUMNS = ["added_text", "removed_text"]


@dataclass(frozen=True)
class Split:
    train_days: tuple[str, ...]
    val_days: tuple[str, ...]
    test_days: tuple[str, ...]


# v1: the first split, while the collector's diff backlog caught up. Its test
# set is the first 16 hours of 29 October (the part with diffs).
SPLIT_V1 = Split(
    train_days=("2026-10-25", "2026-10-26", "2026-10-27"),
    val_days=("2026-10-28",),
    test_days=("2026-10-29",),
)
# v2: eight days of training, then three whole days of test that come after
# everything the models saw. 29 and 30 October are about two-thirds covered
# (the rest were still waiting for diffs at export), which only means less
# training data from those days.
SPLIT_V2 = Split(
    train_days=("2026-10-25", "2026-10-26", "2026-10-27", "2026-10-28",
                "2026-10-29", "2026-10-30", "2026-10-31", "2026-11-01"),
    val_days=("2026-11-02",),
    test_days=("2026-11-03", "2026-11-04", "2026-11-05"),
)
SPLITS = {"v1": SPLIT_V1, "v2": SPLIT_V2}


def read_day(path: str, with_text: bool = True, bots: bool = False) -> pd.DataFrame:
    """One day's export. Bot edits are dropped while reading unless bots=True,
    which keeps memory down: they are a third to a half of all rows."""
    cols = META_COLUMNS + (TEXT_COLUMNS if with_text else [])
    if path.endswith(".parquet"):
        df = pd.read_parquet(path, columns=cols)
        return df if bots else df[df.is_bot == 0].reset_index(drop=True)
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if bots or not rec.get("is_bot"):
                rows.append({c: rec.get(c) for c in cols})
    return pd.DataFrame.from_records(rows, columns=cols)


def day_of(name: str) -> str:
    # edits-2026-10-25.jsonl.gz or edits-2026-10-25.parquet
    return name.split("edits-", 1)[1][:10]


def load_days(export_dir: str, days, with_text: bool = True, bots: bool = False,
              each=None) -> pd.DataFrame:
    """Load the given days, applying `each` (e.g. clean) to every day before
    combining them, so only the rows that are kept are held in memory."""
    wanted = set(days)
    frames, found = [], set()
    for name in sorted(os.listdir(export_dir)):
        if not name.startswith("edits-") or day_of(name) not in wanted:
            continue
        if not (name.endswith(".jsonl.gz") or name.endswith(".parquet")):
            continue
        df = read_day(os.path.join(export_dir, name), with_text, bots).assign(day=day_of(name))
        found.add(day_of(name))
        frames.append(each(df) if each else df)
    missing = wanted - found
    if missing:
        raise FileNotFoundError(f"no export files for {sorted(missing)} in {export_dir}")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df.is_bot == 0) & (df.label_status == "ok") & (df.diff_status == "ok")].copy()
    df = df[df.reverted_48h.notna()]
    df["label"] = ((df.reverted_48h == 1) & (df.reverter_is_self != 1)).astype("int8")
    df["page"] = df.page_id.astype("Int64").astype(str).where(df.page_id.notna(), "t:" + df.title)
    return df


def make_split(export_dir: str, split: Split | str = "v2", with_text: bool = True):
    """Return train, val, test. No page appears in more than one of them: a
    page seen later is removed from the earlier sets, so the test set stays
    an untouched slice of the live stream."""
    if isinstance(split, str):
        split = SPLITS[split]
    days = split.train_days + split.val_days + split.test_days
    df = load_days(export_dir, days, with_text, each=clean)
    test = df[df.day.isin(split.test_days)]
    val = df[df.day.isin(split.val_days) & ~df.page.isin(set(test.page))]
    later = set(test.page) | set(val.page)
    train = df[df.day.isin(split.train_days) & ~df.page.isin(later)]
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ---------- edit -> text ----------


def _count_bucket(n) -> str:
    if n is None or (isinstance(n, float) and math.isnan(n)):
        return "unknown edits"
    n = int(n)
    if n < 10:
        return f"{n} edits"
    for lo, label in ((10_000, "10,000+"), (1_000, "1,000-9,999"), (100, "100-999"), (10, "10-99")):
        if n >= lo:
            return f"{label} edits"
    return "unknown edits"


def _age_bucket(seconds) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)) or seconds < 0:
        return "unknown age"
    for limit, label in (
        (3600, "under an hour"), (86400, "under a day"), (7 * 86400, "under a week"),
        (30 * 86400, "under a month"), (365 * 86400, "under a year"),
    ):
        if seconds < limit:
            return label
    return f"{int(seconds // (365 * 86400))}+ years"


def _clip(s, n: int) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = " ".join(str(s).split())  # collapse newlines and runs of spaces
    return s if len(s) <= n else s[:n] + " …"


def header(row) -> str:
    """Everything except the diff, in one short block."""
    if row["is_anon"]:
        who = "anonymous editor (temporary account)" if row["is_temp"] else "anonymous editor"
    else:
        who = "registered editor"
    reg = row["user_registration"]
    age = _age_bucket(row["ts"] - reg) if reg == reg and reg is not None else "unknown age"
    groups = [g for g in str(row["user_groups"] or "").split(",")
              if g and g not in ("user", "temp")]
    tags = [t for t in str(row["tags_at_save"] or "").split(",") if t]
    delta = row["size_delta"]
    delta = 0 if delta != delta or delta is None else int(delta)
    lines = (
        f"{who}, account {age}, {_count_bucket(row['user_editcount'])}"
        + (f", groups: {', '.join(groups)}" if groups else "")
        + f"\nedit: {'minor, ' if row['minor'] else ''}size {int(row['size_old'] or 0)} to "
        f"{int(row['size_new'] or 0)} ({delta:+d}), {int(row['lines_added'] or 0)} lines added, "
        f"{int(row['lines_removed'] or 0)} removed, {int(row['lines_changed'] or 0)} changed"
        + (f"\ntags: {', '.join(tags)}" if tags else "")
        + f"\npage: {row['title']}"
        + f"\nsummary: {_clip(row['comment'], 200) or '(none)'}"
    )
    return lines


def edit_text(row, diff_chars: int = 1400) -> str:
    """Header, then the removed and added text with a shared character
    budget, so a long addition can use what a short removal leaves."""
    removed = _clip(row["removed_text"], diff_chars)
    added = _clip(row["added_text"], diff_chars)
    r_budget = min(len(removed), max(diff_chars // 3, diff_chars - len(added)))
    a_budget = diff_chars - r_budget
    return (
        header(row)
        + f"\nremoved: {removed[:r_budget] or '(nothing)'}"
        + f"\nadded: {added[:a_budget] or '(nothing)'}"
    )


def capitals_line(added) -> str:
    """The small student models lowercase their input, so they can't see
    shouting. This line tells them how much of the added text is capitals."""
    s = "" if added is None or (isinstance(added, float) and math.isnan(added)) else str(added)
    letters = sum(1 for c in s if "a" <= c.lower() <= "z" and c.isascii())
    if letters < 20:
        return "capitals: too little text"
    caps = sum(1 for c in s if "A" <= c <= "Z")
    share = caps / letters
    for limit, label in ((0.1, "few"), (0.3, "some"), (0.7, "many")):
        if share < limit:
            return f"capitals: {label}"
    return "capitals: mostly"


def student_text(row, diff_chars: int = 1400) -> str:
    """The teacher's input with the capitals line added before the diff, so
    cutting a long input never drops it."""
    t = edit_text(row, diff_chars)
    i = t.index("\nremoved: ")
    return t[:i] + "\n" + capitals_line(row["added_text"]) + t[i:]


def add_text(df: pd.DataFrame, diff_chars: int = 1400, fn=None) -> pd.DataFrame:
    fn = fn or edit_text
    df = df.copy()
    df["text"] = [fn(r, diff_chars) for r in df.to_dict("records")]
    return df
