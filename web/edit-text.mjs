// Turning a live edit into the text the student model reads, in the
// browser. Each function mirrors one in the Python pipeline, and
// web/test-edit-text.mjs checks them against it:
//
//   parseEdit       wikirevert/stream.py   parse_edit
//   isAnonymous     wikirevert/users.py
//   extractDiff     wikirevert/diffs.py    extract, to_row
//   parseEnrich     wikirevert/enrich.py   parse_response
//   studentText     training/data.py       student_text (header, edit_text, capitals_line)
//
// Python counts string length in code points and splits on its own set of
// whitespace characters, so both are copied exactly here: JavaScript's
// String.length counts UTF-16 units and its \s is a slightly different set.

// ---------- editors ----------

const TEMP_ACCOUNT = /^~\d{4}-\d+-\d+$/;

export function isTempAccount(user) {
  return !!user && TEMP_ACCOUNT.test(user);
}

const IPV4 = /^(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}$/;

export function isIp(user) {
  if (!user) return false;
  if (IPV4.test(user)) return true;
  if (!user.includes(':') || !/^[0-9A-Fa-f:.]+$/.test(user)) return false;
  try {
    new URL(`http://[${user}]/`); // the URL parser validates IPv6 addresses
    return true;
  } catch {
    return false;
  }
}

export const isAnonymous = (user) => isTempAccount(user) || isIp(user);

// ---------- stream events ----------

export function parseEdit(ev, wiki = 'enwiki', namespace = 0) {
  if (!ev || typeof ev !== 'object') return null;
  const meta = ev.meta || {};
  if (meta.domain === 'canary') return null;
  if (ev.wiki !== wiki || ev.type !== 'edit' || ev.namespace !== namespace) return null;
  const rev = ev.revision || {};
  if (!Number.isInteger(rev.new)) return null;
  const length = ev.length || {};
  const sizeOld = length.old;
  const sizeNew = length.new;
  const user = ev.user;
  return {
    rev_id: rev.new,
    old_rev_id: rev.old ?? null,
    title: ev.title || '',
    user,
    is_bot: ev.bot ? 1 : 0,
    is_anon: isAnonymous(user) ? 1 : 0,
    is_temp: isTempAccount(user) ? 1 : 0,
    comment: ev.comment ?? null,
    minor: ev.minor ? 1 : 0,
    size_old: sizeOld ?? null,
    size_new: sizeNew ?? null,
    size_delta: Number.isInteger(sizeOld) && Number.isInteger(sizeNew) ? sizeNew - sizeOld : null,
    ts: Math.trunc(Number(ev.timestamp) || 0),
  };
}

// ---------- the diff ----------

const MAX_TEXT_CHARS = 50000;

// Python's str.strip("\n") and str.strip().
const stripNewlines = (s) => s.replace(/^\n+|\n+$/g, '');

const cut = (s, n) => {
  const cps = Array.from(s);
  return cps.length <= n ? s : cps.slice(0, n).join('');
};

const joinCapped = (parts, limit) => cut(parts.join('\n'), limit);

// `doc` is the result of parsing `<table>${body}</table>` as HTML.
export function extractDiff(body, parse) {
  const out = { added_text: [], removed_text: [], lines_added: 0, lines_removed: 0, lines_changed: 0, moved_paras: 0 };
  if (body && body.trim()) {
    const doc = parse(`<table>${body}</table>`);
    for (const row of doc.querySelectorAll('tr')) {
      const cells = [...row.children].filter((c) => c.tagName === 'TD');
      // contains(@class, ...) in the Python version: a substring of the class attribute.
      const has = (c, name) => (c.getAttribute('class') || '').includes(name);
      const deleted = cells.filter((c) => has(c, 'diff-deletedline'));
      const added = cells.filter((c) => has(c, 'diff-addedline'));
      if (!deleted.length && !added.length) continue;
      const moved = (c) => [...c.querySelectorAll('a')].some((a) => (a.getAttribute('class') || '').includes('mw-diff-movedpara'));
      if ([...deleted, ...added].some(moved)) {
        out.moved_paras += 1;
        continue;
      }
      if (deleted.length && added.length) {
        out.lines_changed += 1;
        const dels = [...deleted[0].querySelectorAll('del')].map((x) => x.textContent).filter((t) => pyStrip(t));
        const ins = [...added[0].querySelectorAll('ins')].map((x) => x.textContent).filter((t) => pyStrip(t));
        if (dels.length) out.removed_text.push(dels.join(' '));
        if (ins.length) out.added_text.push(ins.join(' '));
      } else if (added.length) {
        out.lines_added += 1;
        out.added_text.push(stripNewlines(added[0].textContent));
      } else {
        out.lines_removed += 1;
        out.removed_text.push(stripNewlines(deleted[0].textContent));
      }
    }
  }
  return {
    added_text: joinCapped(out.added_text, MAX_TEXT_CHARS),
    removed_text: joinCapped(out.removed_text, MAX_TEXT_CHARS),
    lines_added: out.lines_added,
    lines_removed: out.lines_removed,
    lines_changed: out.lines_changed,
    moved_paras: out.moved_paras,
  };
}

// ---------- editor details and tags ----------

const OUTCOME_TAGS = new Set(['mw-reverted']);

export const parseTs = (value) => (value ? Math.trunc(Date.parse(value) / 1000) : null);

// One action=query response (formatversion=2) with prop=revisions and list=users.
export function parseEnrich(data) {
  const q = (data && data.query) || {};
  const tags = new Map();
  for (const page of q.pages || []) {
    for (const rev of page.revisions || []) tags.set(rev.revid, (rev.tags || []).filter((t) => !OUTCOME_TAGS.has(t)));
  }
  const users = new Map();
  for (const u of q.users || []) {
    if (u.missing || u.invalid) continue;
    users.set(u.name, {
      registration: parseTs(u.registration),
      editcount: u.editcount ?? null,
      groups: (u.groups || []).filter((g) => g !== '*').join(','),
    });
  }
  return { tags, users };
}

// ---------- edit -> text (training/data.py) ----------

// Characters Python's str.split() treats as whitespace.
const PY_SPACE = new Set([
  '\t', '\n', '\x0b', '\x0c', '\r', '\x1c', '\x1d', '\x1e', '\x1f', ' ', '\x85', '\xa0', ' ',
  ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
  ' ', ' ', ' ', ' ', '　',
]);

function pySplit(s) {
  const words = [];
  let cur = '';
  for (const ch of s) {
    if (PY_SPACE.has(ch)) {
      if (cur) words.push(cur);
      cur = '';
    } else cur += ch;
  }
  if (cur) words.push(cur);
  return words;
}

function pyStrip(s) {
  const cps = Array.from(s);
  let a = 0;
  let b = cps.length;
  while (a < b && PY_SPACE.has(cps[a])) a++;
  while (b > a && PY_SPACE.has(cps[b - 1])) b--;
  return cps.slice(a, b).join('');
}

const missing = (v) => v === null || v === undefined || (typeof v === 'number' && Number.isNaN(v));

function countBucket(n) {
  if (missing(n)) return 'unknown edits';
  n = Math.trunc(n);
  if (n < 10) return `${n} edits`;
  for (const [lo, label] of [[10000, '10,000+'], [1000, '1,000-9,999'], [100, '100-999'], [10, '10-99']]) {
    if (n >= lo) return `${label} edits`;
  }
  return 'unknown edits';
}

function ageBucket(seconds) {
  if (missing(seconds) || seconds < 0) return 'unknown age';
  const day = 86400;
  for (const [limit, label] of [[3600, 'under an hour'], [day, 'under a day'], [7 * day, 'under a week'], [30 * day, 'under a month'], [365 * day, 'under a year']]) {
    if (seconds < limit) return label;
  }
  return `${Math.floor(seconds / (365 * day))}+ years`;
}

// Collapse whitespace, then cut to n code points with " …" if it was longer.
// Returns an array of code points so callers can keep slicing by them.
function clip(s, n) {
  if (missing(s)) return [];
  const cps = Array.from(pySplit(String(s)).join(' '));
  return cps.length <= n ? cps : [...cps.slice(0, n), ' ', '…'];
}

const int0 = (v) => (missing(v) || !v ? 0 : Math.trunc(v));

export function header(row) {
  const who = row.is_anon ? (row.is_temp ? 'anonymous editor (temporary account)' : 'anonymous editor') : 'registered editor';
  const reg = row.user_registration;
  const age = missing(reg) ? 'unknown age' : ageBucket(row.ts - reg);
  const groups = String(row.user_groups || '').split(',').filter((g) => g && g !== 'user' && g !== 'temp');
  const tags = String(row.tags_at_save || '').split(',').filter((t) => t);
  const delta = missing(row.size_delta) ? 0 : Math.trunc(row.size_delta);
  const signed = delta >= 0 ? `+${delta}` : String(delta);
  const summary = clip(row.comment, 200).join('') || '(none)';
  return (
    `${who}, account ${age}, ${countBucket(row.user_editcount)}` +
    (groups.length ? `, groups: ${groups.join(', ')}` : '') +
    `\nedit: ${row.minor ? 'minor, ' : ''}size ${int0(row.size_old)} to ${int0(row.size_new)} (${signed}), ` +
    `${int0(row.lines_added)} lines added, ${int0(row.lines_removed)} removed, ${int0(row.lines_changed)} changed` +
    (tags.length ? `\ntags: ${tags.join(', ')}` : '') +
    `\npage: ${row.title}` +
    `\nsummary: ${summary}`
  );
}

export function capitalsLine(added) {
  const s = missing(added) ? '' : String(added);
  let letters = 0;
  let caps = 0;
  for (const ch of s) {
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')) letters++;
    if (ch >= 'A' && ch <= 'Z') caps++;
  }
  if (letters < 20) return 'capitals: too little text';
  const share = caps / letters;
  for (const [limit, label] of [[0.1, 'few'], [0.3, 'some'], [0.7, 'many']]) {
    if (share < limit) return `capitals: ${label}`;
  }
  return 'capitals: mostly';
}

export function studentText(row, diffChars = 1400) {
  const removed = clip(row.removed_text, diffChars);
  const added = clip(row.added_text, diffChars);
  const rBudget = Math.min(removed.length, Math.max(Math.floor(diffChars / 3), diffChars - added.length));
  const aBudget = diffChars - rBudget;
  return (
    header(row) +
    `\n${capitalsLine(row.added_text)}` +
    `\nremoved: ${removed.slice(0, rBudget).join('') || '(nothing)'}` +
    `\nadded: ${added.slice(0, aBudget).join('') || '(nothing)'}`
  );
}
