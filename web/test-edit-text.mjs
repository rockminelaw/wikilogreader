// Checks web/edit-text.mjs against the Python pipeline.
//   node web/test-edit-text.mjs text CASES.jsonl     student text from exported rows (training/data.py)
//   node web/test-edit-text.mjs parse CASES.jsonl    stream events and enrichment responses (wikirevert/)
// The diff extraction needs a browser's HTML parser, so it is checked by
// web/test-diffs.py in headless Chromium.
import fs from 'node:fs';
import readline from 'node:readline';
import { studentText, parseEdit, parseEnrich } from './edit-text.mjs';

const [mode, file] = process.argv.slice(2);
if (mode !== 'text' && mode !== 'parse') throw new Error('usage: node web/test-edit-text.mjs text|parse CASES.jsonl');

// Key order differs between the two, so compare as JSON with sorted keys.
const canon = (v) => JSON.stringify(v, (k, x) => (x && typeof x === 'object' && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort()) : x));
function parseCase(c) {
  if (c.kind === 'edit') {
    let ev;
    try {
      ev = JSON.parse(c.raw);
    } catch {
      ev = null;
    }
    return [canon(parseEdit(ev)), canon(c.want)];
  }
  const { tags, users } = parseEnrich(c.raw);
  return [canon({ tags: Object.fromEntries([...tags].map(([k, v]) => [String(k), v])), users: Object.fromEntries(users) }), canon(c.want)];
}

let n = 0;
let bad = 0;
const rl = readline.createInterface({ input: fs.createReadStream(file) });
for await (const line of rl) {
  if (!line) continue;
  const c = JSON.parse(line);
  n++;
  const [got, want] = mode === 'text' ? [studentText(c.row), c.text] : parseCase(c);
  if (got !== want) {
    bad++;
    if (bad <= 3) {
      const a = Array.from(got);
      const b = Array.from(want);
      let i = 0;
      while (i < a.length && a[i] === b[i]) i++;
      console.log(`mismatch at code point ${i}:\n  js: ${JSON.stringify(a.slice(Math.max(0, i - 40), i + 40).join(''))}\n  py: ${JSON.stringify(b.slice(Math.max(0, i - 40), i + 40).join(''))}`);
    }
  }
}
console.log(`${n - bad}/${n} match`);
process.exit(bad ? 1 : 0);
