// Check the JavaScript tokeniser against ids from Hugging Face's tokeniser.
// Usage: node web/test-tokeniser.mjs vocab.txt cases.jsonl
// Each line of cases.jsonl: {"text": ..., "ids": [...] (cut at 256), "full": [...] (not cut)}
import { readFileSync } from "node:fs";
import { WordPiece } from "./wordpiece.mjs";

const [vocabPath, casesPath] = process.argv.slice(2);
const cut = new WordPiece(readFileSync(vocabPath, "utf8"), { maxLen: 256 });
const full = new WordPiece(readFileSync(vocabPath, "utf8"), { maxLen: 1e9 });
const lines = readFileSync(casesPath, "utf8").split("\n").filter(Boolean);
let bad = 0;
const t0 = performance.now();
for (const line of lines) {
  const c = JSON.parse(line);
  const a = cut.encode(c.text);
  const b = full.encode(c.text);
  if (a.join() !== c.ids.join() || b.join() !== c.full.join()) {
    bad += 1;
    if (bad <= 5) {
      const i = b.findIndex((x, k) => x !== c.full[k]);
      console.log("MISMATCH at token", i, JSON.stringify(c.text.slice(0, 120)));
      console.log("  js:", b.slice(Math.max(0, i - 3), i + 5).join(" "));
      console.log("  py:", c.full.slice(Math.max(0, i - 3), i + 5).join(" "));
    }
  }
}
const ms = (performance.now() - t0) / lines.length / 2;
console.log(`${lines.length - bad} of ${lines.length} match; ${ms.toFixed(3)} ms per text`);
process.exit(bad ? 1 : 0);
