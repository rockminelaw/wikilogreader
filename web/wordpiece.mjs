// WordPiece tokeniser for the uncased BERT-family students, matching Hugging
// Face's BertTokenizerFast: clean the text, space out CJK characters,
// lowercase and strip accents, split on whitespace and punctuation, then
// split each word into the longest vocabulary pieces ("##" marks a piece
// that continues a word). Tested against the Python tokeniser on real edits
// (web/test-tokeniser.mjs).

const PUNCT = /[\p{P}]/u;
const MARK = /\p{Mn}/u;
// Control, format, private-use and surrogate characters are dropped, as in
// Hugging Face's clean-up. Unassigned code points are kept: a browser's
// Unicode tables can be older than the Python tokeniser's, and a character
// that is new to Unicode should become [UNK] as it does there.
const CONTROL = /[\p{Cc}\p{Cf}\p{Co}\p{Cs}]/u;
const SPACE_SEP = /\p{Zs}/u;

function isWhitespace(ch) {
  return ch === " " || ch === "\t" || ch === "\n" || ch === "\r" || SPACE_SEP.test(ch);
}

function isControl(ch) {
  if (ch === "\t" || ch === "\n" || ch === "\r") return false;
  return CONTROL.test(ch);
}

function isPunctuation(ch) {
  const c = ch.codePointAt(0);
  // ASCII symbols such as $ + < = > ^ ` | ~ count as punctuation in BERT.
  if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126)) {
    return true;
  }
  return PUNCT.test(ch);
}

function isCJK(c) {
  return (c >= 0x4e00 && c <= 0x9fff) || (c >= 0x3400 && c <= 0x4dbf) ||
    (c >= 0x20000 && c <= 0x2a6df) || (c >= 0x2a700 && c <= 0x2b73f) ||
    (c >= 0x2b740 && c <= 0x2b81f) || (c >= 0x2b820 && c <= 0x2ceaf) ||
    (c >= 0xf900 && c <= 0xfaff) || (c >= 0x2f800 && c <= 0x2fa1f);
}

export function normalise(text, lowercase = true) {
  let out = "";
  for (const ch of text) {
    const c = ch.codePointAt(0);
    if (c === 0 || c === 0xfffd || isControl(ch)) continue;
    if (isWhitespace(ch)) out += " ";
    else if (isCJK(c)) out += " " + ch + " ";
    else out += ch;
  }
  if (lowercase) {
    // Accents are stripped (decompose, drop combining marks) before lowercasing.
    let stripped = "";
    // Lowercase one character at a time: lowercasing a whole string applies
    // context rules (a Greek capital sigma at the end of a word becomes the
    // final form) that Hugging Face's tokeniser doesn't.
    for (const ch of out.normalize("NFD")) if (!MARK.test(ch)) stripped += ch.toLowerCase();
    out = stripped;
  }
  return out;
}

export function preTokenise(text) {
  const words = [];
  let cur = "";
  for (const ch of text) {
    if (isWhitespace(ch)) {
      if (cur) words.push(cur);
      cur = "";
    } else if (isPunctuation(ch)) {
      if (cur) words.push(cur);
      words.push(ch);
      cur = "";
    } else {
      cur += ch;
    }
  }
  if (cur) words.push(cur);
  return words;
}

export class WordPiece {
  constructor(vocabText, { lowercase = true, maxLen = 256, unk = "[UNK]", cls = "[CLS]", sep = "[SEP]" } = {}) {
    const tokens = vocabText.split("\n");
    if (tokens[tokens.length - 1] === "") tokens.pop();
    this.vocab = new Map(tokens.map((t, i) => [t, i]));
    this.lowercase = lowercase;
    this.maxLen = maxLen;
    this.unkId = this.vocab.get(unk);
    this.clsId = this.vocab.get(cls);
    this.sepId = this.vocab.get(sep);
    this.maxWordChars = 100;
  }

  wordIds(word, out) {
    const chars = Array.from(word);
    if (chars.length > this.maxWordChars) {
      out.push(this.unkId);
      return;
    }
    const pieces = [];
    let start = 0;
    while (start < chars.length) {
      let end = chars.length;
      let id;
      while (start < end) {
        let piece = chars.slice(start, end).join("");
        if (start > 0) piece = "##" + piece;
        id = this.vocab.get(piece);
        if (id !== undefined) break;
        end -= 1;
      }
      if (id === undefined) {
        out.push(this.unkId);
        return;
      }
      pieces.push(id);
      start = end;
    }
    for (const id of pieces) out.push(id);
  }

  // Token ids with [CLS] and [SEP], cut to maxLen like truncation=True.
  encode(text) {
    const ids = [this.clsId];
    const limit = this.maxLen - 1;
    for (const word of preTokenise(normalise(text, this.lowercase))) {
      this.wordIds(word, ids);
      if (ids.length >= limit) break;
    }
    ids.length = Math.min(ids.length, limit);
    ids.push(this.sepId);
    return ids;
  }
}
