/**
 * Robust TTS text normalisation — a port of the Apache-2.0 reference in
 * `upstream/MOSS-TTS-Nano/tts_robust_normalizer_single_script.py`.
 *
 * Ported from the Python rather than from the vendored JavaScript on purpose:
 * the Reader repository ships no licence (PLAN.md §10.1), so anything we keep
 * long-term has to come from an Apache-2.0 source. The upstream test corpus
 * (41 cases + idempotence) is the conformance vector — see
 * `tests/engine/textNormalize.test.mjs`.
 */

const CJK_CHARS = "\\u3400-\\u4dbf\\u4e00-\\u9fff\\u3040-\\u30ff";
const CJK = `[${CJK_CHARS}]`;
/** Placeholder protecting URLs/emails/handles from the spacing rules. */
const PROT = "___PROT\\d+___";

const PROT_CAPTURE = new RegExp(`(${PROT})`, "gu");
const PROT_FULL = new RegExp(`^${PROT}$`, "u");
const URL_RE = /https?:\/\/[^\s\u3000，。！？；、）】》〉」』]+/gu;
const EMAIL_RE = /(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w.-])/gu;
const MENTION_RE = /(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}/gu;
const REDDIT_RE = /(?<![A-Za-z0-9_])(?:u|r)\/[A-Za-z0-9_]+/gu;
const HASHTAG_RE = /(?<![A-Za-z0-9_])#(?!\s)[^\s#]+/gu;
/** `.map` / `.env` / `.gitignore` */
const DOT_TOKEN_RE = /(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+/gu;
/**
 * `app.js.map` / `index.d.ts` / `v2.3.1` / `foo/bar-baz.py`. Plain `snake_case`
 * is deliberately NOT matched, so the `_` -> space rule still applies to it.
 */
const FILELIKE_RE =
  /(?<![A-Za-z0-9_])(?=[A-Za-z0-9._/+:-]*[A-Za-z])(?=[A-Za-z0-9._/+:-]*[./+:-])[A-Za-z0-9][A-Za-z0-9._/+:-]*(?![A-Za-z0-9_])/gu;
/** Tokens that keep a space at CJK/Latin boundaries: has a latin letter, or is protected. */
const LATINISH = `(?:${PROT}|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)`;

const ZERO_WIDTH_RE = /[\u200b-\u200d\ufeff]/gu;
const TRAILING_CLOSERS = new Set(`"'）]}）】》〉」』『”’`);
const CONTROL_RE = /^[\p{Cc}]$/u;
const PUNCT_RE = /^\p{P}$/u;
const LINE_SPLIT_RE = /\n|\u2028|\u2029/gu;

function baseCleanup(text: string): string {
  const normalized = String(text || "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/\u3000/g, " ");
  const withoutZeroWidth = normalized.replace(ZERO_WIDTH_RE, "");
  const cleaned: string[] = [];
  // keep \n, \t and space; drop the remaining control characters
  for (const ch of withoutZeroWidth) {
    if (ch === "\n" || ch === "\t" || ch === " " || !CONTROL_RE.test(ch)) cleaned.push(ch);
  }
  return cleaned.join("");
}

function ensureTerminalPunctuation(text: string): string {
  if (!text) return text;
  let index = text.length - 1;
  while (index >= 0 && /\s/u.test(text[index])) index -= 1;
  while (index >= 0 && TRAILING_CLOSERS.has(text[index])) index -= 1;
  if (index >= 0 && PUNCT_RE.test(text[index])) return text;
  return `${text}。`;
}

function ensureTerminalPunctuationByLine(text: string): string {
  if (!text) return text;
  return text
    .split(LINE_SPLIT_RE)
    .map((line) => (line.trim() ? ensureTerminalPunctuation(line.trim()) : ""))
    .join("\n")
    .trim();
}

function normalizeFlowArrows(text: string): string {
  return text.replace(/\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*/gu, "，");
}

function normalizeMarkdownAndLines(text: string): string {
  const normalized = text.replace(/\[([^[\]]+?)\]\((https?:\/\/[^)\s]+)\)/gu, "$1 $2");
  const lines: string[] = [];
  for (const raw of normalized.split(LINE_SPLIT_RE)) {
    const line = raw.trim();
    if (!line) continue;
    lines.push(
      line
        .replace(/^#{1,6}\s+/u, "") // heading
        .replace(/^>\s+/u, "") // quote
        .replace(/^[-*+]\s+/u, "") // bullet
        .replace(/^\d+[.)]\s+/u, ""), // ordered list
    );
  }
  if (!lines.length) return "";
  const merged: string[] = [lines[0]];
  for (let i = 1; i < lines.length; i += 1) {
    merged[merged.length - 1] = ensureTerminalPunctuation(merged[merged.length - 1]);
    merged.push(lines[i]);
  }
  return merged.join("");
}

function protectSpans(text: string): { text: string; protected: string[] } {
  const protectedSpans: string[] = [];
  const replace = (match: string) => {
    const index = protectedSpans.length;
    protectedSpans.push(match);
    return `___PROT${index}___`;
  };
  let normalized = text;
  for (const pattern of [
    URL_RE,
    EMAIL_RE,
    MENTION_RE,
    REDDIT_RE,
    HASHTAG_RE,
    DOT_TOKEN_RE,
    FILELIKE_RE,
  ]) {
    normalized = normalized.replace(pattern, replace);
  }
  return { text: normalized, protected: protectedSpans };
}

function restoreSpans(text: string, protectedSpans: string[]): string {
  let restored = text;
  protectedSpans.forEach((span, index) => {
    restored = restored.split(`___PROT${index}___`).join(span);
  });
  return restored;
}

function normalizeVisibleUnderscores(text: string): string {
  return text
    .split(PROT_CAPTURE)
    .map((part) => (PROT_FULL.test(part) ? part : part.replace(/_/gu, " ")))
    .join("");
}

function normalizeSpaces(text: string): string {
  let normalized = text.replace(/[ \t\r\f\v]+/gu, " ");
  // spaces inside CJK runs are dropped
  normalized = normalized.replace(new RegExp(`(${CJK})\\s+(?=${CJK})`, "gu"), "$1");
  normalized = normalized.replace(new RegExp(`(${CJK})\\s+(?=\\d)`, "gu"), "$1");
  normalized = normalized.replace(new RegExp(`(\\d)\\s+(?=${CJK})`, "gu"), "$1");
  // exactly one space at CJK <-> latinish boundaries
  normalized = normalized.replace(new RegExp(`(${CJK})(?=(${LATINISH}))`, "gu"), "$1 ");
  normalized = normalized.replace(new RegExp(`((${LATINISH}))(?=${CJK})`, "gu"), "$1 ");
  normalized = normalized.replace(/ {2,}/gu, " ");
  // no space before CJK punctuation, none after an opening bracket
  normalized = normalized.replace(/\s+([，。！？；：、”’」』】）》])/gu, "$1");
  normalized = normalized.replace(/([（【「『《“‘])\s+/gu, "$1");
  normalized = normalized.replace(/([，。！？；：、])\s*/gu, "$1");
  normalized = normalized.replace(/\s+([,.;!?])/gu, "$1");
  return normalized.replace(/ {2,}/gu, " ").trim();
}

function normalizeStructuralPunctuation(text: string): string {
  let normalized = text
    .replace(/\[\s*([^[\]]+?)\s*\]/gu, '"$1"')
    .replace(/\{\s*([^{}]+?)\s*\}/gu, '"$1"')
    .replace(/[【〖『「]\s*([^】〗』」]+?)\s*[】〗』」]/gu, '"$1"');
  // 《》 only as a standalone title, not inline
  normalized = normalized.replace(
    /(^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))/gu,
    "$1$2",
  );
  normalized = normalizeFlowArrows(normalized);
  normalized = normalized.replace(/\s*(?:—|–|―|-){2,}\s*/gu, "。");
  return normalized;
}

function normalizeRepeatedPunctuation(text: string): string {
  let normalized = text.replace(/(?:\.{3,}|…{2,}|……+)/gu, "。");
  normalized = normalized
    .replace(/[。．]{2,}/gu, "。")
    .replace(/[，,]{2,}/gu, "，")
    .replace(/[!！]{2,}/gu, "！")
    .replace(/[?？]{2,}/gu, "？");
  normalized = normalized.replace(/[!?！？]{2,}/gu, (match) => {
    const hasQuestion = [...match].some((c) => c === "?" || c === "？");
    const hasExclamation = [...match].some((c) => c === "!" || c === "！");
    if (hasQuestion && hasExclamation) return "？！";
    return hasQuestion ? "？" : "！";
  });
  return normalized;
}

/** Normalise text for the TTS input. Idempotent. */
export function normalizeTtsText(text: string): string {
  let normalized = baseCleanup(text);
  normalized = normalizeMarkdownAndLines(normalized);
  normalized = normalizeFlowArrows(normalized);
  const { text: protectedText, protected: protectedSpans } = protectSpans(normalized);
  normalized = normalizeVisibleUnderscores(protectedText);
  normalized = normalizeSpaces(normalized);
  normalized = normalizeStructuralPunctuation(normalized);
  normalized = normalizeRepeatedPunctuation(normalized);
  normalized = normalizeSpaces(normalized);
  normalized = restoreSpans(normalized, protectedSpans);
  normalized = normalized.trim();
  return ensureTerminalPunctuationByLine(normalized);
}
