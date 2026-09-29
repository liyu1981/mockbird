/**
 * Emits `public/tokenizer.worker.js` from the vendored SentencePiece bundle.
 *
 * The bundle embeds the emscripten/SentencePiece wasm as base64, so the worker
 * is self-contained: it only receives the 0.47 MB `tokenizer.model` at runtime
 * from the OPFS model cache.
 *
 * It lives in `public/` (not in the module graph) on purpose: dev uses webpack
 * and Vercel uses Turbopack, and a plain `new Worker("/tokenizer.worker.js",
 * { type: "module" })` is the one loading strategy that is identical under both.
 * See docs/adr/0001-bundler-parity.md.
 */
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const src = resolve(root, "src/workers/vendor/tokenizerSandbox.js");
const dest = resolve(root, "public/tokenizer.worker.js");

const banner = `/* Mockbird: generated from src/workers/vendor/tokenizerSandbox.js by
 * scripts/copy-tokenizer-worker.mjs. Do not edit. Upstream:
 * https://github.com/OpenMOSS/MOSS-TTS-Nano-Reader (extension/tokenizer_sandbox.js)
 */`;

const edits = [
  [
    'window.addEventListener("message", async (event) => {',
    'self.addEventListener("message", async (event) => {',
  ],
  // `event.source.postMessage({...}, "*")` -> `self.postMessage({...})`
  ["event.source?.postMessage({", "self.postMessage({"],
  ['}, "*");', "});"],
];

mkdirSync(dirname(dest), { recursive: true });

let code;
try {
  code = readFileSync(src, "utf8");
} catch (error) {
  if (error?.code === "ENOENT") {
    console.warn(
      "[copy-tokenizer-worker] vendored tokenizer bundle missing; run scripts/sync-runtime-vendor.mjs",
    );
    process.exit(0);
  }
  throw error;
}

for (const [needle, replacement] of edits) {
  if (!code.includes(needle)) {
    console.error(`[copy-tokenizer-worker] patch target not found: ${needle.slice(0, 50)}…`);
    process.exitCode = 1;
    continue;
  }
  code = code.split(needle).join(replacement);
}

writeFileSync(dest, `${banner}\n${code}`);

const destStat = statSync(dest);
const upToDate = destStat.mtimeMs >= statSync(src).mtimeMs;
console.log(
  `[copy-tokenizer-worker] public/tokenizer.worker.js (${(destStat.size / 1024).toFixed(0)} KB)${upToDate ? "" : ""}`,
);
