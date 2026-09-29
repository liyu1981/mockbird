/**
 * Copies the ONNX Runtime Web wasm artifacts into `public/ort/`.
 *
 * Why: the MOSS-TTS-Nano worker imports `onnxruntime-web/wasm`, whose emscripten
 * glue is bundled inline, but the 12 MB `.wasm` binary is fetched at runtime from
 * `ort.env.wasm.wasmPaths`. We self-host it (instead of using a CDN) because the
 * page is served with `Cross-Origin-Embedder-Policy: require-corp`, and a
 * third-party host would need to send CORP headers.
 */
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const from = resolve(root, "node_modules/onnxruntime-web/dist");
const to = resolve(root, "public/ort");

const files = ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"];

mkdirSync(to, { recursive: true });

for (const file of files) {
  const src = join(from, file);
  const dest = join(to, file);
  try {
    const srcStat = statSync(src);
    const destStat = statSync(dest, { throwIfNoEntry: false });
    if (destStat && destStat.size === srcStat.size && destStat.mtimeMs >= srcStat.mtimeMs) {
      console.log(`[copy-ort] up to date: public/ort/${file}`);
      continue;
    }
    copyFileSync(src, dest);
    console.log(
      `[copy-ort] public/ort/${file} (${(srcStat.size / 1024 / 1024).toFixed(1)} MB)`,
    );
  } catch (error) {
    if (error?.code === "ENOENT") {
      console.warn(
        `[copy-ort] skipped ${file}: onnxruntime-web is not installed yet (run pnpm install)`,
      );
      continue;
    }
    throw error;
  }
}
