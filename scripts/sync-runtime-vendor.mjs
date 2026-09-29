/**
 * Re-vendors the MOSS-TTS-Nano browser runtime from upstream.
 *
 *   git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano-Reader /tmp/reader
 *   node scripts/sync-runtime-vendor.mjs
 *
 * Patches live in `scripts/vendor-patches.json` as exact `{find, replace}` pairs
 * validated against the upstream file, so the delta between Mockbird and
 * OpenMOSS's browser port is always auditable and re-appliable in one command.
 * See src/lib/tts/vendor/README.md for why each patch exists.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const upstream = process.env.MOSS_READER_PATH ?? "/tmp/reader/extension";

const targets = [
  {
    from: join(upstream, "browser_onnx_runtime.js"),
    to: join(root, "src/lib/tts/vendor/mossTtsRuntime.js"),
    patches: readJson("scripts/vendor-patches.json"),
  },
  {
    from: join(upstream, "tokenizer_sandbox.js"),
    to: join(root, "src/workers/vendor/tokenizerSandbox.js"),
    // The tokenizer bundle is patched at copy time by copy-tokenizer-worker.mjs
    // (window -> self), so no patches are applied here.
    patches: [],
  },
];

function readJson(relativePath) {
  return JSON.parse(readFileSync(resolve(root, relativePath), "utf8"));
}

let failed = false;

for (const target of targets) {
  if (!existsSync(target.from)) {
    console.error(`[sync-vendor] missing upstream file: ${target.from}`);
    console.error(
      "[sync-vendor] clone it with: git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano-Reader /tmp/reader",
    );
    failed = true;
    continue;
  }

  let source = readFileSync(target.from, "utf8");
  for (const patch of target.patches) {
    const count = source.split(patch.find).length - 1;
    if (count !== 1) {
      console.error(
        `[sync-vendor] patch "${patch.id}" matched ${count} times (expected 1) in ${target.from}`,
      );
      failed = true;
      continue;
    }
    source = source.replace(patch.find, patch.replace);
  }

  mkdirSync(dirname(target.to), { recursive: true });
  writeFileSync(target.to, source);
  console.log(
    `[sync-vendor] ${target.to.replace(`${root}/`, "")} (${target.patches.length} patches applied)`,
  );
}

if (failed) {
  process.exit(1);
}

console.log("[sync-vendor] done — run `pnpm typecheck && pnpm lint` to verify.");
