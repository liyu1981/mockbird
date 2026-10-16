/**
 * Prefill conformance: the GPU prefill against a reference captured from
 * ONNX Runtime (`tests/fixtures/golden/prefill-*.f32`, produced from the
 * Apache-2.0 `moss_tts_prefill.onnx` — provenance in the reference JSON).
 *
 *   node --import ./tests/gpu/register-ts.mjs tests/gpu/prefill.test.mjs [8|195]
 *
 * This is the first kernel work that is not `M=1`, so the reference comes from
 * the authoritative graph rather than a second JS implementation.
 * onnxruntime-node is deliberately NOT a dependency of this repo, hence the
 * committed fixture. Set MOCKBIRD_MODEL_DIR for the weights.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createDevice } from "../../src/lib/tts/engine/device.ts";
import { openOnnx } from "../../src/lib/tts/engine/onnxWeights.ts";
import { PrefillRunner, resolveLayerWeights } from "../../src/lib/tts/engine/prefill.ts";
import { uploadWeights } from "../../src/lib/tts/engine/weights.ts";
import { gpu } from "./webgpu.mjs";

const ROWS = Number(process.argv[2] ?? 8);
const MAX_SLOTS = 256;
const HIDDEN = 768;

const dir = process.env.MOCKBIRD_MODEL_DIR;
const FIXTURES = new URL("../fixtures/golden/", import.meta.url);
const reference = JSON.parse(readFileSync(new URL("prefill-reference.json", FIXTURES), "utf8"));
const testCase = reference.cases.find((c) => c.rows === ROWS);
if (!testCase) {
  console.log(
    `skip: no committed reference for ${ROWS} rows (have ${reference.cases.map((c) => c.rows).join(", ")})`,
  );
  process.exit(0);
}
if (!dir) {
  console.log("skip: set MOCKBIRD_MODEL_DIR to a directory with the decode weights");
  process.exit(0);
}

// Node's Buffer is a Uint8Array: `new Float32Array(buf)` would copy BYTES as
// ELEMENTS, and `buf.buffer.slice(0)` would grab the whole shared pool. Slice the
// exact byte range out of the underlying ArrayBuffer instead.
function asArrayBuffer(url) {
  const buffer = readFileSync(url);
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}
const f32 = (name) => new Float32Array(asArrayBuffer(new URL(name, FIXTURES)));
const ids = new Uint32Array(asArrayBuffer(new URL(testCase.ids, FIXTURES)));
const refHidden = f32(testCase.hidden);
const refKey = f32(testCase.key);
const refValue = f32(testCase.value);
console.log(`replaying the ${ROWS}-row reference (max|hidden| = ${testCase.maxAbs})`);

const table = openOnnx(
  readFileSync(join(dir, "moss_tts_decode_step.onnx")),
  readFileSync(join(dir, "moss_tts_global_shared.data")),
);
const { device, info } = await createDevice(gpu);
const weights = uploadWeights(device, table, info);
const layers = resolveLayerWeights(weights, {
  get: (name) => {
    const block = weights.blocks.get(name);
    if (!block) throw new Error(`missing ${name}`);
    return block;
  },
});
console.log(`layers wired: ${layers.length}`);

const runner = new PrefillRunner(device, weights, MAX_SLOTS);
runner.stageIds(ids);
const embed = {
  wte: weights.blocks.get("core.model.transformer.wte.weight"),
  audio: Array.from({ length: 16 }, (_, k) =>
    weights.blocks.get(`core.model.audio_embeddings.${k}.weight`),
  ),
};
const started = Date.now();
const hidden = await runner.run(ROWS, layers, embed, MAX_SLOTS);
const gpuMs = Date.now() - started;
const flops = 2 * 85e6 * ROWS;
console.log(
  `  GPU prefill: ${gpuMs} ms for ${ROWS} rows (${(flops / 1e9 / (gpuMs / 1000)).toFixed(2)} GFLOP/s)`,
);

let failures = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failures += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
};

const maxAbsDiff = (a, b) => {
  let e = 0;
  let at = -1;
  for (let i = 0; i < b.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > e) {
      e = d;
      at = i;
    }
  }
  return { err: e, at };
};

const hiddenDiff = maxAbsDiff(hidden, refHidden);
check(
  "hidden state matches the reference (abs)",
  hiddenDiff.err < 5e-3,
  `maxErr=${hiddenDiff.err.toExponential(3)} at ${hiddenDiff.at}` +
    (hiddenDiff.at >= 0
      ? ` (row ${Math.floor(hiddenDiff.at / HIDDEN)}, dim ${hiddenDiff.at % HIDDEN})`
      : ""),
);

const perRow = [];
for (let r = 0; r < ROWS; r += 1) {
  let e = 0;
  for (let d = 0; d < HIDDEN; d += 1) {
    e = Math.max(e, Math.abs(hidden[r * HIDDEN + d] - refHidden[r * HIDDEN + d]));
  }
  perRow.push(e);
}
console.log(
  `  per-row maxErr: ${perRow
    .slice(0, 6)
    .map((v) => v.toExponential(1))
    .join(" ")}${ROWS > 6 ? " …" : ""}`,
);
check(
  "every row matches (not just the first)",
  Math.max(...perRow) < 5e-3,
  `worst=${Math.max(...perRow).toExponential(3)}`,
);

// the layer-0 cache is where the rotary convention and the head layout live, so
// compare it too: a wrong sign or layout would still give a plausible hidden row
// but a broken decode step
const cache = await runner.readLayerCache(0, ROWS);
const keyDiff = maxAbsDiff(cache.key, refKey);
const valueDiff = maxAbsDiff(cache.value, refValue);
check(
  "layer-0 key cache matches (RoPE convention)",
  keyDiff.err < 5e-3,
  `maxErr=${keyDiff.err.toExponential(3)} at ${keyDiff.at}`,
);
check(
  "layer-0 value cache matches",
  valueDiff.err < 5e-3,
  `maxErr=${valueDiff.err.toExponential(3)} at ${valueDiff.at}`,
);

weights.dispose();
runner.dispose();
await device.destroy?.();
console.log(
  failures ? `\n${failures} prefill check(s) failed` : "\nprefill matches the ONNX reference",
);
process.exit(failures ? 1 : 0);
