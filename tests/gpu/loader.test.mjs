/**
 * Loader conformance test: runs the real `onnxWeights.ts` loader against the
 * published MOSS-TTS-Nano graphs.
 *
 * The decode weights are not committed (715 MB). Point MOCKBIRD_MODEL_DIR at a
 * directory holding them; the test is skipped when it is absent.
 *
 *   MOCKBIRD_MODEL_DIR=/tmp/mockbird-oracle pnpm test:gpu
 */
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { openOnnx } from "../../src/lib/tts/engine/onnxWeights.ts";

const dir = process.env.MOCKBIRD_MODEL_DIR;
const graph = join(dir ?? "", "moss_tts_decode_step.onnx");
const sidecar = join(dir ?? "", "moss_tts_global_shared.data");

if (!dir || !existsSync(graph) || !existsSync(sidecar)) {
  console.log("skip: set MOCKBIRD_MODEL_DIR to a directory with the decode weights");
  process.exit(0);
}

let failures = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failures += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
};

const bytes = readFileSync(graph);
const data = readFileSync(sidecar);
const table = openOnnx(bytes, data);

console.log(`loader: ${table.size} parameters resolved from ${graph.split("/").pop()}`);

// 1. every parameter the reference forward needs must resolve
const names = [
  "core.model.transformer.wte.weight",
  "core.model.transformer.ln_f.weight",
  "core.model.transformer.ln_f.bias",
  ...Array.from({ length: 16 }, (_, k) => `core.model.audio_embeddings.${k}.weight`),
  ...Array.from({ length: 12 }, (_, i) => `core.model.transformer.h.${i}.ln_1.weight`),
  ...Array.from({ length: 12 }, (_, i) => `core.model.transformer.h.${i}.ln_2.weight`),
  ...Array.from({ length: 12 }, (_, i) => `core.model.transformer.h.${i}.attn.c_attn.bias`),
  ...Array.from({ length: 12 }, (_, i) => `core.model.transformer.h.${i}.mlp.fc_out.bias`),
];
const missing = names.filter((n) => !table.has(n));
check(
  `${names.length} named parameters resolve (incl. Identity-aliased LayerNorm scales)`,
  missing.length === 0,
  missing.slice(0, 3).join(", "),
);

// 2. anonymous MatMul initializers group into 12 layers x 4 shapes, layer-major
const counts = ["768x2304", "768x768", "768x3072", "3072x768"].map(
  (s) => table.byShape(s).length,
);
check(
  "anonymous matrices: 12 x [c_attn, c_proj, fc_in, fc_out]",
  counts.every((c) => c === 12),
  counts.join("/"),
);

// 3. the LayerNorm scale is an all-ones alias of layer 0's tensor
const ln0 = table.get("core.model.transformer.h.0.ln_1.weight");
const ln11 = table.get("core.model.transformer.h.11.ln_2.weight");
check(
  "LayerNorm scales are all ones and share storage (Identity alias)",
  ln0.data.every((v) => v === 1) &&
    ln11.data.every((v) => v === 1) &&
    ln11.data.buffer === ln0.data.buffer,
);

// 4. zero-copy: external weights are views into the OPFS/sidecar buffer
const wte = table.get("core.model.transformer.wte.weight");
check(
  "external weights are zero-copy views into the sidecar",
  wte.data.buffer === data.buffer && wte.source === "external",
  `wte[0]=${wte.data[0].toFixed(6)} dims=${wte.dims.join("x")}`,
);

// 5. inline constants (none of the big ones here, but the path must not throw)
const inline = table.names().filter((n) => {
  const t = table.get(n);
  return t.source === "inline-constant" || t.source === "inline-raw";
});
check("inline-constant tensors parse", Array.isArray(inline));

// 6. the weight inventory matches the measured model
const totalFloats = [...table.names()].reduce((sum, n) => sum + table.get(n).data.length, 0);
check(
  "weight inventory = 440.8 MB of fp32",
  Math.abs((totalFloats * 4) / 1e6 - 440.8) < 1.5,
  `${((totalFloats * 4) / 1e6).toFixed(1)} MB over ${table.size} tensors`,
);

console.log(failures ? `\n${failures} loader check(s) failed` : "\nall loader checks passed");
process.exit(failures ? 1 : 0);
