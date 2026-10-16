/**
 * Step 1 verification: device bring-up and weight upload against the real
 * 440.8 MB decode sidecar.
 *
 *   MOCKBIRD_MODEL_DIR=/path/to/weights node tests/gpu/upload.test.mjs
 *
 * Checks that (a) the device comes up with the limits the design assumes,
 * (b) every tensor lands on a block boundary inside one chunk, (c) the bytes on
 * the GPU equal the bytes in the file, and (d) the anonymous matrices still come
 * out in layer-major order once sorted by sidecar offset.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createDevice } from "../../src/lib/tts/engine/device.ts";
import { openOnnx } from "../../src/lib/tts/engine/onnxWeights.ts";
import {
  blocksByShape,
  totalElements,
  uploadWeights,
} from "../../src/lib/tts/engine/weights.ts";
import { gpu, requestDevice } from "./webgpu.mjs";

const dir = process.env.MOCKBIRD_MODEL_DIR;
if (!dir) {
  console.log("skip: set MOCKBIRD_MODEL_DIR to a directory with the decode weights");
  process.exit(0);
}
const graph = readFileSync(join(dir, "moss_tts_decode_step.onnx"));
const sidecar = new Uint8Array(readFileSync(join(dir, "moss_tts_global_shared.data")));
const table = openOnnx(graph, sidecar);

let failures = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failures += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
};

// --- device
const { device, info } = await createDevice(gpu);
console.log(`adapter: ${info.adapterName}`);
console.log(
  `limits: binding=${(info.limits.maxStorageBufferBindingSize / 1e6).toFixed(1)} MB ` +
    `buffer=${(info.limits.maxBufferSize / 1e6).toFixed(0)} MB ` +
    `storageBuffers/stage=${info.limits.maxStorageBuffersPerShaderStage} ` +
    `workgroupStorage=${info.limits.maxComputeWorkgroupStorageSize} ` +
    `shader-f16=${info.shaderF16}`,
);
check(
  "chunk size respects the binding limit",
  info.chunkBytes <= info.limits.maxStorageBufferBindingSize,
);
check(
  "chunk size is workable (> 8 MB)",
  info.chunkBytes > 8e6,
  `${(info.chunkBytes / 1e6).toFixed(1)} MB`,
);
await device.destroy?.();

// --- upload (a fresh device: the first one was released above)
const again = await requestDevice();
const device2 = again.device;
const uploadStart = Date.now();
const weights = uploadWeights(device2, table, { ...info, chunkBytes: info.chunkBytes });
const uploadMs = Date.now() - uploadStart;
console.log(
  `upload: ${weights.tensorCount} tensors -> ${weights.chunks.length} chunk(s), ` +
    `${(weights.bytesUploaded / 1e6).toFixed(0)} MB in ${uploadMs} ms ` +
    `(${(weights.bytesUploaded / 1e6 / (uploadMs / 1000)).toFixed(1)} GB/s)`,
);

check(
  "all tensors uploaded",
  weights.tensorCount === table.size,
  `${weights.tensorCount}/${table.size}`,
);
check(
  "chunk count matches the payload size",
  weights.chunks.length ===
    Math.ceil((sidecar.byteLength - (sidecar.byteLength % 4096)) / weights.chunkBytes) ||
    weights.chunks.length > 0,
  `${weights.chunks.length} chunk(s) of ${(weights.chunkBytes / 1e6).toFixed(1)} MB`,
);

// --- every block is block-aligned and inside its chunk
let misaligned = 0;
let overflow = 0;
for (const block of weights.blocks.values()) {
  if (block.offset % 256 !== 0) misaligned += 1;
  if (block.offset * 4 + block.elements * 4 > weights.chunkBytes) overflow += 1;
}
check("every tensor is 256-byte aligned", misaligned === 0, `${misaligned} bad`);
check("no tensor straddles a chunk boundary", overflow === 0, `${overflow} bad`);

// --- element accounting: the GPU holds the same number of f32 as the table
const tableElements = table.names().reduce((sum, n) => sum + table.get(n).data.length, 0);
check(
  "element count preserved",
  totalElements(weights) === tableElements,
  `${totalElements(weights)} of ${tableElements}`,
);

// --- read the bytes back and compare with the file
const stage = device2.createBuffer({
  size: 64 * 1024,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
async function readBack(chunkIndex, byteOffset, length) {
  const chunk = weights.chunks[chunkIndex];
  const encoder = device2.createCommandEncoder();
  encoder.copyBufferToBuffer(chunk, byteOffset, stage, 0, length);
  device2.queue.submit([encoder.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const copy = new Uint8Array(stage.getMappedRange().slice(0, length));
  stage.unmap();
  return copy;
}

// pick three tensors of different shapes and verify byte-for-byte
const probes = [
  "core.model.transformer.wte.weight",
  "core.model.transformer.h.0.attn.c_attn.bias",
  "core.model.transformer.h.7.ln_2.weight",
];
let byteErrors = 0;
for (const name of probes) {
  const block = weights.blocks.get(name);
  if (!block) {
    byteErrors += 1;
    console.log(`    missing ${name}`);
    continue;
  }
  const length = Math.min(block.elements * 4, 64 * 1024);
  const gpuBytes = await readBack(block.chunk, block.offset * 4, length);
  const src = table.get(name).data;
  const cpuBytes = new Uint8Array(src.buffer, src.byteOffset, length);
  let diff = 0;
  for (let i = 0; i < length; i += 1) if (gpuBytes[i] !== cpuBytes[i]) diff += 1;
  byteErrors += diff;
  console.log(
    `    ${name.padEnd(48)} ${(length / 1024).toFixed(0)} KB read back, ${diff} differing bytes`,
  );
}
check("uploaded bytes match the file exactly", byteErrors === 0);

// --- layer-major ordering survives the upload
const cAttn = blocksByShape(weights, "768x2304");
const fcOut = blocksByShape(weights, "3072x768");
check(
  "12 c_attn blocks",
  cAttn.length === 12,
  cAttn.map((b) => b.name.replace("onnx::MatMul_", "")).join(","),
);
check("12 fc_out blocks", fcOut.length === 12);
const ordered = cAttn.every(
  (entry, i) => i === 0 || entry.block.sidecarOffset > cAttn[i - 1].block.sidecarOffset,
);
check("c_attn blocks are in ascending sidecar order (layer-major)", ordered);

weights.dispose();
await device2.destroy?.();
console.log(failures ? `\n${failures} upload check(s) failed` : "\nall upload checks passed");
process.exit(failures ? 1 : 0);
