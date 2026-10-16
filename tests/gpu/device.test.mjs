/**
 * Headless WebGPU device probe.
 *
 * The WGSL engine (docs/moss-tts-runtime-analysis.md §9) has no browser fallback,
 * so every design decision is validated here against a real device. Run:
 *
 *   pnpm test:gpu
 *
 * Reports the limits the weight/activation buffer design depends on, and runs a
 * trivial compute kernel to prove the adapter actually executes work.
 */
import { requestDevice } from "./webgpu.mjs";

const REQUIRED_LIMITS = {
  // The 440 MB global weight sidecar must be split across bindings if this is the
  // default. Design assumption: chunk the weights, never rely on one big binding.
  maxStorageBufferBindingSize: 134_217_728,
  maxBufferSize: 268_435_456,
  maxStorageBuffersPerShaderStage: 8,
  maxComputeWorkgroupStorageSize: 16_384,
  maxComputeInvocationsPerWorkgroup: 256,
};

let adapter, device;
try {
  ({ adapter, device } = await requestDevice());
} catch (error) {
  console.error(`${error.message} — this machine cannot run the WGSL engine`);
  process.exit(1);
}
const L = adapter.limits;

console.log("adapter limits");
let ok = true;
for (const [name, needed] of Object.entries(REQUIRED_LIMITS)) {
  const have = L[name];
  const pass = have >= needed;
  ok = ok && pass;
  console.log(`  ${pass ? "ok  " : "FAIL"} ${name.padEnd(34)} ${have} (need >= ${needed})`);
}
console.log(
  `  ${L.maxStorageBuffersPerShaderStage >= 16 ? "ok  " : "note"} maxStorageBuffersPerShaderStage 16+ enables single-buffer weight packing`,
);
console.log(
  `features: ${["shader-f16", "timestamp-query", "float32-filterable"].map((f) => `${f}=${adapter.features.has(f)}`).join(" ")}`,
);

// trivial kernel: proves the adapter computes
const module = device.createShaderModule({
  code: `
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  out[gid.x] = f32(gid.x) * 2.0 + 1.0;
}`,
});
const pipeline = device.createComputePipeline({
  layout: "auto",
  compute: { module, entryPoint: "main" },
});
const buf = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
});
const read = device.createBuffer({
  size: 256,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
});
const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline);
pass.setBindGroup(
  0,
  device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: buf } }],
  }),
);
pass.dispatchWorkgroups(1);
pass.end();
encoder.copyBufferToBuffer(buf, 0, read, 0, 256);
device.queue.submit([encoder.finish()]);
await read.mapAsync(GPUMapMode.READ);
const values = new Float32Array(read.getMappedRange()).slice(0, 4);
const correct = values[0] === 1 && values[1] === 3 && values[2] === 5 && values[3] === 7;
console.log(`compute smoke test: [${values.join(", ")}] ${correct ? "ok" : "FAIL"}`);

if (!ok || !correct) process.exit(1);
console.log("headless WebGPU ready");
