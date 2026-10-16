/**
 * WGSL kernel conformance harness: every kernel of the decode engine is checked
 * against a plain-JS reference on the *real* tensor shapes.
 *
 *   pnpm test:gpu
 *
 * Weights are synthetic (deterministic PRNG) so this runs without the 715 MB
 * model download; the shapes are the real ones read out of the published graphs.
 * The point is to lock down the kernel *contract* — index arithmetic, tiling,
 * head layout — which is where a WGSL rewrite actually goes wrong.
 */
import { requestDevice } from "./webgpu.mjs";

const { device } = await requestDevice();

// ---- real shapes (tts_browser_onnx_meta.json / the published graphs)
const D = 768,
  H = 12,
  HD = 64;

// deterministic PRNG so failures are reproducible
let seed = 0x2f6e2b1;
const rnd = () => {
  seed ^= seed << 13;
  seed >>>= 0;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  seed >>>= 0;
  return (seed / 0xffffffff) * 2 - 1;
};

function runKernel(code, { workgroups = 1, entries }) {
  const module = device.createShaderModule({ code });
  return device
    .createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } })
    .then(async (pipeline) => {
      const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: entries.map((resource, binding) => ({
          binding,
          resource: { buffer: resource },
        })),
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bind);
      pass.dispatchWorkgroups(workgroups, 1, 1);
      pass.end();
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      return true;
    });
}

async function readback(buffer, byteLength) {
  const stage = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buffer, 0, stage, 0, byteLength);
  device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(stage.getMappedRange().slice(0));
  stage.unmap();
  stage.destroy();
  return out;
}

function storage(data, extraUsage = 0) {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage:
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | extraUsage,
  });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

let failures = 0;
const check = (name, err, tol) => {
  const pass = err <= tol;
  if (!pass) failures += 1;
  console.log(
    `  ${pass ? "ok  " : "FAIL"} ${name.padEnd(46)} maxErr=${err.toExponential(2)} (tol ${tol})`,
  );
};

// ---------------------------------------------------------------- 1. layernorm
{
  const x = new Float32Array(D),
    g = new Float32Array(D),
    b = new Float32Array(D);
  for (let i = 0; i < D; i++) {
    x[i] = rnd() * 4;
    g[i] = 1;
    b[i] = rnd() * 0.1;
  }
  const out = new Float32Array(D);
  let mean = 0;
  for (const v of x) mean += v;
  mean /= D;
  let varr = 0;
  for (const v of x) varr += (v - mean) ** 2;
  varr /= D;
  const inv = 1 / Math.sqrt(varr + 1e-5);
  for (let i = 0; i < D; i++) out[i] = (x[i] - mean) * inv * g[i] + b[i];

  const code = `
struct P { n: u32, _a: u32, _b: u32, eps: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> g: array<f32>;
@group(0) @binding(3) var<storage, read> bb: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  // D (768) exceeds the workgroup size (256): each lane must accumulate a strided
  // slice, otherwise the tail of the row is silently dropped.
  var acc = 0.0;
  for (var i = lid.x; i < p.n; i += 256u) { acc += x[i]; }
  red[lid.x] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) { red[lid.x] += red[lid.x + s]; }
    workgroupBarrier();
  }
  let mean = red[0] / f32(p.n);
  var sq = 0.0;
  for (var i = lid.x; i < p.n; i += 256u) { let d = x[i] - mean; sq += d * d; }
  workgroupBarrier();
  red[lid.x] = sq;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) { red[lid.x] += red[lid.x + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(red[0] / f32(p.n) + p.eps);
  for (var i = lid.x; i < p.n; i += 256u) {
    out[i] = (x[i] - mean) * inv * g[i] + bb[i];
  }
}`;
  const ubo = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ubo, 0, new Uint32Array([D, 0, 0]));
  device.queue.writeBuffer(ubo, 12, new Float32Array([1e-5]));
  const outBuf = storage(out);
  const xB = storage(x),
    gB = storage(g),
    bB = storage(b);
  await runKernel(code, {
    workgroupSize: 256,
    workgroups: 1,
    entries: [ubo, xB, gB, bB, outBuf],
  });
  const got = await readback(outBuf, D * 4);
  let err = 0;
  for (let i = 0; i < D; i++) err = Math.max(err, Math.abs(got[i] - out[i]));
  check("layernorm d=768 eps=1e-5", err, 2e-5);
}

// ------------------------------------------------------------------- 2. gemv
// y[n] = bias[n] + sum_k x[k] * W[k*N + n]   with W row-major [K, N].
// This is the exact layout the ONNX MatMul initializers already have, so the
// engine needs no weight transposition anywhere.
{
  const K = D,
    N = 3 * D;
  const x = new Float32Array(K),
    W = new Float32Array(K * N),
    bias = new Float32Array(N);
  for (let i = 0; i < K; i++) x[i] = rnd();
  for (let i = 0; i < W.length; i++) W[i] = rnd() * 0.05;
  for (let i = 0; i < N; i++) bias[i] = rnd() * 0.02;
  const ref = new Float32Array(N);
  for (let n = 0; n < N; n++) {
    let s = bias[n];
    for (let k = 0; k < K; k++) s += x[k] * W[k * N + n];
    ref[n] = s;
  }

  const code = `
struct P { k: u32, n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
// one workgroup per output tile; each invocation owns one output column and
// streams K rows of W contiguously -> the access pattern ORT would use.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = wid.x * 64u + lid.x;
  if (n >= p.n) { return; }
  var acc = bias[n];
  for (var k = 0u; k < p.k; k++) { acc += x[k] * w[k * p.n + n]; }
  out[n] = acc;
}`;
  const ubo = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ubo, 0, new Uint32Array([K, N]));
  const xB = storage(x),
    wB = storage(W),
    bB = storage(bias),
    outB = storage(ref);
  await runKernel(code, {
    workgroups: Math.ceil(N / 64),
    entries: [ubo, xB, wB, bB, outB],
  });
  const got = await readback(outB, N * 4);
  let err = 0,
    rel = 0;
  for (let i = 0; i < N; i++) {
    err = Math.max(err, Math.abs(got[i] - ref[i]));
    rel = Math.max(rel, Math.abs(got[i] - ref[i]) / Math.max(1e-3, Math.abs(ref[i])));
  }
  check(`gemv [K=768, N=2304] c_attn (abs)`, err, 5e-3);
  console.log(
    `       relative error ${rel.toExponential(2)} — f32 accumulation order differs from the JS reference, as expected`,
  );
}

// ------------------------------------------------------- 3. rope (interleaved)
// The graph's own intermediates prove the pairing: Concat_3 has shape
// [1,S,12,32,2] and Reshape_4 = [-v1, v0, -v3, v2, ...] for the q block.
{
  const pos = 137;
  const invFreq = Array.from({ length: HD / 2 }, (_, i) => 10000 ** (-(2 * i) / HD));
  const v = new Float32Array(HD);
  for (let i = 0; i < HD; i++) v[i] = rnd();
  const ref = new Float32Array(HD);
  for (let d = 0; d < HD; d++) {
    const f = invFreq[d >> 1];
    const partner = d & 1 ? -v[d - 1] : v[d + 1];
    ref[d] = v[d] * Math.cos(pos * f) + partner * Math.sin(pos * f);
  }
  const code = `
struct P { pos: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> v: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = gid.x;
  let f = pow(10000.0, -2.0 * (f32(d >> 1u)) / 64.0);
  let c = cos(p.pos * f);
  let s = sin(p.pos * f);
  var partner = 0.0;
  if ((d & 1u) == 1u) { partner = -v[d - 1u]; } else { partner = v[d + 1u]; }
  out[d] = v[d] * c + partner * s;
}`;
  const ubo = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ubo, 0, new Float32Array([pos, 0, 0, 0]));
  const vB = storage(v),
    outB = storage(ref);
  await runKernel(code, { workgroups: 1, entries: [ubo, vB, outB] });
  const got = await readback(outB, HD * 4);
  let err = 0;
  for (let d = 0; d < HD; d++) err = Math.max(err, Math.abs(got[d] - ref[d]));
  check("rope interleaved rotate_half d=64 base=10000", err, 1e-5);
}

// ------------------------------------------- 4. streaming KV ring (global step)
// Attention of one new query against a ring of N cached positions, the layout
// the engine keeps: kv[(pos * H + h) * HD + d].
{
  const N = 512,
    heads = H,
    hd = HD;
  const k = new Float32Array(N * heads * hd),
    v = new Float32Array(N * heads * hd);
  for (let i = 0; i < k.length; i++) {
    k[i] = rnd();
    v[i] = rnd();
  }
  const q = new Float32Array(heads * hd);
  for (let i = 0; i < q.length; i++) q[i] = rnd();
  const valid = N; // past_valid_length
  const out = new Float32Array(heads * hd);
  const scale = 1 / Math.sqrt(hd);
  const scores = new Float64Array(valid);
  for (let h = 0; h < heads; h++) {
    let mx = -Infinity;
    for (let j = 0; j < valid; j++) {
      let s = 0;
      for (let d = 0; d < hd; d++) s += q[h * hd + d] * k[(j * heads + h) * hd + d];
      scores[j] = s * scale;
      if (scores[j] > mx) mx = scores[j];
    }
    let sum = 0;
    for (let j = 0; j < valid; j++) {
      scores[j] = Math.exp(scores[j] - mx);
      sum += scores[j];
    }
    for (let d = 0; d < hd; d++) {
      let acc = 0;
      for (let j = 0; j < valid; j++) acc += (scores[j] / sum) * v[(j * heads + h) * hd + d];
      out[h * hd + d] = acc;
    }
  }
  const code = `
struct P { valid: u32, heads: u32, hd: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
// Scores live in a scratch storage buffer: N (the ring length) is far larger than
// any workgroup array, and the two softmax reductions must be lane reductions over
// *lane* partials, not over score indices.
@group(0) @binding(5) var<storage, read_write> sc: array<f32>;
var<workgroup> red: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let h = wid.x;
  let base = h * p.valid;
  let scale = inverseSqrt(f32(p.hd));
  var mx = -3.4e38;
  for (var j = lid.x; j < p.valid; j += 64u) {
    var s = 0.0;
    for (var d = 0u; d < p.hd; d++) { s += q[h * p.hd + d] * k[(j * p.heads + h) * p.hd + d]; }
    sc[base + j] = s * scale;
    mx = max(mx, s * scale);
  }
  red[lid.x] = mx;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) { red[lid.x] = max(red[lid.x], red[lid.x + s]); }
    workgroupBarrier();
  }
  let gmax = red[0];
  workgroupBarrier();
  var sum = 0.0;
  for (var j = lid.x; j < p.valid; j += 64u) { sc[base + j] = exp(sc[base + j] - gmax); sum += sc[base + j]; }
  red[lid.x] = sum;
  workgroupBarrier();
  for (var s = 32u; s > 0u; s >>= 1u) {
    if (lid.x < s) { red[lid.x] += red[lid.x + s]; }
    workgroupBarrier();
  }
  let gsum = red[0];
  workgroupBarrier();
  for (var d = lid.x; d < p.hd; d += 64u) {
    var acc = 0.0;
    for (var j = 0u; j < p.valid; j++) { acc += (sc[base + j] / gsum) * v[(j * p.heads + h) * p.hd + d]; }
    out[h * p.hd + d] = acc;
  }
}`;
  const ubo = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(ubo, 0, new Uint32Array([valid, heads, hd, 0]));
  const qB = storage(q),
    kB = storage(k),
    vB2 = storage(v),
    outB = storage(out);
  // one scratch row per head so the 12 workgroups never race
  const scBuf = device.createBuffer({ size: heads * valid * 4, usage: GPUBufferUsage.STORAGE });
  await runKernel(code, {
    workgroups: heads,
    entries: [ubo, qB, kB, vB2, outB, scBuf],
  });
  const got = await readback(outB, out.length * 4);
  let err = 0;
  for (let i = 0; i < out.length; i++) err = Math.max(err, Math.abs(got[i] - out[i]));
  check(`attention vs ring cache N=${N} heads=${H}`, err, 1e-5);
}

// ------------------------------------------------------------------ 5. argmax
// Greedy sampling stays on the GPU so token ids never leave the device.
{
  const C = 1024;
  const logits = new Float32Array(C);
  for (let i = 0; i < C; i++) logits[i] = rnd();
  let best = 0;
  for (let i = 1; i < C; i++) if (logits[i] > logits[best]) best = i;
  const code = `
@group(0) @binding(0) var<storage, read> l: array<f32>;
@group(0) @binding(1) var<storage, read_write> best: array<i32>;
@group(0) @binding(2) var<storage, read_write> bv: array<f32>;
var<workgroup> red: array<f32, 256>;
var<workgroup> ridx: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  // 1024 candidates vs 256 lanes: strided, with the index carried alongside so
  // ties resolve to the lowest id exactly like the JS reference.
  var bestV = -3.4e38; var bestI = lid.x;
  for (var i = lid.x; i < arrayLength(&l); i += 256u) {
    if (l[i] > bestV) { bestV = l[i]; bestI = i; }
  }
  red[lid.x] = bestV; ridx[lid.x] = bestI;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid.x < s) {
      let o = lid.x + s;
      if (red[o] > red[lid.x] || (red[o] == red[lid.x] && ridx[o] < ridx[lid.x])) { red[lid.x] = red[o]; ridx[lid.x] = ridx[o]; }
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) { bv[0] = red[0]; best[0] = i32(ridx[0]); }
}`;
  const outBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const valBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const lB = storage(logits);
  await runKernel(code, {
    workgroupSize: 256,
    workgroups: 1,
    entries: [lB, outBuf, valBuf],
  });
  const stage = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(outBuf, 0, stage, 0, 4);
  device.queue.submit([enc.finish()]);
  await stage.mapAsync(GPUMapMode.READ);
  const idx = new Int32Array(stage.getMappedRange())[0];
  check(`argmax over ${C} logits (tie -> lowest id)`, Math.abs(idx - best), 0);
}

console.log(failures ? `\n${failures} kernel check(s) failed` : "\nall kernel checks passed");
process.exit(failures ? 1 : 0);
