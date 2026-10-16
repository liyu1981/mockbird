/**
 * WGSL kernels for the MOSS-TTS-Nano decode path.
 *
 * One kernel per stage of the verified reference (docs §10.5/§10.6). Every
 * weight is addressed as `weights[base + k*N + n]` from the metadata table, so
 * the ONNX `[K, N]` row-major initializers are used verbatim — no transposes,
 * no repacking.
 */

/** uniform block shared by every kernel: offsets + sizes, 16-byte aligned. */
const PRELUDE = /* wgsl */ `
struct Params {
  // row width (N) of the weight block being applied
  n: u32,
  // number of rows (K) of the weight block
  k: u32,
  // element offset of the weight block inside the packed weight buffer
  wOff: u32,
  _pad0: u32,
  // element offset of the bias block (length n) inside the weights buffer
  bOff: u32,
  // element offset of the destination inside the scratch buffer
  dOff: u32,
  // source length (embedding rows are gathered one at a time)
  srcLen: u32,
  _pad1: u32,
}
`;

/**
 * y[n] = bias[n] + sum_k x[k] * W[k*N + n]
 *
 * One invocation per output column; each streams K contiguous-ish rows of W, so
 * the weight block is read exactly once per call. This is the only shape ORT
 * and this kernel ever run at M=1 (decode), and it is memory-bound by design.
 */
export const GEMV = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = wid.x * 64u + lid.x;
  if (n >= p.n) { return; }
  let w = p.wOff + n;
  let b = p.bOff + n;
  var acc = weights[b];
  for (var k = 0u; k < p.k; k = k + 1u) {
    acc = acc + x[k] * weights[w + k * p.n];
  }
  scratch[p.dOff + n] = acc;
}
`;

/**
 * Row embedding: y = wte[textId] + sum_k audioEmb_k[code_k]  (padded channels
 * contribute nothing). `p.srcLen` is 17: [0] is the text token, [1..16] the
 * audio codes; the layout of the embedding tables is a run of [1024, 768]
 * blocks starting at `wOff`, and the text table precedes them.
 */
export const EMBED_ROW = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> rowIds: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

const HIDDEN: u32 = 768u;
const PAD_ID: u32 = 1024u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = gid.x;
  if (d >= HIDDEN) { return; }
  // text embedding table at the start of the block
  var acc = weights[p.wOff + rowIds[0] * HIDDEN + d];
  for (var k = 1u; k < p.srcLen; k = k + 1u) {
    let id = rowIds[k];
    if (id == PAD_ID) { continue; }
    // 16 audio tables of 1024x768 follow the text table
    acc = acc + weights[p.wOff + 16384u * HIDDEN + (k - 1u) * 1024u * HIDDEN + id * HIDDEN + d];
  }
  scratch[p.dOff + d] = acc;
}
`;

/**
 * LayerNorm over `p.n` elements: y = (x - mean) / sqrt(var + eps) * g + b.
 *
 * `p.n` (768) exceeds the workgroup size, so each lane accumulates a strided
 * slice — the harness caught the non-strided version dropping the tail.
 */
export const LAYER_NORM = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let base = p.dOff;
  var acc = 0.0;
  for (var i = lid.x; i < p.n; i = i + WG) { acc = acc + x[base + i]; }
  partial[lid.x] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { partial[lid.x] = partial[lid.x] + partial[lid.x + s]; }
    workgroupBarrier();
  }
  let mean = partial[0] / f32(p.n);
  var sq = 0.0;
  for (var i = lid.x; i < p.n; i = i + WG) { let d = x[base + i] - mean; sq = sq + d * d; }
  partial[lid.x] = sq;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { partial[lid.x] = partial[lid.x] + partial[lid.x + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(partial[0] / f32(p.n) + 1e-5);
  for (var i = lid.x; i < p.n; i = i + WG) {
    let g = weights[p.wOff + i];
    let b = weights[p.bOff + i];
    scratch[base + i] = (x[base + i] - mean) * inv * g + b;
  }
}
`;

/**
 * Rotary embedding, measured convention (docs §10.5):
 *   out[d] = v[d]*cos(pos*f) + partner*sin(pos*f),
 *   partner = -v[d+1] for even d, +v[d-1] for odd d,
 *   f = 10000^(-2*(d>>1)/64)  — pairs (2i, 2i+1) share one frequency.
 * Operates in place on `scratch[dOff .. dOff+768)`, head-major [12, 64].
 */
export const ROPE_INPLACE = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = gid.x;
  if (d >= 64u) { return; }
  let base = p.dOff + d;
  let f = pow(10000.0, -2.0 * f32(d >> 1u) / 64.0);
  let c = cos(f32(p.srcLen) * f);
  let s = sin(f32(p.srcLen) * f);
  var partner = 0.0;
  if ((d & 1u) == 1u) { partner = x[p.dOff + d - 1u]; } else { partner = -x[p.dOff + d + 1u]; }
  scratch[base] = x[base] * c + partner * s;
}
`;

/** GELU (tanh approximation), elementwise, in place. */
export const GELU = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let v = x[p.dOff + i];
  let inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
  scratch[p.dOff + i] = 0.5 * v * (1.0 + tanh(inner));
}
`;

/** Residual add: dst += src, `p.k` elements, p.dOff is the destination base. */
export const ADD_RESIDUAL = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  scratch[p.dOff + i] = scratch[p.dOff + i] + x[p.k + i];
}
`;

export const KERNELS = {
  gemv: GEMV,
  embedRow: EMBED_ROW,
  layerNorm: LAYER_NORM,
  rope: ROPE_INPLACE,
  gelu: GELU,
  addResidual: ADD_RESIDUAL,
} as const;

export type KernelName = keyof typeof KERNELS;
