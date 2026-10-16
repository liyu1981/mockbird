/**
 * Batched WGSL kernels: everything the 195-row prefill needs (S > 1).
 *
 * The single-row kernels in `kernels.ts` are decode-path; these are the prefill
 * shapes, all taken from the measured conventions (§10.5): LayerNorm eps 1e-5,
 * RoPE interleaved with the negation on the even element, attention scale
 * 0.125, GELU tanh approximation, residuals on the hidden state (not the query).
 *
 * Layouts (all row-major f32):
 *   ids    [S, 17]  u32
 *   x      [S, 768]
 *   qkv    [S, 2304] = q | k | v, head-major inside each 768-wide block
 *   kv     [(layer*2 + isV) * maxSlots * 768 + slot*768 + head*64 + dim]
 *   scores [heads, S, S]
 */

/** One uniform block: shapes + weight/activation offsets, 16-byte aligned. */
const PRELUDE = /* wgsl */ `
struct Params {
  rows: u32,
  cols: u32,      // N of the weight block / feature width
  depth: u32,     // K of the weight block
  wOff: u32,      // element offset of the weight block inside the weights binding
  bOff: u32,      // element offset of the bias block
  yOff: u32,      // element offset of the destination in the output binding
  xOff: u32,      // element offset of the source in the input binding
  aux: u32,       // secondary offset (attention scratch / value cache)
  aux2: u32,
  aux3: u32,
  // per-channel audio embedding table offsets, element index in the weight
  // block. Explicit scalars, not an array: a WGSL array in uniform space needs a
  // 16-byte stride and would not match the packed uniform block. The sidecar
  // interleaves LayerNorm parameters, so the audio tables do NOT sit at a fixed
  // distance from the text table.
  a0: u32, a1: u32, a2: u32, a3: u32, a4: u32, a5: u32, a6: u32, a7: u32,
  a8: u32, a9: u32, a10: u32, a11: u32, a12: u32, a13: u32, a14: u32, a15: u32,
}
`;

/**
 * Input embedding for a whole prompt:
 *   x[s] = wte[ids[s,0]] + sum_k audioEmb_k[ids[s,1+k]]   (pad channels skipped)
 * `weights` holds the text table first, then the 16 audio tables.
 */
export const EMBED_ROWS = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> ids: array<u32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

const HIDDEN: u32 = 768u;
const PAD_ID: u32 = 1024u;
const WG: u32 = 256u;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let s = wid.x;
  if (s >= p.rows) { return; }
  let row = s * 17u;
  // 768 > 256 lanes, so each lane walks a strided slice of the hidden state
  for (var d = lid.x; d < HIDDEN; d = d + WG) {
    var off = p.wOff + ids[row] * HIDDEN + d;
    var acc = weights[off];
    for (var k = 1u; k < 17u; k = k + 1u) {
      let id = ids[row + k];
      if (id == PAD_ID) { continue; }
      var table = 0u;
      switch k {
        case 1u: { table = p.a0; }
        case 2u: { table = p.a1; }
        case 3u: { table = p.a2; }
        case 4u: { table = p.a3; }
        case 5u: { table = p.a4; }
        case 6u: { table = p.a5; }
        case 7u: { table = p.a6; }
        case 8u: { table = p.a7; }
        case 9u: { table = p.a8; }
        case 10u: { table = p.a9; }
        case 11u: { table = p.a10; }
        case 12u: { table = p.a11; }
        case 13u: { table = p.a12; }
        case 14u: { table = p.a13; }
        case 15u: { table = p.a14; }
        default: { table = p.a15; }
      }
      acc = acc + weights[table + id * HIDDEN + d];
    }
    scratch[p.yOff + s * HIDDEN + d] = acc;
  }
}
`;

/**
 * Row-wise LayerNorm (eps 1e-5). One workgroup per row, 256 lanes, strided
 * reduction: a 768-wide row exceeds the workgroup size, and the harness already
 * caught the non-strided version silently dropping the tail.
 */
export const LAYERNORM_ROWS = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;   // see GEMM_NT

const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wid.x;
  if (row >= p.rows) { return; }
  let src = p.xOff + row * p.cols;
  let dst = p.yOff + row * p.cols;

  // mean
  var acc = 0.0;
  for (var i = lid.x; i < p.cols; i = i + WG) { acc = acc + x[src + i]; }
  partial[lid.x] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { partial[lid.x] = partial[lid.x] + partial[lid.x + s]; }
    workgroupBarrier();
  }
  let mean = partial[0] / f32(p.cols);
  // every lane must have read partial[0] before the next pass overwrites it —
  // without this barrier the variance reduction races the mean read
  workgroupBarrier();

  // variance
  var sq = 0.0;
  for (var i = lid.x; i < p.cols; i = i + WG) { let d = x[src + i] - mean; sq = sq + d * d; }
  partial[lid.x] = sq;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { partial[lid.x] = partial[lid.x] + partial[lid.x + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(partial[0] / f32(p.cols) + 1e-5);
  workgroupBarrier();

  for (var i = lid.x; i < p.cols; i = i + WG) {
    scratch[dst + i] = (x[src + i] - mean) * inv * weights[p.wOff + i] + bias[p.bOff + i];
  }
}
`;

/**
 * y[m][n] = bias[n] + sum_k x[m][k] * W[k][n]   (W is [K, N] row-major, as the
 * ONNX initializers already are — no transposition anywhere).
 *
 * One workgroup per (row, 64-wide output tile); the 64 lanes each own one output
 * column and stream K rows of W, so the weight block is read exactly once.
 */
export const GEMM_NT = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;
// the bias is a separate binding because a weight block and its bias routinely
// land in DIFFERENT chunks: the sidecar keeps every bias and LayerNorm parameter
// in one 50 MB block up front, while the [768,2304] matrices follow the
// embedding tables. Reading p.bOff out of binding 2 then reads the wrong chunk
// (and layer 0 was the only layer where it happened to work)
@group(0) @binding(4) var<storage, read> bias: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let tile = wid.x;
  let m = wid.y;
  if (m >= p.rows) { return; }
  let n = tile * 64u + lid.x;
  if (n >= p.cols) { return; }
  let xRow = p.xOff + m * p.depth;
  let wCol = p.wOff + n;
  var acc = bias[p.bOff + n];
  for (var k = 0u; k < p.depth; k = k + 1u) {
    acc = acc + x[xRow + k] * weights[wCol + k * p.cols];
  }
  scratch[p.yOff + m * p.cols + n] = acc;
}
`;

/**
 * Split the fused QKV row into roped Q, and write K (post-RoPE) and V into the
 * KV cache.
 *
 * RoPE (measured): out[d] = v[d]*cos(pos*f) + partner*sin(pos*f) with
 * partner = -v[d+1] for even d, +v[d-1] for odd d, and f = 10000^(-2*(d>>1)/64),
 * applied per 64-wide head. The cache layout is head-major [12, 64] per slot.
 */
export const ROPE_SPLIT_CACHE = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
// read_write: the roped Q is written back into the fused QKV row in place
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> kv: array<f32>;

const HD: u32 = 64u;
const HIDDEN: u32 = 768u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let row = wid.x / 12u;
  let head = wid.x % 12u;
  let d = lid.x;
  if (d >= HD) { return; }
  let base = row * 2304u + head * HD + d;

  // Q: rope in place
  let f = pow(10000.0, -2.0 * f32(d >> 1u) / 64.0);
  let c = cos(f32(row) * f);
  let s = sin(f32(row) * f);
  var qPartner = 0.0;
  if ((d & 1u) == 1u) { qPartner = qkv[row * 2304u + head * HD + d - 1u]; }
  else { qPartner = -qkv[row * 2304u + head * HD + d + 1u]; }
  let q = qkv[base] * c + qPartner * s;
  qkv[base] = q;

  // K: same rotation, into the cache
  let kBase = row * 2304u + HIDDEN + head * HD + d;
  var kPartner = 0.0;
  if ((d & 1u) == 1u) { kPartner = qkv[row * 2304u + HIDDEN + head * HD + d - 1u]; }
  else { kPartner = -qkv[row * 2304u + HIDDEN + head * HD + d + 1u]; }
  let k = qkv[kBase] * c + kPartner * s;
  kv[p.xOff + row * HIDDEN + head * HD + d] = k;

  // V: no rotation, into the cache
  kv[p.aux + row * HIDDEN + head * HD + d] = qkv[row * 2304u + 2u * HIDDEN + head * HD + d];
}
`;

/**
 * Causal attention scores for one (head, 64-query tile): scale by 1/sqrt(64) and
 * write -inf for j > q. Row max is kept in `scratch` for the softmax pass.
 *
 * The query is read from the FUSED QKV row, so the row stride is 2304, not 768:
 * `ropeSplitCache` leaves the roped query at `row·2304 + head·64 + dim` and the
 * key/value live in the separate ring. Reading with a 768 stride silently
 * returns the *key* block for every row after the first, which is why only
 * row 0 looked right.
 */
export const ATTENTION_SCORES = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

const HD: u32 = 64u;
const HIDDEN: u32 = 768u;
const QKV: u32 = 2304u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let head = wid.x;
  let q0 = wid.y * 64u + lid.x;
  if (q0 >= p.rows) { return; }
  let scale = 0.125;   // 1 / sqrt(64), a literal in the graph
  var best = -1e30;   // f32 min overflows the literal; the loop always runs
  for (var j = 0u; j <= q0; j = j + 1u) {
    var dot = 0.0;
    for (var d = 0u; d < HD; d = d + 1u) {
      dot = dot + q[q0 * QKV + head * HD + d] * kv[p.xOff + j * HIDDEN + head * HD + d];
    }
    let score = dot * scale;
    scratch[(p.aux + head) * p.rows * p.rows + q0 * p.rows + j] = score;
    best = max(best, score);
  }
  scratch[p.aux2 + head * p.rows + q0] = best;
}
`;

/**
 * Row softmax over the causal scores: binding 1 holds the scores (and the row
 * maxima the scoring pass left at `p.aux2`), binding 3 receives the
 * probabilities.
 *
 * The workgroup index is (head, query), so the guard is HEADS × rows — NOT
 * `p.rows * p.aux`, which is zero here: the scores live at `p.aux` = 0 in both
 * the writing and the reading pass, and using it as the bound made every
 * workgroup return before doing any work. That left the probabilities at zero,
 * the context at zero, and every prefill row wrong while the KV cache — which
 * is written by a different kernel — still matched the reference.
 */
export const ATTENTION_SOFTMAX = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> scores: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

const WG: u32 = 256u;
const HEADS: u32 = 12u;
var<workgroup> sh: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let index = wid.x;                 // head * rows + query
  if (index >= HEADS * p.rows) { return; }
  let head = index / p.rows;
  let q0 = index % p.rows;
  let base = (p.aux + head) * p.rows * p.rows + q0 * p.rows;
  let best = scores[p.aux2 + head * p.rows + q0];
  var local = 0.0;
  for (var j = lid.x; j <= q0; j = j + WG) {
    let e = exp(scores[base + j] - best);
    scratch[base + j] = e;
    local = local + e;
  }
  sh[lid.x] = local;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { sh[lid.x] = sh[lid.x] + sh[lid.x + s]; }
    workgroupBarrier();
  }
  let total = sh[0];
  workgroupBarrier();
  for (var j = lid.x; j <= q0; j = j + WG) { scratch[base + j] = scratch[base + j] / total; }
}
`;

/** out[q][d] = sum_j probs[q][j] * V[j][d], head-major. */
export const ATTENTION_APPLY = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read> kv: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

const HD: u32 = 64u;
const HIDDEN: u32 = 768u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let head = wid.x;
  let d = lid.x;
  if (d >= HD) { return; }
  for (var q0 = 0u; q0 < p.rows; q0 = q0 + 1u) {
    var acc = 0.0;
    for (var j = 0u; j <= q0; j = j + 1u) {
      acc = acc + probs[(p.aux + head) * p.rows * p.rows + q0 * p.rows + j] * kv[p.xOff + j * HIDDEN + head * HD + d];
    }
    scratch[p.yOff + q0 * HIDDEN + head * HD + d] = acc;
  }
}
`;

/** dst[m][n] += src[m][n] (residuals are on the hidden state, not the query). */
export const ADD_ROWS = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.rows * p.cols) { return; }
  scratch[p.yOff + i] = scratch[p.yOff + i] + x[p.xOff + i];
}
`;

/** GELU (tanh approximation) in place over [rows, cols]. */
export const GELU_ROWS = /* wgsl */ `
${PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.rows * p.cols) { return; }
  let v = scratch[p.yOff + i];
  let inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
  scratch[p.yOff + i] = 0.5 * v * (1.0 + tanh(inner));
}
`;

// NOTE: binding 1 is declared `read_write` in every kernel on purpose. All of
// them share ONE explicit bind group layout (see prefill.ts), whose binding 1
// must be the writable `storage` type because ropeSplitCache updates the roped Q
// in place; a `read`-only declaration would then be incompatible with it.
// Reading is a strict subset of writing, so this costs nothing.

export const BATCH_KERNELS = {
  embedRows: EMBED_ROWS,
  layerNormRows: LAYERNORM_ROWS,
  gemmNt: GEMM_NT,
  ropeSplitCache: ROPE_SPLIT_CACHE,
  attentionScores: ATTENTION_SCORES,
  attentionSoftmax: ATTENTION_SOFTMAX,
  attentionApply: ATTENTION_APPLY,
  addRows: ADD_ROWS,
  geluRows: GELU_ROWS,
} as const;

export type BatchKernelName = keyof typeof BATCH_KERNELS;
