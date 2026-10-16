/**
 * Single-row WGSL kernels: the global decode step, the 17-step local head and
 * the codec decoder. The shapes are the ones measured out of the published
 * graphs (docs/moss-tts-runtime-analysis.md §10.5–§10.7); the batched prefill
 * twins live in `kernels-batch.ts`.
 *
 * Every kernel shares ONE uniform layout and ONE bind group layout, so a
 * pipeline never has to be re-created when the buffers rotate:
 *
 *   0 uniforms  1 input (read_write)  2 weights (read)  3 output (read_write)
 *
 * The attention kernels are the exception — they need Q, the ring and the
 * context at once — and use the 5- and 6-binding layouts in `ATTEND_LAYOUT` /
 * `ATTEND_CODEC_LAYOUT`.
 *
 * The uniform block carries the 16 per-channel audio-table offsets after the
 * scalar fields (`E0…E15`): the weight upload places each tensor on a block
 * boundary inside its chunk, so the embedding tables are *not* at a fixed
 * distance from the text table — §10.2 measured the sidecar interleaving them.
 */

/**
 * 12 scalar fields + 16 table offsets = 28 u32 = 112 B, padded to 192 B so the
 * block stays 16-byte aligned and leaves room for the next field.
 */
const DECODE_PRELUDE = /* wgsl */ `
struct Params {
  n: u32,       // width of this call (weight block N / activation length)
  k: u32,       // weight block K (GEMM depth)
  wOff: u32,    // element offset of the weight block in the weights binding
  bOff: u32,    // element offset of the bias block
  dOff: u32,    // element offset of the destination in the output binding
  sOff: u32,    // element offset of the source in the input binding
  aOff: u32,    // aux: secondary input/output offset
  cOff: u32,    // aux2: third offset (cache base, seen mask, …)
  p0: u32,      // per-kernel scalars: position, valid length, channel, …
  p1: u32,
  p2: u32,
  p3: u32,
  // audio-embedding table offsets, in elements inside the weights binding
  e0: u32, e1: u32, e2: u32, e3: u32, e4: u32, e5: u32, e6: u32, e7: u32,
  e8: u32, e9: u32, e10: u32, e11: u32, e12: u32, e13: u32, e14: u32, e15: u32,
}
`;

const DECODE_BINDINGS = /* wgsl */ `
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;
/**
 * The bias/LayerNorm-bias parameter is bound separately from the weight block
 * because the two routinely live in DIFFERENT chunks: the sidecar keeps every
 * bias and LayerNorm parameter in one contiguous block near the front, while
 * the big projections follow the embedding tables, so a 128 MB chunk boundary
 * falls between them. Reading p.bOff out of binding 2 reads the wrong chunk
 * (and layer 0 is the only case where it accidentally works).
 */
@group(0) @binding(4) var<storage, read> bias: array<f32>;

/** the 16 audio-embedding tables, indexed by channel */
fn audioTable(channel: u32) -> u32 {
  switch channel {
    case 0u: { return p.e0; }
    case 1u: { return p.e1; }
    case 2u: { return p.e2; }
    case 3u: { return p.e3; }
    case 4u: { return p.e4; }
    case 5u: { return p.e5; }
    case 6u: { return p.e6; }
    case 7u: { return p.e7; }
    case 8u: { return p.e8; }
    case 9u: { return p.e9; }
    case 10u: { return p.e10; }
    case 11u: { return p.e11; }
    case 12u: { return p.e12; }
    case 13u: { return p.e13; }
    case 14u: { return p.e14; }
    default: { return p.e15; }
  }
}
`;

/**
 * Embed into the residual stream. `p.aOff` selects the mode:
 *   0  a 17-wide row at `p.sOff`:  wte[row[0]] + Σ_{k=p1}^{p2} audioEmb_k[row[1+k]]
 *      (audio_pad_token_id contributes nothing), with `p.wOff` = the text table
 *   1  wte[p.p1] — the assistant slot the local head re-embeds each frame
 *   2  audioEmb_{p0}[ tokens[p1] ] — channel p0 continuing from channel p0−1
 *
 * In mode 0 the audio range `[p1, p2)` exists because the 16 tables are 3 MB
 * each and the weight upload rolls over between them: each chunk dispatches its
 * own contiguous range and `p3` makes it accumulate into the same destination.
 * (The range is contiguous because the tables sit in sidecar order.)
 */
export const EMBED_ROW_LOCAL = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}
const HIDDEN: u32 = 768u;
const PAD_ID: u32 = 1024u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = gid.x;
  if (d >= HIDDEN) { return; }
  var acc = 0.0;
  if (p.aOff == 0u) {
    acc = w[p.wOff + u32(src[p.sOff]) * HIDDEN + d];
    for (var k = p.p1; k <= p.p2; k = k + 1u) {
      let id = u32(src[p.sOff + 1u + k]);
      if (id == PAD_ID) { continue; }
      acc = acc + w[audioTable(k) + id * HIDDEN + d];
    }
  } else if (p.aOff == 1u) {
    acc = w[p.wOff + p.p1 * HIDDEN + d];
  } else {
    // the previous channel's token is already on the device, which is what keeps
    // the 16-step chain off the CPU
    acc = w[audioTable(p.p0) + u32(src[p.p1]) * HIDDEN + d];
  }
  if (p.p3 == 1u) { dst[p.dOff + d] = dst[p.dOff + d] + acc; }
  else { dst[p.dOff + d] = acc; }
}
`;

/** y[n] = bias[n] + Σ_k x[k]·W[k·N + n], one 64-lane tile per output block. */
export const GEMV = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = wid.x * 64u + lid.x;
  if (n >= p.n) { return; }
  let col = p.wOff + n;
  var acc = bias[p.bOff + n];
  for (var i = 0u; i < p.k; i = i + 1u) {
    acc = acc + src[p.sOff + i] * w[col + i * p.n];
  }
  dst[p.dOff + n] = acc;
}
`;

/** The same projection without a bias — the codec decoder's are bias-free. */
export const GEMV_NOB = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = wid.x * 64u + lid.x;
  if (n >= p.n) { return; }
  let col = p.wOff + n;
  var acc = 0.0;
  for (var i = 0u; i < p.k; i = i + 1u) {
    acc = acc + src[p.sOff + i] * w[col + i * p.n];
  }
  dst[p.dOff + n] = acc;
}
`;

/**
 * LayerNorm over `p.n` elements: y = (x − mean)/√(var + 1e-5)·g + b.
 * `p.n` (768, or 256 in the codec) exceeds the workgroup size, so each lane
 * accumulates a strided slice — the harness caught the non-strided version
 * silently dropping the tail on the first run.
 */
export const LAYERNORM_1 = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}
const WG: u32 = 256u;
var<workgroup> partial: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  var acc = 0.0;
  for (var i = lid.x; i < p.n; i = i + WG) { acc = acc + src[p.sOff + i]; }
  partial[lid.x] = acc;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { partial[lid.x] = partial[lid.x] + partial[lid.x + s]; }
    workgroupBarrier();
  }
  let mean = partial[0] / f32(p.n);
  workgroupBarrier();

  var sq = 0.0;
  for (var i = lid.x; i < p.n; i = i + WG) { let d = src[p.sOff + i] - mean; sq = sq + d * d; }
  partial[lid.x] = sq;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) { partial[lid.x] = partial[lid.x] + partial[lid.x + s]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(partial[0] / f32(p.n) + 1e-5);
  workgroupBarrier();

  for (var i = lid.x; i < p.n; i = i + WG) {
    dst[p.dOff + i] = (src[p.sOff + i] - mean) * inv * w[p.wOff + i] + bias[p.bOff + i];
  }
}
`;

/**
 * RoPE + KV write for one row of the fused [2304] QKV buffer, head-major:
 *   out[d] = v[d]·cos(pos·f) + partner·sin(pos·f),
 *   partner = −v[d+1] (even d), +v[d−1] (odd d),  f = 10000^(−2(d>>1)/64)
 * (the GPT-J interleaving with the negation on the even element — §10.5).
 *
 * Q is roped **in place** in the QKV row, K is roped into the key cache at
 * `p.sOff`, V (un-rotated) into the value cache at `p.aOff`. `p.p0` is the ring
 * slot: the caller passes the absolute position, which is also the slot because
 * neither ring wraps — the global one is sized prefill_rows + max_new_frames, the
 * local one has exactly 17 slots.
 */
export const ROPE_SPLIT_RING = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}
const HIDDEN: u32 = 768u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let head = wid.x % 12u;
  let d = lid.x;
  let f = pow(10000.0, -2.0 * f32(d >> 1u) / 64.0);
  let c = cos(f32(p.p0) * f);
  let s = sin(f32(p.p0) * f);

  let qBase = head * 64u + d;
  var qPartner = 0.0;
  if ((d & 1u) == 1u) { qPartner = src[qBase - 1u]; } else { qPartner = -src[qBase + 1u]; }
  src[qBase] = src[qBase] * c + qPartner * s;

  let kBase = HIDDEN + head * 64u + d;
  var kPartner = 0.0;
  if ((d & 1u) == 1u) { kPartner = src[kBase - 1u]; } else { kPartner = -src[kBase + 1u]; }
  dst[p.sOff + p.p0 * HIDDEN + qBase] = src[kBase] * c + kPartner * s;

  dst[p.aOff + p.p0 * HIDDEN + qBase] = src[2u * HIDDEN + qBase];
}
`;

/**
 * One query against a non-wrapping KV ring, one workgroup per head, 64 lanes
 * (one per head dim). Three in-dispatch passes separated by workgroup barriers:
 * scores → row max / exp / sum → context. Keeping the 64-slot tile in workgroup
 * memory avoids a second score buffer; the passes are exactly the ones the
 * reference does in JS.
 *
 * Bindings: 0 uniforms, 1 the roped QKV row, 2 weights (unused), 3 the KV ring,
 * 4 the context. `p.sOff`/`p.aOff` are the key/value bases, `p.p1` the number of
 * valid slots (position + 1), `p.dOff` the context offset in binding 4.
 */
export const ATTEND_RING = /* wgsl */ `
${DECODE_PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> ring: array<f32>;
@group(0) @binding(4) var<storage, read_write> ctx: array<f32>;

const HIDDEN: u32 = 768u;
const WG: u32 = 64u;
var<workgroup> tile: array<f32, 64>;
var<workgroup> red: array<f32, 64>;

/**
 * Online (streaming) softmax. The ring can hold 570 slots and workgroup memory
 * holds 64 scores, so the row maximum and denominator cannot be known from one
 * tile: each tile is folded in with the usual max/rescale correction instead.
 * Per-tile normalisation — the tempting shortcut — is exact for a single tile
 * and silently wrong for every ring longer than 64, which is exactly what the
 * decode step needs after the first few frames.
 */
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let head = wid.x;
  let d = lid.x;
  let valid = p.p1;
  var acc = 0.0;          // this lane's output dim, running
  var runMax = -3.0e38;   // -inf, but the literal must be representable in f32
  var runSum = 0.0;

  for (var base = 0u; base < valid; base = base + WG) {
    let j = base + d;
    var s = -3.0e38;
    if (j < valid) {
      let k = p.sOff + j * HIDDEN + head * 64u;
      var dot = 0.0;
      for (var i = 0u; i < 64u; i = i + 1u) { dot = dot + q[head * 64u + i] * ring[k + i]; }
      s = dot * 0.125;
    }
    tile[d] = s;
    red[d] = s;
    workgroupBarrier();
    for (var t = 32u; t > 0u; t = t >> 1u) {
      if (d < t) { red[d] = max(red[d], red[d + t]); }
      workgroupBarrier();
    }
    let newMax = max(runMax, red[0]);
    let correction = exp(runMax - newMax);
    workgroupBarrier();

    var e = 0.0;
    if (j < valid) { e = exp(s - newMax); }
    tile[d] = e;
    red[d] = e;
    workgroupBarrier();
    for (var t = 32u; t > 0u; t = t >> 1u) {
      if (d < t) { red[d] = red[d] + red[d + t]; }
      workgroupBarrier();
    }
    runSum = runSum * correction + red[0];
    runMax = newMax;
    workgroupBarrier();

    // each lane owns one output dim, so it walks every slot of the tile
    let stop = min(base + WG, valid);
    acc = acc * correction;
    for (var jj = base; jj < stop; jj = jj + 1u) {
      acc = acc + tile[jj - base] * ring[p.aOff + jj * HIDDEN + head * 64u + d];
    }
    workgroupBarrier();
  }
  ctx[p.dOff + head * 64u + d] = acc / runSum;
}
`;

/** dst += src (the residuals run on the hidden state, not the query). */
export const ADD_RESIDUAL = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  dst[p.dOff + i] = dst[p.dOff + i] + src[p.sOff + i];
}
`;

/** dst = dst·a + src·b — the codec decoder's LayerScale residual. */
export const SCALE_ADD = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  dst[p.dOff + i] = dst[p.dOff + i] * p.p0 + src[p.sOff + i] * p.p1;
}
`;

/**
 * `dst[i] += w[wOff + i] · src[i]` — the codec decoder's LayerScale residual,
 * with the per-channel scale read straight out of the weight chunk (it is a
 * learned [256] vector, not a scalar).
 */
export const LAYER_SCALE_ADD = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  dst[p.dOff + i] = dst[p.dOff + i] + w[p.wOff + i] * src[p.sOff + i];
}
`;

/** GELU, tanh approximation — the global transformer's activation. */
export const GELU_TANH = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let v = dst[p.dOff + i];
  let inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
  dst[p.dOff + i] = 0.5 * v * (1.0 + tanh(inner));
}
`;

/**
 * GELU, exact `0.5·x·(1+erf(x/√2))` — what the codec decoder's ffn.1 is: the
 * graph builds it from Div(√2) → Erf → +1 → ×x → ×0.5. WGSL has no erf, so this
 * is Abramowitz & Stegun 7.1.26 (|ε| ≤ 1.5e-7, inside f32 rounding of it).
 */
export const GELU_ERF = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

fn erf(x: f32) -> f32 {
  let a1 = 0.254829592;
  let a2 = -0.284496736;
  let a3 = 1.421413741;
  let a4 = -1.453152027;
  let a5 = 1.061405429;
  let p = 0.3275911;
  let sign = select(-1.0, 1.0, x >= 0.0);
  let ax = abs(x);
  let t = 1.0 / (1.0 + p * ax);
  let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
  return sign * y;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let x = dst[p.dOff + i];
  dst[p.dOff + i] = 0.5 * x * (1.0 + erf(x * 0.7071067811865476));
}
`;

/**
 * Greedy argmax over `p.n` logits with the HuggingFace repetition penalty applied
 * to channel `p.p1`'s seen-set — a 1024-bit mask at binding 3 + `p.cOff` (32
 * u32 words). Ties resolve to the lowest id, matching the reference's
 * `if (s > best)`. The winner is written as an f32 *value* into `src[p.p2]` (the
 * token buffer the next local step embeds from) and its bit is set in the mask.
 * `p.p0` is the penalty — which the reference only applies once a channel has
 * emitted anything, and a zeroed mask reproduces that exactly.
 */
export const ARGMAX_PENALTY = /* wgsl */ `
${DECODE_PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
// binding 3 carries the 1024 logits; the seen-mask is its own binding because
// it is u32 while the logits are f32
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read_write> seen: array<u32>;

const WG: u32 = 256u;
var<workgroup> bestVal: array<f32, WG>;
var<workgroup> bestIdx: array<u32, WG>;
var<workgroup> red: array<f32, WG>;
var<workgroup> redi: array<u32, WG>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  var value = -1e30;
  var index = 0u;
  for (var i = lid.x; i < p.n; i = i + WG) {
    var s = logits[p.dOff + i];
    if ((seen[p.cOff + i / 32u] & (1u << (i % 32u))) != 0u) {
      // the uniform is u32, so a fractional penalty travels as float BITS:
      // truncating 1.2 to an integer gives 1, which disables the penalty
      let penalty = bitcast<f32>(p.p0);
      s = select(s * penalty, s / penalty, s >= 0.0);
    }
    // strict >: a tie keeps the earlier (lower) index, as the reference does
    if (s > value) { value = s; index = i; }
  }
  bestVal[lid.x] = value;
  bestIdx[lid.x] = index;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      let other = bestVal[lid.x + s];
      if (other > bestVal[lid.x] || (other == bestVal[lid.x] && bestIdx[lid.x + s] < bestIdx[lid.x])) {
        bestVal[lid.x] = other;
        bestIdx[lid.x] = bestIdx[lid.x + s];
      }
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    src[p.p2] = f32(bestIdx[0]);
    seen[p.cOff + bestIdx[0] / 32u] = seen[p.cOff + bestIdx[0] / 32u] | (1u << (bestIdx[0] % 32u));
  }
}
`;

/**
 * The local head's continue/stop decision: two dot products of the step-0 hidden
 * state against columns `p.p0` (audio_end) and `p.p1` (assistant_slot) of the
 * 16 384-wide text head. Only those two columns are ever needed, so this reads
 * 2 × 768 floats of a 12.6 M-element block instead of running a GEMV.
 * Writes 1 for "continue" into binding 3 at `p.dOff`.
 */
export const TEXT_HEAD_DECIDE = /* wgsl */ `
${DECODE_PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> src: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;

var<workgroup> red: array<f32, 256>;
var<workgroup> redi: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  var end = 0.0;
  var asst = 0.0;
  for (var i = lid.x; i < 768u; i = i + 256u) {
    end = end + src[p.sOff + i] * w[p.wOff + i * 16384u + p.p0];
    asst = asst + src[p.sOff + i] * w[p.wOff + i * 16384u + p.p1];
  }
  red[lid.x] = end;
  redi[lid.x] = asst;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      red[lid.x] = red[lid.x] + red[lid.x + s];
      redi[lid.x] = redi[lid.x] + redi[lid.x + s];
    }
    workgroupBarrier();
  }
  if (lid.x == 0u) { out[p.dOff] = select(0.0, 1.0, redi[0] >= red[0]); }
}
`;

// ----------------------------------------------------------------- the codec

/**
 * The RVQ front, the codec decoder's stage-0 input. Two dispatches, because the
 * 16-quantiser sum must finish before the 768-wide output projection reads it:
 *
 *   acc[512] = Σ_k (W_k[512,8]·codebook_k[codes[k]] + b_k)
 *   g[768]   = W_out[768,512]·acc + b_out
 *   x[4s+m]  = g[4m+s]                        — interleaved ×4 sub-pixel
 *
 * `p.sOff` points at this frame's codes (16 f32 values read as `codes[k]`), and
 * at +16 a descriptor block of `biasOff[16]` then `codebookOff[16]` as u32 — the
 * descriptors are unavoidable because the sidecar interleaves each quantiser's
 * bias with its codebook, so there is no contiguous run to point one offset at.
 * `p.wOff` = the W_k run (16 × [512,8]), `p.aOff` = W_out [768,512], `p.cOff` =
 * b_out, `p.dOff` = destination.
 */
export const RVQ_ACCUM = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}
const DESC: u32 = 16u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = wid.x * 64u + lid.x;
  if (n >= 512u) { return; }
  var acc = 0.0;
  for (var k = 0u; k < 16u; k = k + 1u) {
    let code = u32(src[p.sOff + k]);
    // the descriptors are element offsets the runner staged next to the codes;
    // bitcast, not a value conversion, so an offset never becomes a NaN
    let entry = bitcast<u32>(src[p.sOff + DESC + k]);
    let book = bitcast<u32>(src[p.sOff + DESC + 16u + k]);
    let row = p.wOff + k * 512u * 8u + n * 8u;
    var dot = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) { dot = dot + w[row + j] * w[book + code * 8u + j]; }
    acc = acc + dot + w[entry + n];
  }
  dst[p.dOff + n] = acc;
}
`;

export const RVQ_PROJECT = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let n = wid.x * 64u + lid.x;
  if (n >= 768u) { return; }
  var acc = bias[p.cOff + n];
  for (var i = 0u; i < 512u; i = i + 1u) { acc = acc + src[p.sOff + i] * w[p.aOff + n * 512u + i]; }
  // interleaved ×4 sub-pixel expansion: position (4t + s), channel m
  dst[p.dOff + (n & 3u) * 192u + (n >> 2u)] = acc;
}
`;

/**
 * Codec attention over a **wrapping** ring cache, one workgroup per head (4
 * heads of 64 dims at d=256), for one query position at a time. Unlike the
 * global ring this one does wrap, so validity is decided per slot from the
 * stored absolute position:
 *
 *   valid_j = (stored_j ≥ 0) ∧ (0 ≤ cur − stored_j < C)
 *
 * which is the shift-append the graph spells as concat + mask (§9.1). The scale
 * is 0.125 = 1/√64 — the graph scales q and k by √0.125 each, whose product is
 * the same. `p.p0` = query position, `p.p1` = context C, `p.cOff` = key base,
 * `p.aOff` = value base, `p.dOff` = context destination.
 */
export const ATTEND_RING_CODEC = /* wgsl */ `
${DECODE_PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> q: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> ring: array<f32>;
@group(0) @binding(4) var<storage, read_write> ctx: array<f32>;
@group(0) @binding(5) var<storage, read_write> slots: array<i32>;

const D: u32 = 256u;
const HD: u32 = 64u;
const WG: u32 = 64u;
var<workgroup> tile: array<f32, 64>;
var<workgroup> red: array<f32, 64>;

/** The streaming softmax of ATTEND_RING, over a wrapping, position-masked ring. */
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let head = wid.x;
  let d = lid.x;
  let context = p.p1;
  let cur = i32(p.p0);
  var acc = 0.0;
  var runMax = -3.0e38;
  var runSum = 0.0;

  for (var base = 0u; base < context; base = base + WG) {
    let j = base + d;
    // Guard against the tail: context is not always a multiple of 64 (500,
    // 800, 1200 …), and an unguarded read past this layer's slot region lands
    // in the next layer's slots — whose live positions would then be attended.
    let inWindow = j < context;
    let delta = cur - slots[p.sOff + j];
    let live = inWindow && slots[p.sOff + j] >= 0 && delta >= 0 && u32(delta) < context;
    var s = -3.0e38;
    if (live) {
      let k = p.cOff + j * D + head * HD;
      var dot = 0.0;
      for (var i = 0u; i < HD; i = i + 1u) { dot = dot + q[head * HD + i] * ring[k + i]; }
      s = dot * 0.125;
    }
    tile[d] = s;
    red[d] = s;
    workgroupBarrier();
    for (var t = 32u; t > 0u; t = t >> 1u) {
      if (d < t) { red[d] = max(red[d], red[d + t]); }
      workgroupBarrier();
    }
    let newMax = max(runMax, red[0]);
    let correction = exp(runMax - newMax);
    workgroupBarrier();

    var e = 0.0;
    if (live) { e = exp(s - newMax); }
    tile[d] = e;
    red[d] = e;
    workgroupBarrier();
    for (var t = 32u; t > 0u; t = t >> 1u) {
      if (d < t) { red[d] = red[d] + red[d + t]; }
      workgroupBarrier();
    }
    runSum = runSum * correction + red[0];
    runMax = newMax;
    workgroupBarrier();

    let stop = min(base + WG, context);
    acc = acc * correction;
    for (var jj = base; jj < stop; jj = jj + 1u) {
      let delta2 = cur - slots[p.sOff + jj];
      if (slots[p.sOff + jj] >= 0 && delta2 >= 0 && u32(delta2) < context) {
        acc = acc + tile[jj - base] * ring[p.aOff + jj * D + head * HD + d];
      }
    }
    workgroupBarrier();
  }
  ctx[p.dOff + head * HD + d] = acc / runSum;
}
`;

/**
 * RoPE + ring write for one codec query position: takes the fused [768] qkv row
 * for that position, ropes q and k at the **absolute** position `p.p0`, writes k
 * to ring slot `pos % C` and v to the value ring, and records the slot's
 * absolute position in `slots` (binding 5). `p.p1` = context C, `p.cOff` = key
 * base, `p.aOff` = value base, `p.sOff` = slot-array offset, `p.dOff` = the
 * roped-q destination (one row per position in the context buffer).
 *
 * The qkv row keeps its roped q in place, so the attention reads binding 1.
 */
export const ROPE_SPLIT_CODEC = /* wgsl */ `
${DECODE_PRELUDE}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> qkv: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> ring: array<f32>;
@group(0) @binding(4) var<storage, read_write> ctx: array<f32>;
@group(0) @binding(5) var<storage, read_write> slots: array<i32>;

const D: u32 = 256u;
const HD: u32 = 64u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let head = wid.x % 4u;
  let d = lid.x;
  let pos = p.p0;
  let f = pow(10000.0, -2.0 * f32(d >> 1u) / 64.0);
  let c = cos(f32(pos) * f);
  let s = sin(f32(pos) * f);

  let qBase = head * HD + d;
  var qPartner = 0.0;
  if ((d & 1u) == 1u) { qPartner = qkv[qBase - 1u]; } else { qPartner = -qkv[qBase + 1u]; }
  qkv[qBase] = qkv[qBase] * c + qPartner * s;
  ctx[qBase] = qkv[qBase];

  let kBase = D + head * HD + d;
  var kPartner = 0.0;
  if ((d & 1u) == 1u) { kPartner = qkv[kBase - 1u]; } else { kPartner = -qkv[kBase + 1u]; }
  let slot = pos % p.p1;
  ring[p.cOff + slot * D + head * HD + d] = qkv[kBase] * c + kPartner * s;
  ring[p.aOff + slot * D + head * HD + d] = qkv[2u * D + head * HD + d];
  slots[p.sOff + slot] = i32(pos);
}
`;

/**
 * Interleaved ×2 sub-pixel upsample between codec stages, measured in §10.6:
 *   new[2t + r, m] = old[t, 2m + r]
 * `p.n` = time positions, `p.p0` = half the channel count (`p.p0·2` in, `p.p0`
 * out), `p.sOff`/`p.dOff` element offsets.
 */
export const UPSAMPLE2 = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

// Interleaved ×2 sub-pixel upsample: new[2t + r, m] = old[t, 2m + r].
// The input is [n, outWidth*2] and the output is [2n, outWidth], both row-major;
// p.p0 is outWidth. (Convention verified against the graph in probe-up.mjs.)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let outWidth = p.p0;
  let row = gid.x / outWidth;
  let m = gid.x % outWidth;
  if (row >= p.n * 2u) { return; }
  let t = row / 2u;
  let r = row % 2u;
  dst[p.dOff + row * outWidth + m] = src[p.sOff + t * (outWidth * 2u) + 2u * m + r];
}
`;

/**
 * The codec head is a pure layout transform (§10.6, verified maxErr 0):
 *   audio[c][t·120 + j] = latent[t][2j + c]
 * `p.n` = latent positions, `p.sOff` = latent source, `p.dOff` = destination
 * holding 2 channels × 120·n samples, channel-blocks in order.
 */
export const CODEC_HEAD = /* wgsl */ `
${DECODE_PRELUDE}
${DECODE_BINDINGS}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let per = 120u;
  let c = gid.x / (per * p.n);
  let rest = gid.x % (per * p.n);
  let t = rest / per;
  let j = rest % per;
  dst[p.dOff + gid.x] = src[p.sOff + t * 240u + 2u * j + c];
}
`;
