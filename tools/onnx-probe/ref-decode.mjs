// Complete CPU reference for the MOSS-TTS-Nano greedy decode path.
//
// This is the oracle the WGSL kernels are ported from and validated against.
// Every convention below was *measured* against the published ONNX graphs
// (docs/moss-tts-runtime-analysis.md §10.5) — none of it is inferred from the
// exporter source alone.
//
//   MOCKBIRD_MODEL_DIR=<weights> node tools/onnx-probe/ref-decode.mjs [frames]
//
// Requires the decode weight set (715 MB, see the analysis doc). Compares its own
// token stream against the committed golden trace.
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { openOnnx } from "../../src/lib/tts/engine/onnxWeights.ts";

const DIR = process.env.MOCKBIRD_MODEL_DIR ?? ".";
const M = JSON.parse(readFileSync(`${DIR}/browser_poc_manifest.json`, "utf8"));
const CFG = M.tts_config,
  GEN = M.generation_defaults;
const PAD = CFG.audio_pad_token_id,
  N_VQ = CFG.n_vq;
const D = 768,
  H = 12,
  HD = 64,
  LAYERS = 12,
  F = 3072,
  EPS = 1e-5;
const FRAMES = Number(process.argv[2] ?? 40);

// ------------------------------------------------------------------ weights
const G = openOnnx(
  readFileSync(`${DIR}/moss_tts_decode_step.onnx`),
  readFileSync(`${DIR}/moss_tts_global_shared.data`),
);
const L = openOnnx(
  readFileSync(`${DIR}/moss_tts_local_cached_step.onnx`),
  readFileSync(`${DIR}/moss_tts_local_shared.data`),
);
const byShape = (P, shape) =>
  P.names()
    .map((n) => ({ n, p: P.get(n) }))
    .filter((x) => x.p.dims.join("x") === shape && x.p.offset >= 0)
    .sort((a, b) => a.p.offset - b.p.offset)
    .map((x) => x.p.data);
const GL = Array.from({ length: LAYERS }, (_, i) => ({
  cAttn: byShape(G, "768x2304")[i],
  cAttnB: G.get(`core.model.transformer.h.${i}.attn.c_attn.bias`).data,
  cProj: byShape(G, "768x768")[i],
  cProjB: G.get(`core.model.transformer.h.${i}.attn.c_proj.bias`).data,
  ln1W: G.get(`core.model.transformer.h.${i}.ln_1.weight`).data,
  ln1B: G.get(`core.model.transformer.h.${i}.ln_1.bias`).data,
  ln2W: G.get(`core.model.transformer.h.${i}.ln_2.weight`).data,
  ln2B: G.get(`core.model.transformer.h.${i}.ln_2.bias`).data,
  fcIn: byShape(G, "768x3072")[i],
  fcInB: G.get(`core.model.transformer.h.${i}.mlp.fc_in.bias`).data,
  fcOut: byShape(G, "3072x768")[i],
  fcOutB: G.get(`core.model.transformer.h.${i}.mlp.fc_out.bias`).data,
}));
const glnFW = G.get("core.model.transformer.ln_f.weight").data,
  glnFB = G.get("core.model.transformer.ln_f.bias").data;
const wte = G.get("core.model.transformer.wte.weight").data;
const AUD = Array.from(
  { length: N_VQ },
  (_, k) => G.get(`core.model.audio_embeddings.${k}.weight`).data,
);
const LOC = {
  cAttn: L.get("onnx::MatMul_770").data,
  cAttnB: L.get("model.local_transformer.h.0.attn.c_attn.bias").data,
  cProj: L.get("onnx::MatMul_803").data,
  cProjB: L.get("model.local_transformer.h.0.attn.c_proj.bias").data,
  ln1W: L.get("model.local_transformer.h.0.ln_1.weight").data,
  ln1B: L.get("model.local_transformer.h.0.ln_1.bias").data,
  ln2W: L.get("model.local_transformer.h.0.ln_2.weight").data,
  ln2B: L.get("model.local_transformer.h.0.ln_2.bias").data,
  fcIn: L.get("onnx::MatMul_804").data,
  fcInB: L.get("model.local_transformer.h.0.mlp.fc_in.bias").data,
  fcOut: L.get("onnx::MatMul_805").data,
  fcOutB: L.get("model.local_transformer.h.0.mlp.fc_out.bias").data,
  lnFW: L.get("model.local_transformer.ln_f.weight").data,
  lnFB: L.get("model.local_transformer.ln_f.bias").data,
  textHead: L.get("onnx::MatMul_806").data,
  audioHeads: Array.from({ length: N_VQ }, (_, k) => L.get(`onnx::MatMul_${807 + k}`).data),
};

// ------------------------------------------------------------------ kernels
const layerNorm = (x, w, b) => {
  let m = 0;
  for (let i = 0; i < D; i++) m += x[i];
  m /= D;
  let v = 0;
  for (let i = 0; i < D; i++) {
    const d = x[i] - m;
    v += d * d;
  }
  v /= D;
  const inv = 1 / Math.sqrt(v + EPS);
  const o = new Float32Array(D);
  for (let i = 0; i < D; i++) o[i] = (x[i] - m) * inv * (w ? w[i] : 1) + (b ? b[i] : 0);
  return o;
};
function gemv(x, W, bias, out) {
  const y = new Float32Array(out);
  for (let n = 0; n < out; n++) {
    let s = bias ? bias[n] : 0;
    for (let k = 0; k < x.length; k++) s += x[k] * W[k * out + n];
    y[n] = s;
  }
  return y;
}
const geluTanh = (v) =>
  0.5 * v * (1 + Math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)));
const invFreq = Array.from({ length: HD / 2 }, (_, i) => 10000 ** (-(2 * i) / HD));

/** RoPE, measured convention: pairs (2i, 2i+1) share freq i, out[2i] = -v[2i+1]. */
function ropeInto(out, v, pos, off) {
  for (let h = 0; h < H; h++) {
    const base = off + h * HD;
    for (let d = 0; d < HD; d++) {
      const i = base + d,
        f = invFreq[d >> 1];
      const partner = d & 1 ? v[i - 1] : -v[i + 1];
      out[i] = v[i] * Math.cos(pos * f) + partner * Math.sin(pos * f);
    }
  }
}
/** Expand a fused [2304] qkv into three head-major [12,64] vectors. */
function splitHeads(qkv) {
  const q = new Float32Array(D),
    k = new Float32Array(D),
    v = new Float32Array(D);
  for (let i = 0; i < D; i++) {
    q[i] = qkv[i];
    k[i] = qkv[768 + i];
    v[i] = qkv[1536 + i];
  }
  return { q, k, v };
}
/** Attention of one query against a KV cache, head-major, scale 0.125. */
function attend(q, kCache, vCache, valid, validLen) {
  const out = new Float32Array(D);
  const scores = new Float64Array(valid);
  for (let h = 0; h < H; h++) {
    const off = h * HD;
    let mx = -Infinity;
    for (let j = 0; j < valid; j++) {
      let s = 0;
      for (let d = 0; d < HD; d++) s += q[off + d] * kCache[j][off + d];
      scores[j] = j < validLen ? s * 0.125 : -Infinity;
      if (scores[j] > mx) mx = scores[j];
    }
    let sum = 0;
    for (let j = 0; j < valid; j++) {
      scores[j] = Math.exp(scores[j] - mx);
      sum += scores[j];
    }
    for (let j = 0; j < valid; j++) {
      const p = scores[j] / sum;
      for (let d = 0; d < HD; d++) out[off + d] += p * vCache[j][off + d];
    }
  }
  return out;
}
const embedRow = (row) => {
  const x = new Float32Array(D);
  for (let j = 0; j < D; j++) x[j] = wte[row[0] * D + j];
  for (let k = 0; k < N_VQ; k++) {
    if (row[k + 1] === PAD) continue;
    const e = AUD[k],
      b = row[k + 1] * D;
    for (let j = 0; j < D; j++) x[j] += e[b + j];
  }
  return x;
};

// --------------------------------------------------------- global prefill
function prefill(rows) {
  const S = rows.length;
  const hs = rows.map(embedRow);
  const kv = [];
  for (let l = 0; l < LAYERS; l++) {
    const lay = GL[l],
      nk = [],
      nv = [];
    for (let i = 0; i < S; i++) {
      const qkv = gemv(layerNorm(hs[i], lay.ln1W, lay.ln1B), lay.cAttn, lay.cAttnB, 2304);
      const { q, k, v } = splitHeads(qkv);
      const qr = new Float32Array(D),
        kr = new Float32Array(D);
      ropeInto(qr, q, i, 0);
      ropeInto(kr, k, i, 0);
      nk.push(kr);
      nv.push(v);
      const proj = gemv(attend(qr, nk, nv, i + 1, i + 1), lay.cProj, lay.cProjB, D);
      for (let j = 0; j < D; j++) hs[i][j] += proj[j];
    }
    for (let i = 0; i < S; i++) {
      const a = gemv(layerNorm(hs[i], lay.ln2W, lay.ln2B), lay.fcIn, lay.fcInB, F);
      for (let j = 0; j < F; j++) a[j] = geluTanh(a[j]);
      const b = gemv(a, lay.fcOut, lay.fcOutB, D);
      for (let j = 0; j < D; j++) hs[i][j] += b[j];
    }
    kv.push({ k: nk, v: nv });
  }
  return { hidden: hs.map((h) => layerNorm(h, glnFW, glnFB)), kv, n: S };
}

// --------------------------------------------------------- global step
function decodeStep(globalHidden, pastKv, position) {
  const hs = globalHidden;
  for (let l = 0; l < LAYERS; l++) {
    const lay = GL[l];
    const qkv = gemv(layerNorm(hs, lay.ln1W, lay.ln1B), lay.cAttn, lay.cAttnB, 2304);
    const { q, k, v } = splitHeads(qkv);
    const qr = new Float32Array(D),
      kr = new Float32Array(D);
    ropeInto(qr, q, position, 0);
    ropeInto(kr, k, position, 0);
    const lc = pastKv[l];
    lc.k.push(kr);
    lc.v.push(v);
    const proj = gemv(
      attend(qr, lc.k, lc.v, lc.k.length, lc.k.length),
      lay.cProj,
      lay.cProjB,
      D,
    );
    for (let j = 0; j < D; j++) hs[j] += proj[j];
    // the MLP belongs to the SAME layer — interleaving matters
    const a = gemv(layerNorm(hs, lay.ln2W, lay.ln2B), lay.fcIn, lay.fcInB, F);
    for (let j = 0; j < F; j++) a[j] = geluTanh(a[j]);
    const b = gemv(a, lay.fcOut, lay.fcOutB, D);
    for (let j = 0; j < D; j++) hs[j] += b[j];
  }
  return layerNorm(hs, glnFW, glnFB);
}

// --------------------------------------------------------- local head
/** One local_cached_step: append to the frame-local KV cache, return logits. */
function localStep(
  globalHidden,
  { text = 0, audio = 0, channel = 0, stepType = 0 },
  kvCache,
  pos,
) {
  const e = new Float32Array(D);
  if (stepType === 0) for (let j = 0; j < D; j++) e[j] = globalHidden[j];
  else if (stepType === 1) for (let j = 0; j < D; j++) e[j] = wte[text * D + j];
  else {
    const t = AUD[channel],
      b = audio * D;
    for (let j = 0; j < D; j++) e[j] = t[b + j];
  }
  const qkv = gemv(layerNorm(e, LOC.ln1W, LOC.ln1B), LOC.cAttn, LOC.cAttnB, 2304);
  const { q, k, v } = splitHeads(qkv);
  const qr = new Float32Array(D),
    kr = new Float32Array(D);
  ropeInto(qr, q, pos, 0);
  ropeInto(kr, k, pos, 0);
  kvCache.k.push(kr);
  kvCache.v.push(v);
  const proj = gemv(
    attend(qr, kvCache.k, kvCache.v, kvCache.k.length, kvCache.k.length),
    LOC.cProj,
    LOC.cProjB,
    D,
  );
  for (let j = 0; j < D; j++) e[j] += proj[j];
  const a = gemv(layerNorm(e, LOC.ln2W, LOC.ln2B), LOC.fcIn, LOC.fcInB, F);
  for (let j = 0; j < F; j++) a[j] = geluTanh(a[j]);
  const b = gemv(a, LOC.fcOut, LOC.fcOutB, D);
  for (let j = 0; j < D; j++) e[j] += b[j];
  return layerNorm(e, LOC.lnFW, LOC.lnFB);
}
const argmaxPenalty = (vals, seen, penalty) => {
  let best = 0,
    bestVal = -Infinity;
  const apply = seen.size > 0 && penalty !== 1;
  for (let i = 0; i < vals.length; i++) {
    let s = vals[i];
    if (apply && seen.has(i)) s = s < 0 ? s * penalty : s / penalty;
    if (s > bestVal) {
      bestVal = s;
      best = i;
    }
  }
  return best;
};

// ------------------------------------------------------------------ prompt
const buildRows = (codes, ids) => {
  const t = (id) => {
    const r = new Array(17).fill(PAD);
    r[0] = id;
    return r;
  };
  const a = (c) => {
    const r = new Array(17).fill(PAD);
    r[0] = CFG.audio_user_slot_token_id;
    for (let i = 0; i < Math.min(c.length, N_VQ); i++) r[i + 1] = c[i];
    return r;
  };
  return [
    ...[...M.prompt_templates.user_prompt_prefix_token_ids, CFG.audio_start_token_id].map(t),
    ...codes.map(a),
    ...[
      CFG.audio_end_token_id,
      ...M.prompt_templates.user_prompt_after_reference_token_ids,
      ...ids,
      ...M.prompt_templates.assistant_prompt_prefix_token_ids,
      CFG.audio_start_token_token_id ?? CFG.audio_start_token_id,
    ].map(t),
  ];
};

// ------------------------------------------------------------------- run
const golden = JSON.parse(
  readFileSync(
    new URL("../../tests/fixtures/golden/greedy-junhao-zh.json", import.meta.url),
    "utf8",
  ),
);
const voice = M.builtin_voices[0],
  sample = M.text_samples[0];
const rows = buildRows(voice.prompt_audio_codes, sample.text_token_ids);
console.log(`prefill rows = ${rows.length}`);
const t0 = Date.now();
let hidden, kv;
const CACHE = `${DIR}/ref-prefill.bin`;
if (existsSync(CACHE)) {
  // cache the prefill so iterating on the local head is fast
  const buf = readFileSync(CACHE);
  const S = buf.readInt32LE(0);
  const need = 4 + S * 4 * D + LAYERS * 2 * S * 4 * D;
  let off = 4;
  hidden = [];
  for (let i = 0; i < S; i++) {
    hidden.push(
      new Float32Array(buf.buffer.slice(buf.byteOffset + off, buf.byteOffset + off + D * 4)),
    );
    off += D * 4;
  }
  kv = [];
  for (let l = 0; l < LAYERS; l++) {
    const nk = [],
      nv = [];
    for (let i = 0; i < S; i++) {
      nk.push(
        new Float32Array(buf.buffer.slice(buf.byteOffset + off, buf.byteOffset + off + D * 4)),
      );
      off += D * 4;
    }
    for (let i = 0; i < S; i++) {
      nv.push(
        new Float32Array(buf.buffer.slice(buf.byteOffset + off, buf.byteOffset + off + D * 4)),
      );
      off += D * 4;
    }
    kv.push({ k: nk, v: nv });
  }
  console.log(
    `prefill loaded from cache (${(buf.byteLength / 1e6).toFixed(0)} MB, need ${need})`,
  );
} else {
  ({ hidden, kv } = prefill(rows));
  const parts = [Buffer.alloc(4)];
  parts[0].writeInt32LE(rows.length);
  for (const h of hidden) parts.push(Buffer.from(h.buffer, h.byteOffset, D * 4));
  for (const l of kv)
    for (const arr of [...l.k, ...l.v])
      parts.push(Buffer.from(arr.buffer, arr.byteOffset, D * 4));
  writeFileSync(CACHE, Buffer.concat(parts));
  console.log(`prefill done in ${((Date.now() - t0) / 1000).toFixed(1)}s (cached)`);
}
let globalHidden = hidden[hidden.length - 1];
let position = rows.length;

const seen = Array.from({ length: N_VQ }, () => new Set());
const frames = [];
for (let step = 0; step < FRAMES; step++) {
  const cache = { k: [], v: [] };
  // step 0: decide assistant vs end (only 2 columns of the 16384-wide head)
  const h0 = localStep(globalHidden, { stepType: 0 }, cache, 0);
  // text_lm_head is [768, 16384]; only columns audio_end(7) and assistant(9) are used
  const col = (j) => {
    let s = 0;
    for (let d = 0; d < D; d++) s += h0[d] * LOC.textHead[d * CFG.vocab_size + j];
    return s;
  };
  const tEnd = col(CFG.audio_end_token_id),
    tAsst = col(CFG.audio_assistant_slot_token_id);
  const nextText = tAsst >= tEnd ? CFG.audio_assistant_slot_token_id : CFG.audio_end_token_id;
  if (nextText !== CFG.audio_assistant_slot_token_id) {
    console.log(`stop at step ${step}`);
    break;
  }
  const frame = [];
  let prev = 0;
  for (let ch = 0; ch < N_VQ; ch++) {
    const h = localStep(
      globalHidden,
      ch === 0
        ? { text: nextText, stepType: 1 }
        : { audio: prev, channel: ch - 1, stepType: 2 },
      cache,
      ch + 1,
    );
    const logits = gemv(h, LOC.audioHeads[ch], null, 1024);
    const tok = argmaxPenalty(logits, seen[ch], GEN.audio_repetition_penalty);
    frame.push(tok);
    seen[ch].add(tok);
    prev = tok;
  }
  frames.push(frame);
  if ((step + 1) % 10 === 0)
    console.log(
      `  step ${step + 1}/${FRAMES} (position ${position}) ${((Date.now() - t0) / 1000).toFixed(0)}s`,
    );
  const row = new Array(17).fill(PAD);
  row[0] = CFG.audio_assistant_slot_token_id;
  for (let i = 0; i < N_VQ; i++) row[i + 1] = frame[i];
  globalHidden = decodeStep(embedRow(row), kv, position);
  position += 1;
  if (golden.frames[step + 1]) {
    // the committed golden trace is a flat token array, so per-frame hidden
    // statistics are only present when an extended trace is supplied
    const g = golden.frames[step + 1].hidden ?? golden.frames[step + 1];
    if (g?.min !== undefined) {
      let mn = Infinity,
        mx = -Infinity,
        sum = 0;
      for (const v of globalHidden) {
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
      }
      const ok =
        Math.abs(mn - g.min) < 1e-3 &&
        Math.abs(mx - g.max) < 1e-3 &&
        Math.abs(sum - g.sum) < 1e-2;
      console.log(
        `  hidden after step ${step + 1}: mine min=${mn.toFixed(4)} max=${mx.toFixed(4)} sum=${sum.toFixed(3)} | golden min=${g.min} max=${g.max} sum=${g.sum} ${ok ? "OK" : "MISMATCH"}`,
      );
    }
  }
}

// --------------------------------------------------------------- validate
const gframes = golden.frames.map((f) => (Array.isArray(f) ? f : f.frame));
let match = 0,
  firstDiff = -1;
for (let i = 0; i < Math.min(frames.length, gframes.length); i++) {
  const a = frames[i],
    b = gframes[i];
  if (a.length === b.length && a.every((v, j) => v === b[j])) match += 1;
  else if (firstDiff < 0) {
    firstDiff = i;
    console.log(
      `first mismatch at frame ${i}:\n  mine  ${a.slice(0, 8)}\n  golden${b.slice(0, 8)}`,
    );
  }
}
console.log(`\nframes produced: ${frames.length}, golden: ${gframes.length}`);
console.log(
  `EXACT frame matches: ${match}/${Math.min(frames.length, gframes.length)}${firstDiff < 0 && match === Math.min(frames.length, gframes.length) ? "  ✅" : ""}`,
);
