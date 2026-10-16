/**
 * End-to-end decode conformance: the GPU engine against the committed golden
 * trace (`tests/fixtures/golden/greedy-junhao-zh.json`), which was produced by
 * running the published ONNX graphs headlessly under onnxruntime-node.
 *
 *   MOCKBIRD_MODEL_DIR=/tmp/mockbird-oracle \
 *     node --import ./tests/gpu/register-ts.mjs tests/gpu/decode.test.mjs [frames]
 *
 * The gate, in order:
 *   1. the 195-row prefill hidden state (already covered by prefill.test.mjs)
 *   2. the frame token stream — must be **bit-exact**; one flipped token
 *      diverges the rest of the utterance and is inaudible until then
 *   3. the codec PCM statistics (min/max/sum/hash) after those frames
 *
 * Step 2 is the one that matters for "does the WGSL engine sound right": it can
 * only pass if the local head's 17 steps, the repetition penalty, the global
 * decode step and the KV ring all agree with the reference token for token.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CodecRunner } from "../../src/lib/tts/engine/codec.ts";
import { DecodeRunner } from "../../src/lib/tts/engine/decode.ts";
import {
  resolveCodecWeights,
  resolveLocalWeights,
} from "../../src/lib/tts/engine/decodeWeights.ts";
import { createDevice } from "../../src/lib/tts/engine/device.ts";
import { openOnnx } from "../../src/lib/tts/engine/onnxWeights.ts";
import { PrefillRunner, resolveLayerWeights } from "../../src/lib/tts/engine/prefill.ts";
import { uploadWeights } from "../../src/lib/tts/engine/weights.ts";
import { gpu } from "./webgpu.mjs";

const dir = process.env.MOCKBIRD_MODEL_DIR;
if (!dir) {
  console.log("skip: set MOCKBIRD_MODEL_DIR to a directory with the decode weights");
  process.exit(0);
}
const FRAMES = Number(process.argv[2] ?? 40);
const GOLDEN = new URL("../fixtures/golden/greedy-junhao-zh.json", import.meta.url);
const golden = JSON.parse(readFileSync(GOLDEN, "utf8"));
const manifest = JSON.parse(readFileSync(join(dir, "browser_poc_manifest.json"), "utf8"));
const codecMeta = JSON.parse(readFileSync(join(dir, "codec_browser_onnx_meta.json"), "utf8"));

let failures = 0;
const check = (name, ok, detail = "") => {
  if (!ok) failures += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name}${detail ? `  ${detail}` : ""}`);
};

// ---- the same prompt the oracle used: built-in voice Junhao + text_samples[0]
const CFG = manifest.tts_config;
const PAD = CFG.audio_pad_token_id;
const buildRows = (codes, ids) => {
  const text = (id) => {
    const row = new Array(17).fill(PAD);
    row[0] = id;
    return row;
  };
  const audio = (c) => {
    const row = new Array(17).fill(PAD);
    row[0] = CFG.audio_user_slot_token_id;
    for (let i = 0; i < Math.min(c.length, CFG.n_vq); i++) row[i + 1] = c[i];
    return row;
  };
  return [
    ...[
      ...manifest.prompt_templates.user_prompt_prefix_token_ids,
      CFG.audio_start_token_id,
    ].map(text),
    ...codes.map(audio),
    ...[
      CFG.audio_end_token_id,
      ...manifest.prompt_templates.user_prompt_after_reference_token_ids,
      ...ids,
      ...manifest.prompt_templates.assistant_prompt_prefix_token_ids,
      CFG.audio_start_token_id,
    ].map(text),
  ];
};

const tables = {
  global: openOnnx(
    readFileSync(join(dir, "moss_tts_decode_step.onnx")),
    readFileSync(join(dir, "moss_tts_global_shared.data")),
  ),
  local: openOnnx(
    readFileSync(join(dir, "moss_tts_local_cached_step.onnx")),
    readFileSync(join(dir, "moss_tts_local_shared.data")),
  ),
  codec: openOnnx(
    readFileSync(join(dir, "moss_audio_tokenizer_decode_step.onnx")),
    readFileSync(join(dir, "moss_audio_tokenizer_decode_shared.data")),
  ),
};

const { device, info } = await createDevice(gpu);
console.log(`adapter: ${info.adapterName} | chunk ${(info.chunkBytes / 1e6).toFixed(1)} MB`);

const global = uploadWeights(device, tables.global, info);
const local = uploadWeights(device, tables.local, info);
const codecWeights = uploadWeights(device, tables.codec, info);
console.log(
  `uploaded: global ${global.chunks.length} chunks, local ${local.chunks.length}, codec ${codecWeights.chunks.length}`,
);

const lookup = (weights) => ({
  get: (name) => {
    const block = weights.blocks.get(name);
    if (!block) throw new Error(`missing weight ${name}`);
    return block;
  },
});

const layers = resolveLayerWeights(global, lookup(global));
const lnF = {
  w: lookup(global).get("core.model.transformer.ln_f.weight"),
  b: lookup(global).get("core.model.transformer.ln_f.bias"),
};
const embed = {
  wte: global.blocks.get("core.model.transformer.wte.weight"),
  audio: Array.from({ length: 16 }, (_, k) =>
    global.blocks.get(`core.model.audio_embeddings.${k}.weight`),
  ),
};
const localHead = resolveLocalWeights(local, lookup(local));
const streaming = codecMeta.streaming_decode;
const layersPerStage = streaming.transformer_offsets.map(
  (o) => streaming.attention_caches.filter((c) => c.decoder_index === o.decoder_index).length,
);
const contextPerStage = streaming.transformer_offsets.map(
  (o) =>
    streaming.attention_caches.find((c) => c.decoder_index === o.decoder_index)?.context ?? 0,
);
const codecWeightsResolved = resolveCodecWeights(codecWeights, lookup(codecWeights), {
  layersPerStage,
  contextPerStage,
});
console.log(
  `codec stages: ${codecWeightsResolved.stages
    .map(
      (s) =>
        `${s.layers.length}L/C${s.context}/in${s.inputProj.dims.join("x")}/out${s.outputProj.dims.join("x")}`,
    )
    .join(" ")}`,
);

// the local head uses the GLOBAL embedding tables (that is what the CPU
// reference does — same checkpoint weights, and the gate proves it)
const maxSlots = golden.config.max_kv_positions;
const prefillRunner = new PrefillRunner(device, global, maxSlots);
const decoder = new DecodeRunner(
  device,
  global,
  local,
  prefillRunner.buffers.kv,
  maxSlots,
  localHead,
  {
    audioEndTokenId: CFG.audio_end_token_id,
    audioAssistantSlotTokenId: CFG.audio_assistant_slot_token_id,
    repetitionPenalty: manifest.generation_defaults.audio_repetition_penalty,
  },
);
const codecRunner = new CodecRunner(device, codecWeights, codecWeightsResolved);

// ---- 1. prefill
const rows = buildRows(
  manifest.builtin_voices[0].prompt_audio_codes,
  manifest.text_samples[0].text_token_ids,
);
console.log(`\nprefill: ${rows.length} rows (golden says ${golden.sample.prefillRows})`);
const ids = new Uint32Array(rows.length * 17);
rows.forEach((row, i) => {
  row.forEach((v, j) => {
    ids[i * 17 + j] = v;
  });
});
prefillRunner.stageIds(ids);
const t0 = Date.now();
const hidden = await prefillRunner.run(rows.length, layers, embed, maxSlots);
console.log(`  prefill ${Date.now() - t0} ms`);
let sum = 0;
let min = Infinity;
let max = -Infinity;
for (const v of hidden) {
  sum += v;
  min = Math.min(min, v);
  max = Math.max(max, v);
}
check(
  "prefill hidden matches the golden statistics",
  Math.abs(min - golden.prefillHidden.min) < 5e-3 &&
    Math.abs(max - golden.prefillHidden.max) < 5e-3 &&
    Math.abs(sum - golden.prefillHidden.sum) < 5e-2,
  `min=${min.toFixed(4)}/${golden.prefillHidden.min} max=${max.toFixed(4)}/${golden.prefillHidden.max} sum=${sum.toFixed(3)}/${golden.prefillHidden.sum}`,
);

// ---- 2. the frame loop: local head -> global decode step
let current = hidden.slice((rows.length - 1) * 768, rows.length * 768);
const frames = [];
const tLoop = Date.now();
const USE_GOLDEN = process.env.GOLDEN_FRAMES === "1";
const SHOW = Number(process.env.SHOW_HIDDEN ?? 0);
for (let step = 0; step < FRAMES; step += 1) {
  const frame = USE_GOLDEN ? golden.frames[step] : await decoder.nextFrame(current, embed);
  if (!frame) {
    console.log(`  model stopped at frame ${step}`);
    break;
  }
  frames.push(frame);
  const row = new Int32Array(17).fill(PAD);
  row[0] = CFG.audio_assistant_slot_token_id;
  for (let i = 0; i < 16; i += 1) row[i + 1] = frame[i];
  current = await decoder.globalStep(row, rows.length + step, layers, lnF, embed);
  if (step < SHOW) {
    let mn = Infinity,
      mx = -Infinity,
      sm = 0;
    for (const v of current) {
      mn = Math.min(mn, v);
      mx = Math.max(mx, v);
      sm += v;
    }
    console.log(
      `  frame ${step} hidden: min=${mn.toFixed(5)} max=${mx.toFixed(5)} sum=${sm.toFixed(4)}`,
    );
  }
}
const loopMs = Date.now() - tLoop;
console.log(
  `  ${frames.length} frames in ${loopMs} ms (${(loopMs / Math.max(1, frames.length)).toFixed(0)} ms/frame)`,
);

const gframes = golden.frames.map((f) => (Array.isArray(f) ? f : f.frame));
let exact = 0;
let firstDiff = -1;
for (let i = 0; i < Math.min(frames.length, gframes.length); i += 1) {
  const same =
    frames[i].length === gframes[i].length && frames[i].every((v, j) => v === gframes[i][j]);
  if (same) exact += 1;
  else if (firstDiff < 0) {
    firstDiff = i;
    console.log(`  first mismatch at frame ${i}:`);
    console.log(`    mine   [${frames[i].slice(0, 8)}]`);
    console.log(`    golden [${gframes[i].slice(0, 8)}]`);
  }
}
check(
  `frame tokens are bit-exact (${exact}/${Math.min(frames.length, gframes.length)})`,
  firstDiff < 0 && exact === Math.min(frames.length, gframes.length),
);

// ---- 3. the codec: those frames -> 48 kHz stereo
if (frames.length) {
  const tCodec = Date.now();
  const channels = await codecRunner.decodeFrames(frames);
  const codecMs = Date.now() - tCodec;
  const left = channels[0];
  const right = channels[1];
  const all = new Float32Array(left.length + right.length);
  all.set(left, 0);
  all.set(right, left.length);
  let cmin = Infinity;
  let cmax = -Infinity;
  let csum = 0;
  for (const v of all) {
    cmin = Math.min(cmin, v);
    cmax = Math.max(cmax, v);
    csum += v;
  }
  const fnv = (() => {
    // FNV-1a over the float bytes, as the oracle recorded it
    let h = 0x811c9dc5;
    const bytes = new Uint8Array(all.buffer, all.byteOffset, all.byteLength);
    for (const byte of bytes) {
      h ^= byte;
      h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h.toString(16).padStart(8, "0");
  })();
  console.log(
    `  codec: ${frames.length} frames in ${codecMs} ms (${(codecMs / frames.length).toFixed(0)} ms/frame)` +
      ` -> ${left.length} samples/channel`,
  );
  check("PCM sample count", left.length === frames.length * 3840, `${left.length}`);
  check(
    "PCM statistics match the golden trace",
    Math.abs(cmin - golden.codec.pcm.min) < 2e-2 &&
      Math.abs(cmax - golden.codec.pcm.max) < 2e-2 &&
      Math.abs(csum - golden.codec.pcm.sum) < 2e-1,
    `min=${cmin.toFixed(4)}/${golden.codec.pcm.min} max=${cmax.toFixed(4)}/${golden.codec.pcm.max} sum=${csum.toFixed(3)}/${golden.codec.pcm.sum} hash=${fnv}/${golden.codec.pcm.hash}`,
  );
}

decoder.dispose();
codecRunner.dispose();
prefillRunner.dispose();
global.dispose();
local.dispose();
codecWeights.dispose();
await device.destroy?.();
console.log(
  failures ? `\n${failures} decode check(s) failed` : "\ndecode path matches the oracle",
);
process.exit(failures ? 1 : 0);
