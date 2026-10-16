/**
 * GPU compute layer of the WGSL engine.
 *
 * STATUS (see docs/moss-tts-runtime-analysis.md §9.7):
 *   ✅ device init + chunked weight upload      (`weights.ts`)
 *   ✅ batched prefill, 12 layers               (`prefill.ts`)
 *   ✅ decode step  (single row, KV ring)       (`decode.ts`)
 *   ✅ local head   (17 sequential steps)       (`decode.ts`)
 *   ✅ codec decode (4 stages + waveform head)  (`codec.ts`)
 *   ⬜ codec encode (voice cloning)
 *
 * The decode step, local head and codec are the three pieces the vendored ONNX
 * Runtime runtime used to provide, and they are all implemented as WGSL here;
 * only `encodeWaveform` — the reference-audio encoder that only voice *cloning*
 * needs — is still missing, and it throws rather than guessing.
 */

import { CodecRunner, type CodecWeights } from "./codec";
import { DecodeRunner, type EmbedTables, type LocalHeadWeights } from "./decode";
import { resolveCodecWeights, resolveLocalWeights } from "./decodeWeights";
import { createDevice, type GpuDeviceInfo } from "./device";
import type { WeightTable } from "./onnxWeights";
import { type LayerWeights, PrefillRunner, resolveLayerWeights } from "./prefill";
import { frameRowFrom } from "./promptRows";
import type { BlockLocation, GpuWeights } from "./weights";
import { uploadWeights } from "./weights";

export class EngineNotImplemented extends Error {
  constructor(step: string) {
    super(
      `WGSL engine: "${step}" is not implemented yet. ` +
        `See docs/moss-tts-runtime-analysis.md §9.7 for the build order.`,
    );
    this.name = "EngineNotImplemented";
  }
}

export type AudioChunk = {
  channels: number;
  sampleRate: number;
  /** channel-major PCM */
  chunkData: Float32Array[];
  isPause: boolean;
};

export type PrefillInput = {
  inputIds: number[][];
  attentionMask: number[][];
};

export type DecodeStepInput = {
  /** 17-wide row: column 0 is the text token, 1..n_vq the audio codes */
  row: Int32Array;
  /** sequence position of this row (== number of rows already in the cache) */
  position: number;
};

/** What the local head needs to know about the model besides its weights. */
export type LocalHeadSettings = {
  audioEndTokenId: number;
  audioAssistantSlotTokenId: number;
  repetitionPenalty: number;
};

export interface GpuTtsEngine {
  readonly sampleRate: number;
  readonly channels: number;
  /** number of 48 kHz samples each frame expands to (3840) */
  readonly samplesPerFrame: number;

  /** rows -> KV cache filled; returns the last hidden state [768] */
  prefill(input: PrefillInput): Promise<Float32Array>;
  /** one global transformer step -> new hidden; appends to the KV cache */
  decodeStep(input: DecodeStepInput): Promise<Float32Array>;
  /**
   * One 12.5 Hz frame from the 17-step local head, or null when the model says
   * the utterance is over. `hidden` is the global hidden state of the frame that
   * is about to be generated.
   */
  nextFrame(hidden: Float32Array): Promise<number[] | null>;
  /** frames -> 48 kHz stereo PCM */
  decodeFrames(frames: number[][]): Promise<Float32Array[]>;
  /**
   * A fresh utterance: drop the global KV, the codec ring caches and the local
   * head's repetition-penalty history. The runtime calls this once per text
   * chunk, exactly as the vendored runtime resets its codec streaming session.
   */
  reset(): void;
  /** reference audio (cloning) -> 16 codes per 12.5 Hz frame */
  encodeWaveform(waveform: Float32Array, length: number): Promise<number[][]>;
  dispose(): void;
}

const VOCAB = 16384;
const N_VQ = 16;
const ROW_WIDTH = 17;
const HIDDEN = 768;
const SAMPLES_PER_FRAME = 3840;
const SAMPLE_RATE = 48000;
const CHANNELS = 2;

class WgpuTtsEngineImpl implements GpuTtsEngine {
  readonly sampleRate = SAMPLE_RATE;
  readonly channels = CHANNELS;
  readonly samplesPerFrame = SAMPLES_PER_FRAME;

  private readonly device: GPUDevice;
  private readonly info: GpuDeviceInfo;
  private readonly global: GpuWeights;
  private readonly localWeights: GpuWeights;
  private readonly codecWeights: GpuWeights;
  private readonly layers: LayerWeights[];
  private readonly lnF: { w: BlockLocation; b: BlockLocation };
  private readonly embed: EmbedTables;
  private readonly localHead: LocalHeadWeights;
  private readonly prefillRunner: PrefillRunner;
  private readonly decoder: DecodeRunner;
  private readonly codec: CodecRunner;
  private readonly settings: LocalHeadSettings;
  /** rows currently in the global KV cache (== the next sequence position) */
  private pastValidLength = 0;
  private lastHidden: Float32Array | null = null;

  constructor(
    device: GPUDevice,
    info: GpuDeviceInfo,
    global: GpuWeights,
    localWeights: GpuWeights,
    codecWeights: GpuWeights,
    layers: LayerWeights[],
    lnF: { w: BlockLocation; b: BlockLocation },
    embed: EmbedTables,
    localHead: LocalHeadWeights,
    codec: CodecWeights,
    settings: LocalHeadSettings,
    maxSlots: number,
  ) {
    this.device = device;
    this.info = info;
    this.global = global;
    this.localWeights = localWeights;
    this.codecWeights = codecWeights;
    this.layers = layers;
    this.lnF = lnF;
    this.embed = embed;
    this.localHead = localHead;
    this.settings = settings;
    this.prefillRunner = new PrefillRunner(device, global, maxSlots);
    this.decoder = new DecodeRunner(
      device,
      global,
      // the local head's blocks live in their own upload, so it needs its own pool
      localWeights,
      // the decode step appends into the *same* KV ring the prefill filled, so a
      // frame attends over the prompt rows and every frame since
      this.prefillRunner.buffers.kv,
      maxSlots,
      localHead,
      settings,
    );
    this.codec = new CodecRunner(device, codecWeights, codec);
  }

  get deviceInfo(): GpuDeviceInfo {
    return this.info;
  }

  /** rows in the global KV cache — the decode step appends at this position */
  get position(): number {
    return this.pastValidLength;
  }

  async prefill(input: PrefillInput): Promise<Float32Array> {
    const rows = input.inputIds.length;
    if (rows === 0) throw new Error("prefill called with no rows");
    const ids = new Uint32Array(rows * ROW_WIDTH);
    input.inputIds.forEach((row, i) => {
      for (let j = 0; j < ROW_WIDTH; j++) {
        ids[i * ROW_WIDTH + j] = j < row.length ? row[j] >>> 0 : 1024;
      }
    });
    this.prefillRunner.stageIds(ids);
    const hidden = await this.prefillRunner.run(
      rows,
      this.layers,
      { wte: this.embed.wte, audio: this.embed.audio },
      this.decoder.maxSlotCount,
    );
    this.pastValidLength = rows;
    this.lastHidden = hidden.slice((rows - 1) * HIDDEN, rows * HIDDEN);
    return this.lastHidden;
  }

  /**
   * Warm-up: run one prefill over a minimal prompt, so every pipeline, every
   * weight chunk and the KV buffer are touched once before the real request.
   * This is a *warm-up*, not a validation — the numeric check it does make is
   * "finite and non-zero", which catches uninitialised buffers and dead shader
   * paths long before a user waits on them.
   */
  async warmup(): Promise<{ prefillMs: number; layers: number; rows: number }> {
    const row = new Array(ROW_WIDTH).fill(1024); // audio_pad_token_id
    row[0] = 6; // audio_start_token_id
    const rows = [row, row];
    // prove the read-back path before blaming a kernel: poison the cache, read it
    // back, then let the prefill overwrite it
    await this.prefillRunner.poisonKv(rows.length);
    const poison = await this.prefillRunner.readLayerCache(0, rows.length);
    if (!poison.key.every((value) => value === -12345.5)) {
      throw new Error("warmup read-back is broken: the KV sentinel was not visible");
    }
    const started = Date.now();
    const hidden = await this.prefill({ inputIds: rows, attentionMask: [[1, 1]] });
    const elapsed = Date.now() - started;
    for (const value of hidden) {
      if (!Number.isFinite(value)) {
        throw new Error("warmup produced a non-finite hidden state — a kernel wrote garbage");
      }
    }
    if (!hidden.some((value) => value !== 0)) {
      throw new Error("warmup produced an all-zero hidden state — nothing was computed");
    }
    // the KV ring must have been written for layer 0, otherwise the decode step
    // would read an untouched buffer
    const cache = await this.prefillRunner.readLayerCache(0, rows.length);
    if (!cache.key.some((value) => value !== 0) || !cache.value.some((value) => value !== 0)) {
      throw new Error(
        "warmup left the layer-0 KV cache empty (read-back verified: the sentinel was visible)",
      );
    }
    // leave the engine in a clean state: the warm-up prompt is not a real request
    this.reset();
    return { prefillMs: elapsed, layers: this.layers.length, rows: rows.length };
  }

  /** Drop the cached KV so the next prefill starts from an empty ring. */
  reset(): void {
    this.pastValidLength = 0;
    this.lastHidden = null;
    this.decoder.clearSeen();
    this.codec.resetRings();
  }

  async decodeStep(input: DecodeStepInput): Promise<Float32Array> {
    if (!this.lastHidden) throw new Error("decodeStep before prefill");
    const position = input.position ?? this.pastValidLength;
    const hidden = await this.decoder.globalStep(
      input.row,
      position,
      this.layers,
      this.lnF,
      this.embed,
    );
    this.pastValidLength = position + 1;
    this.lastHidden = hidden;
    return hidden;
  }

  /**
   * The local head. It runs from the hidden state the runtime passes in — the one
   * returned by `prefill` or the previous `decodeStep` — and returns the 16 audio
   * codes of the next frame, or null at end of utterance.
   *
   * This *is* the frame-generating step; `decodeStep` is what feeds the frame
   * back. The order matters and the runtime already has it right: local head
   * first (it reads the current hidden), then one global step on the frame's row.
   */
  async nextFrame(hidden: Float32Array): Promise<number[] | null> {
    const frame = await this.decoder.nextFrame(hidden, this.embed);
    if (frame) this.lastHidden = hidden;
    return frame;
  }

  async decodeFrames(frames: number[][]): Promise<Float32Array[]> {
    if (!frames.length) return [];
    return this.codec.decodeFrames(frames);
  }

  encodeWaveform(_waveform: Float32Array, _length: number): Promise<number[][]> {
    throw new EngineNotImplemented("clone encoder (codec encode convs)");
  }

  dispose(): void {
    this.decoder.dispose();
    this.codec.dispose();
    this.prefillRunner.dispose();
    this.localWeights.dispose();
    this.codecWeights.dispose();
    this.global.dispose();
    void this.device;
  }
}

export type EngineWeights = {
  global: WeightTable;
  local: WeightTable;
  codec: WeightTable;
};

export type EngineOptions = {
  maxSlots?: number;
  gpu?: GPU;
  localHead: LocalHeadSettings;
  /** layers per stage and ring contexts, from `codec_browser_onnx_meta.json` */
  codecShape: { layersPerStage: number[]; contextPerStage: number[] };
};

/**
 * Bring up the device, upload the three weight sets and return a live engine.
 * `maxSlots` is the global KV ring size: prefill rows + max_new_frames.
 *
 * `options.gpu` is how the headless Dawn harness exercises this exact factory
 * path instead of a parallel one; in the browser it is left undefined and
 * createDevice() falls back to navigator.gpu.
 */
export async function createGpuEngineFor(
  tables: EngineWeights,
  log: (message: string) => void,
  options: EngineOptions,
): Promise<GpuTtsEngine> {
  const { device, info } = await createDevice(options.gpu);
  log(
    `adapter: ${info.adapterName} | chunk=${(info.chunkBytes / 1e6).toFixed(1)} MB ` +
      `| storage buffers/stage=${info.storageBuffersPerStage} | shader-f16=${info.shaderF16}`,
  );
  const upload = (table: WeightTable, name: string) => {
    const weights = uploadWeights(device, table, info);
    log(
      `weights ${name}: ${weights.tensorCount} tensors in ${weights.chunks.length} chunk(s) ` +
        `(${(weights.bytesUploaded / 1e6).toFixed(0)} MB of GPU memory)`,
    );
    return weights;
  };
  const global = upload(tables.global, "global");
  const local = upload(tables.local, "local head");
  const codecWeights = upload(tables.codec, "codec");

  const lookup = (weights: GpuWeights) => ({
    get(name: string) {
      const block = weights.blocks.get(name);
      if (!block) throw new Error(`weight not uploaded: ${name}`);
      return block;
    },
  });
  const layers = resolveLayerWeights(global, lookup(global));
  if (layers.length !== 12) {
    throw new Error(`expected 12 transformer layers, resolved ${layers.length}`);
  }
  const lnF = {
    w: lookup(global).get("core.model.transformer.ln_f.weight"),
    b: lookup(global).get("core.model.transformer.ln_f.bias"),
  };
  const embed: EmbedTables = {
    wte: global.blocks.get("core.model.transformer.wte.weight") as BlockLocation,
    audio: Array.from({ length: N_VQ }, (_, k) =>
      global.blocks.get(`core.model.audio_embeddings.${k}.weight`),
    ) as BlockLocation[],
  };
  const localHead = resolveLocalWeights(local, lookup(local));
  const codec = resolveCodecWeights(codecWeights, lookup(codecWeights), options.codecShape);

  const maxSlots = options.maxSlots ?? 570;
  const engine = new WgpuTtsEngineImpl(
    device,
    info,
    global,
    local,
    codecWeights,
    layers,
    lnF,
    embed,
    localHead,
    codec,
    options.localHead,
    maxSlots,
  );
  log(
    `engine ready: ${layers.length} layers, KV ring ${maxSlots} slots, ` +
      `codec ${codec.stages.length} stages (${codec.stages.map((s) => `${s.layers.length}×C${s.context}`).join(", ")}), ` +
      `vocab ${VOCAB}`,
  );

  // warm up here so a broken kernel surfaces during loadEngine rather than on the
  // user's first generate
  const warm = await engine.warmup();
  log(`warmup: 1 prefill over ${warm.rows} rows in ${warm.prefillMs} ms`);
  return engine;
}

/** Kept for callers that only want the "not wired" signal. */
export function createGpuEngine(): GpuTtsEngine {
  throw new EngineNotImplemented(
    "device init (await createGpuEngineFor with parsed weight tables)",
  );
}

export type { CodecWeights, EmbedTables, LocalHeadWeights };
export { frameRowFrom };
