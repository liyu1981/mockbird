/**
 * WGSL engine runtime — the same surface the TTS worker already consumes from
 * the vendored ONNX Runtime runtime, so the flag swap is one line in the worker.
 *
 * Division of labour:
 *   this file          orchestration only: asset IO, manifest, tokenizer, text prep,
 *                      chunking, streaming cadence, cancellation, profiling
 *   policy.ts          the pure functions those steps are built from (measured)
 *   gpu.ts             device work — kernels land there
 *
 * Provenance: text normalisation is ported from the Apache-2.0 reference
 * (not from the unlicensed Reader JavaScript); the tokenizer worker is a
 * deliberate, documented exception.
 */

import {
  type AudioChunk,
  createGpuEngineFor,
  EngineNotImplemented,
  type GpuTtsEngine,
} from "./gpu";
import { openOnnx } from "./onnxWeights";
import {
  buildFrameRow,
  buildVoiceCloneRows,
  containsCjk,
  type GenerationDefaults,
  interChunkPauseSeconds,
  type PromptTemplates,
  resolveAssetPath,
  shouldDecodeNow,
  splitVoiceCloneText,
  streamDecodeBudget,
  type TtsConfig,
} from "./policy";
import { normalizeTtsText } from "./textNormalize";
import { createTokenizer, type TokenizerClient } from "./tokenizer";

export type AssetReader = {
  exists?(args: { rootPath: string; relativePath: string }): Promise<boolean>;
  readText?(args: { rootPath: string; relativePath: string }): Promise<string>;
  readJson?(args: { rootPath: string; relativePath: string }): Promise<unknown>;
  readBuffer?(args: { rootPath: string; relativePath: string }): Promise<Uint8Array>;
};

export type Manifest = {
  model_files: { tts_meta: string; codec_meta: string; tokenizer_model: string };
  tts_config: TtsConfig;
  prompt_templates: PromptTemplates;
  generation_defaults: GenerationDefaults;
  builtin_voices: {
    voice: string;
    display_name: string;
    group: string;
    prompt_audio_codes: number[][];
  }[];
  text_samples: { text?: string; text_token_ids: number[] }[];
};

/** Shape the worker consumes from `onPreparedText` (matches the vendored runtime). */
export type PreparedTextInfo = {
  rawText: string;
  text: string;
  changed: boolean;
  textChunks?: string[];
  normalizationStages: string[];
  warnings: string[];
};

export type Timing = {
  totalMs: number;
  count: number;
  minMs: number;
  maxMs: number;
  averageMs: number;
};

export type ProfileSnapshot = {
  enabled: boolean;
  timings: Record<string, Timing>;
  counters: Record<string, number>;
} | null;

const MANIFEST_CANDIDATES = [
  "browser_poc_manifest.json",
  "MOSS-TTS-Nano-100M-ONNX/browser_poc_manifest.json",
];

export class WgpuTtsRuntime {
  private readonly logger: ((message: string) => void) | null;
  private readonly assetReader: AssetReader | null;
  private readonly tokenizer: TokenizerClient;
  /** injected WebGPU provider; the browser leaves this unset and uses navigator.gpu */
  private readonly gpuProvider: GPU | null;
  private gpu: GpuTtsEngine | null = null;

  private modelRoot = "";
  private threadCount = 1;
  private loaded = false;
  private warmedUp = false;
  private tokenizerKey: string | null = null;
  private manifestValue: Manifest | null = null;
  /** directory the manifest itself lives in, e.g. `MOSS-TTS-Nano-100M-ONNX` */
  private manifestDir = "";
  /** loaded metadata; the GPU layer reads it for layer counts and KV sizing */
  ttsMeta: { model_config: Record<string, unknown> } | null = null;
  private codecMeta: {
    codec_config: {
      sample_rate: number;
      channels: number;
      downsample_rate: number;
      num_quantizers: number;
    };
    streaming_decode?: {
      transformer_offsets: { decoder_index: number }[];
      attention_caches: { decoder_index: number; context: number }[];
    };
  } | null = null;

  private profileState: {
    enabled: boolean;
    timings: Record<string, Timing>;
    counters: Record<string, number>;
  } | null = null;

  constructor(
    options: {
      logger?: ((message: string) => void) | null;
      assetReader?: AssetReader | null;
      /** injectable for Node/CLI use; the browser uses the worker-backed client */
      tokenizer?: TokenizerClient | null;
      /** injectable WebGPU provider for headless Node (Dawn); browser uses navigator.gpu */
      gpu?: GPU | null;
    } = {},
  ) {
    this.logger = options.logger ?? null;
    this.assetReader = options.assetReader ?? null;
    this.tokenizer = options.tokenizer ?? createTokenizer();
    this.gpuProvider = options.gpu ?? null;
  }

  /** The worker reads (and mutates `generation_defaults` on) this, so keep it live. */
  get manifest(): Manifest | null {
    return this.manifestValue;
  }

  private log(message: string): void {
    this.logger?.(message);
  }

  // ------------------------------------------------------------- profiling
  setProfilingEnabled(enabled = true): ProfileSnapshot {
    this.profileState = enabled ? { enabled: true, timings: {}, counters: {} } : null;
    return this.getProfileSnapshot();
  }

  resetProfile(): ProfileSnapshot {
    if (this.profileState?.enabled)
      this.profileState = { enabled: true, timings: {}, counters: {} };
    return this.getProfileSnapshot();
  }

  getProfileSnapshot(): ProfileSnapshot {
    if (!this.profileState?.enabled) return null;
    const timings: Record<string, Timing> = {};
    for (const [key, value] of Object.entries(this.profileState.timings)) {
      timings[key] = { ...value, averageMs: value.count ? value.totalMs / value.count : 0 };
    }
    return { enabled: true, timings, counters: { ...this.profileState.counters } };
  }

  private record(key: string, durationMs: number): void {
    if (!this.profileState?.enabled) return;
    const current = this.profileState.timings[key] ?? {
      totalMs: 0,
      count: 0,
      minMs: 0,
      maxMs: 0,
    };
    current.totalMs += durationMs;
    current.count += 1;
    current.maxMs = Math.max(current.maxMs, durationMs);
    if (current.count === 1) current.minMs = durationMs;
    this.profileState.timings[key] = current;
  }

  private count(key: string, delta = 1): void {
    if (!this.profileState?.enabled) return;
    this.profileState.counters[key] = (this.profileState.counters[key] ?? 0) + delta;
  }

  // ---------------------------------------------------------------- assets
  /**
   * Resolve a path declared by `browser_poc_manifest.json`.
   *
   * Every entry in `model_files` is relative to the *manifest's own* directory,
   * not to the model root, and `codec_meta` uses a `../` parent reference — with
   * Mockbird's layout that means `MOSS-TTS-Nano-100M-ONNX/…` and
   * `MOSS-Audio-Tokenizer-Nano-ONNX/…`. Resolving against the root instead looks
   * like "Model file missing from cache: <bare filename>", which is how this
   * bit twice. One resolver, used for every declared file.
   */
  private resolveModelFile(declared: string): string {
    return resolveAssetPath(this.manifestDir, declared);
  }

  /**
   * The directory a manifest-declared model file lives in, e.g.
   * `MOSS-TTS-Nano-ONNX`. The codec meta is declared as `../<other>/…`, so this
   * has to resolve against the manifest's own directory and hand back the
   * sibling's — that is the second half of why the files were reported missing.
   */
  private modelDirOf(declared: string): string {
    const resolved = resolveAssetPath(this.manifestDir, declared);
    return resolved.includes("/") ? resolved.slice(0, resolved.lastIndexOf("/")) : "";
  }

  private joinPath(dir: string, child: string): string {
    return resolveAssetPath(dir, child);
  }

  private async readText(relativePath: string): Promise<string> {
    if (this.assetReader?.readText)
      return this.assetReader.readText({ rootPath: this.modelRoot, relativePath });
    throw new Error("WGSL engine requires an assetReader (OPFS-backed)");
  }

  private async readBuffer(relativePath: string): Promise<Uint8Array> {
    if (this.assetReader?.exists) {
      const present = await this.assetReader.exists({ rootPath: this.modelRoot, relativePath });
      if (!present) {
        throw new Error(
          `Model file missing from cache: ${relativePath} (looked in ${this.modelRoot}/). ` +
            `Re-download the model from Settings -> Model.`,
        );
      }
    }
    if (this.assetReader?.readBuffer)
      return this.assetReader.readBuffer({ rootPath: this.modelRoot, relativePath });
    throw new Error("WGSL engine requires an assetReader (OPFS-backed)");
  }

  private async readJson<T>(relativePath: string): Promise<T> {
    if (this.assetReader?.readJson)
      return (await this.assetReader.readJson({ rootPath: this.modelRoot, relativePath })) as T;
    return JSON.parse(await this.readText(relativePath)) as T;
  }

  // --------------------------------------------------------------- loading
  async configure({
    modelPath,
    threadCount = 4,
  }: {
    modelPath: string;
    threadCount?: number;
  }): Promise<void> {
    const root = String(modelPath || "").replace(/\/+$/, "");
    const threads = Math.max(1, Number(threadCount) || 1);
    if (this.modelRoot === root && this.threadCount === threads && this.manifestValue) return;
    this.modelRoot = root;
    this.threadCount = threads;
    this.loaded = false;
    this.warmedUp = false;
    this.gpu = null;
    this.manifestValue = null;
    this.tokenizerKey = null;
  }

  async ensureManifestLoaded(): Promise<void> {
    if (this.manifestValue) return;
    let lastError: unknown = null;
    for (const candidate of MANIFEST_CANDIDATES) {
      try {
        this.log(`reading ${candidate}`);
        this.manifestValue = await this.readJson<Manifest>(candidate);
        this.manifestDir = candidate.includes("/")
          ? candidate.slice(0, candidate.lastIndexOf("/"))
          : "";
        const join = (child: string) => resolveAssetPath(this.manifestDir, child);
        const files = this.manifestValue.model_files;
        this.ttsMeta = await this.readJson<{ model_config: Record<string, unknown> }>(
          join(files.tts_meta),
        );
        this.codecMeta = await this.readJson(join(files.codec_meta));
        this.log(
          `manifest ready: ${candidate} (dir=${this.manifestDir || "."}, voices=${this.manifestValue.builtin_voices.length})`,
        );
        return;
      } catch (error) {
        lastError = error;
      }
    }
    throw lastError ?? new Error("no manifest candidate resolved");
  }

  async ensureSynthesisLoaded(): Promise<void> {
    await this.ensureManifestLoaded();
    if (this.loaded) return;
    // The graphs are siblings of the manifest, NOT of model_files.tts_meta — and
    // the codec ones live in the *other* model's directory, which the manifest
    // references as `../` from codec_meta. Three graphs are needed for synthesis:
    // the global transformer, the local frame head and the codec decoder.
    const ttsDir = this.modelDirOf(this.manifestValue!.model_files.tts_meta);
    const codecDir = this.modelDirOf(this.manifestValue!.model_files.codec_meta);
    const global = openOnnx(
      await this.readBuffer(this.joinPath(ttsDir, "moss_tts_decode_step.onnx")),
      await this.readBuffer(this.joinPath(ttsDir, "moss_tts_global_shared.data")),
    );
    const local = openOnnx(
      await this.readBuffer(this.joinPath(ttsDir, "moss_tts_local_cached_step.onnx")),
      await this.readBuffer(this.joinPath(ttsDir, "moss_tts_local_shared.data")),
    );
    const codec = openOnnx(
      await this.readBuffer(this.joinPath(codecDir, "moss_audio_tokenizer_decode_step.onnx")),
      await this.readBuffer(this.joinPath(codecDir, "moss_audio_tokenizer_decode_shared.data")),
    );
    this.log(
      `weights parsed: global ${global.size}, local head ${local.size}, codec ${codec.size} tensors`,
    );
    // KV ring sized from the manifest: prefill rows + max_new_frames (§9.3)
    const prefillRows =
      this.manifestValue!.prompt_templates.user_prompt_prefix_token_ids.length +
      1 +
      Math.max(
        ...this.manifestValue!.builtin_voices.map((voice) => voice.prompt_audio_codes.length),
      ) +
      this.manifestValue!.prompt_templates.user_prompt_after_reference_token_ids.length +
      this.manifestValue!.prompt_templates.assistant_prompt_prefix_token_ids.length +
      2 +
      75; // voice_clone_max_text_tokens
    const maxSlots = prefillRows + this.manifestValue!.generation_defaults.max_new_frames;
    this.log(`KV ring: ${maxSlots} slots (prefill rows + max_new_frames)`);

    const streaming = this.codecMeta?.streaming_decode;
    if (!streaming) {
      throw new Error(
        "codec_browser_onnx_meta.json has no streaming_decode block — the WGSL codec " +
          "decoder needs the per-stage layer counts and ring contexts",
      );
    }
    const layersPerStage: number[] = [];
    const contextPerStage: number[] = [];
    for (const offset of streaming.transformer_offsets) {
      const count = streaming.attention_caches.filter(
        (c: { decoder_index: number }) => c.decoder_index === offset.decoder_index,
      ).length;
      layersPerStage.push(count);
      contextPerStage.push(
        streaming.attention_caches.find(
          (c: { decoder_index: number }) => c.decoder_index === offset.decoder_index,
        )?.context ?? 0,
      );
    }

    const defaults = this.manifestValue!.generation_defaults;
    const config = this.manifestValue!.tts_config;
    this.gpu = await createGpuEngineFor(
      { global, local, codec },
      (message: string) => this.log(message),
      {
        maxSlots,
        codecShape: { layersPerStage, contextPerStage },
        gpu: this.gpuProvider ?? undefined,
        localHead: {
          audioEndTokenId: config.audio_end_token_id,
          audioAssistantSlotTokenId: config.audio_assistant_slot_token_id,
          repetitionPenalty: defaults.audio_repetition_penalty,
        },
      },
    );
    this.loaded = true;
  }

  async ensureCodecEncodeLoaded(): Promise<void> {
    await this.ensureSynthesisLoaded();
    throw new EngineNotImplemented("clone encoder (codec encode convs)");
  }

  /** The GPU engine's own warm-up report, for the Settings panel. */
  warmupReport(): { prefillMs: number; layers: number; rows: number } | null {
    const gpu = this.gpu as (GpuTtsEngine & { warmupReport?: () => unknown }) | null;
    return (
      (gpu?.warmupReport?.() as
        | { prefillMs: number; layers: number; rows: number }
        | undefined) ?? null
    );
  }

  async warmup(): Promise<void> {
    await this.ensureSynthesisLoaded();
    if (this.warmedUp) return;
    // the GPU engine warms itself up inside createGpuEngineFor (one prefill over a
    // minimal prompt), so there is nothing left to do here
    this.warmedUp = true;
    this.log("warmup complete");
  }

  async ensureTokenizerLoaded(): Promise<void> {
    await this.ensureManifestLoaded();
    if (this.tokenizerKey) return;
    const file = this.resolveModelFile(
      this.manifestValue!.model_files.tokenizer_model || "tokenizer.model",
    );
    const bytes = await this.readBuffer(file);
    this.tokenizerKey = await this.tokenizer.load(bytes);
    this.log(`tokenizer loaded: ${file}`);
  }

  // ------------------------------------------------------------------ text
  prepareSynthesisText(
    text: string,
    options: { enableNormalizeTtsText?: boolean; enableWeTextProcessing?: boolean } = {},
  ) {
    const rawText = String(text ?? "");
    const stages: string[] = [];
    const warnings: string[] = [];
    if (options.enableWeTextProcessing !== false) {
      // WeTextProcessing is not part of the Apache-2.0 reference pipeline
      warnings.push("WeTextProcessing requested but not available in the WGSL engine.");
      stages.push("wetext_unavailable");
    }
    const normalized =
      options.enableNormalizeTtsText === false ? rawText : normalizeTtsText(rawText);
    if (options.enableNormalizeTtsText !== false) stages.push("robust");
    if (normalized !== rawText) {
      this.log(`normalisation applied: ${rawText.length} -> ${normalized.length} chars`);
    }
    return {
      rawText,
      text: normalized,
      changed: normalized !== rawText,
      textChunks: undefined,
      normalizationStages: stages,
      warnings,
    } satisfies PreparedTextInfo;
  }

  async encodeText(text: string): Promise<number[]> {
    await this.ensureTokenizerLoaded();
    return this.tokenizer.encode(this.tokenizerKey!, String(text || ""));
  }

  async countTextTokens(text: string): Promise<number> {
    await this.ensureTokenizerLoaded();
    return (await this.encodeText(text)).length;
  }

  async splitVoiceCloneText(text: string, maxTokens = 75): Promise<string[]> {
    await this.ensureTokenizerLoaded();
    const normalized = String(text || "").trim();
    if (!normalized) return [];
    const prepared = containsCjk(normalized)
      ? normalized.replace(/\r/g, " ").replace(/\n/g, " ").replace(/ {2,}/g, " ")
      : normalized.replace(/\r/g, " ").replace(/\n/g, " ").replace(/ {2,}/g, " ");
    return splitVoiceCloneText(prepared, maxTokens, (t) => this.countTextTokens(t));
  }

  listBuiltinVoices() {
    return this.manifestValue?.builtin_voices ?? [];
  }

  // ------------------------------------------------------------ synthesis
  async synthesizeVoiceClone(options: {
    text: string;
    voiceName?: string | null;
    promptAudioCodes?: number[][] | null;
    preencodedTextTokenIds?: number[] | null;
    doSample?: boolean | null;
    sampleMode?: "greedy" | "fixed" | null;
    streaming?: boolean;
    onAudioChunk?: (chunk: AudioChunk) => void | Promise<void>;
    onPreparedText?: (info: PreparedTextInfo) => void | Promise<void>;
    isCancelled?: () => boolean;
    voiceCloneMaxTextTokens?: number;
    enableNormalizeTtsText?: boolean;
    enableWeTextProcessing?: boolean;
  }): Promise<{ textChunks: string[]; outputs: { frames: number; chunks: AudioChunk[] }[] }> {
    const started = Date.now();
    const isCancelled = options.isCancelled ?? (() => false);
    const config = this.manifestValue!.tts_config;
    const defaults = this.manifestValue!.generation_defaults;
    if (defaults.sample_mode !== "greedy" && defaults.do_sample) {
      this.log("WGSL engine implements greedy decoding only; using greedy.");
    }

    const promptAudioCodes =
      options.promptAudioCodes ??
      this.listBuiltinVoices().find((v) => v.voice === options.voiceName)?.prompt_audio_codes;
    if (!promptAudioCodes?.length)
      throw new Error(`Voice preset not found or empty: ${options.voiceName}`);

    const prepared = this.prepareSynthesisText(String(options.text ?? ""), {
      enableNormalizeTtsText: options.enableNormalizeTtsText,
      enableWeTextProcessing: options.enableWeTextProcessing,
    });
    await options.onPreparedText?.(prepared);

    await this.ensureSynthesisLoaded();
    const chunks = await this.splitVoiceCloneText(
      prepared.text,
      options.voiceCloneMaxTextTokens ?? 75,
    );
    const outputs: { frames: number; chunks: AudioChunk[] }[] = [];

    for (let index = 0; index < chunks.length; index += 1) {
      if (isCancelled()) break;
      const chunkText = chunks[index];
      const textTokenIds = await this.encodeText(chunkText);
      const request = buildVoiceCloneRows(
        this.manifestValue!.prompt_templates,
        config,
        promptAudioCodes,
        textTokenIds,
      );
      const engine = this.gpu!;
      // Each text chunk is its own utterance: reset the KV ring, the codec ring
      // caches and the repetition-penalty history, matching the vendored
      // runtime, which resets its codec streaming session per chunk.
      engine.reset();

      const prefillStart = Date.now();
      let hidden: Float32Array;
      try {
        hidden = await engine.prefill(request);
      } catch (error) {
        this.record("tts.prefill_session", Date.now() - prefillStart);
        throw error;
      }
      this.record("tts.prefill_session", Date.now() - prefillStart);
      this.count("generation.prefill_rows", request.inputIds.length);

      const generated: number[][] = [];
      for (let step = 0; step < defaults.max_new_frames; step += 1) {
        if (isCancelled()) break;
        // 17 sequential local-head steps; then one global step
        const frame = await this.nextFrame(hidden);
        if (!frame) {
          this.log(`generation stopped at step ${step} (audio_end)`);
          this.count("generation.stop_step");
          break;
        }
        generated.push(frame);
        this.count("generation.generated_frames");
        const stepStart = Date.now();
        try {
          hidden = await engine.decodeStep({
            row: buildFrameRow(config, frame),
            position: request.inputIds.length + step,
          });
        } catch (error) {
          this.record("tts.decode_session", Date.now() - stepStart);
          throw error;
        }
        this.record("tts.decode_session", Date.now() - stepStart);
      }

      const codecStart = Date.now();
      let pcm: Float32Array[] = [];
      try {
        pcm = await engine.decodeFrames(generated);
      } finally {
        this.record("audio_decode.streaming_codec_decode_session", Date.now() - codecStart);
      }
      // One chunk carrying every channel, like the vendored runtime: the worker
      // forwards `chunkData` as `channels` to the AudioWorklet, so splitting the
      // channels into separate chunks would make the player treat each channel as
      // a whole mono frame and duplicate/interleave them.
      const chunk: AudioChunk = {
        channels: pcm.length,
        sampleRate: this.sampleRate(),
        chunkData: pcm,
        isPause: false,
      };
      outputs.push({ frames: generated.length, chunks: [chunk] });
      await options.onAudioChunk?.(chunk);

      if (index < chunks.length - 1) {
        const seconds = interChunkPauseSeconds(chunkText);
        const samples = Math.max(0, Math.round(this.sampleRate() * seconds));
        if (samples > 0) {
          const pause: AudioChunk = {
            channels: this.channels(),
            sampleRate: this.sampleRate(),
            chunkData: Array.from({ length: this.channels() }, () => new Float32Array(samples)),
            isPause: true,
          };
          await options.onAudioChunk?.(pause);
          outputs.push({ frames: 0, chunks: [pause] });
        }
      }
    }
    this.record("pipeline.synthesize_voice_clone", Date.now() - started);
    return { textChunks: chunks, outputs };
  }

  /** One frame from the 17-step local head, or null when the model stops. */
  private async nextFrame(hidden: Float32Array): Promise<number[] | null> {
    return this.gpu!.nextFrame(hidden);
  }

  async encodeReferenceAudioFromWaveform(
    waveform: Float32Array,
    length: number,
  ): Promise<number[][]> {
    await this.ensureSynthesisLoaded();
    return this.gpu!.encodeWaveform(waveform, length);
  }

  private sampleRate(): number {
    return this.codecMeta?.codec_config.sample_rate ?? 48000;
  }

  private channels(): number {
    return this.codecMeta?.codec_config.channels ?? 2;
  }

  dispose(): void {
    this.gpu?.dispose();
    this.gpu = null;
  }

  // kept for parity with the vendored runtime's unused-but-exported helpers
  streamPacing = { streamDecodeBudget, shouldDecodeNow };
}
