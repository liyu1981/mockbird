/**
 * Types for the vendored MOSS-TTS-Nano browser runtime.
 *
 * The implementation is upstream JavaScript (see ./README.md for provenance and
 * the exact list of local edits). These declarations describe only the surface
 * Mockbird uses, so the port stays type-checked without rewriting 2.4k lines.
 */

export type BuiltinVoicePreset = {
  voice: string;
  display_name: string;
  group: string;
  audio_file: string;
  prompt_audio_codes: number[][];
};

export type PreparedTextInfo = {
  rawText: string;
  text: string;
  wetextApplied: boolean;
  normalizationStages: string[];
  warnings: string[];
};

export type AudioChunk = {
  channels: number;
  sampleRate: number;
  /** Channel-major PCM. */
  chunkData: Float32Array[];
  isPause: boolean;
};

export type SynthesizeVoiceCloneOptions = {
  text: string;
  voiceName?: string | null;
  promptAudioCodes?: number[][] | null;
  preencodedTextTokenIds?: number[] | null;
  extraVoices?: unknown[];
  doSample?: boolean | null;
  sampleMode?: "fixed" | "greedy" | null;
  streaming?: boolean;
  onAudioChunk?: (chunk: AudioChunk) => void | Promise<void>;
  onPreparedText?: (prepared: PreparedTextInfo) => void | Promise<void>;
  isCancelled?: () => boolean;
  voiceCloneMaxTextTokens?: number;
  enableNormalizeTtsText?: boolean;
  enableWeTextProcessing?: boolean;
};

export type VoiceCloneResult = {
  textChunks: string[];
  outputs: { frames: number; chunks: AudioChunk[] }[];
};

export type ProfileSnapshot = {
  enabled: boolean;
  timings: Record<string, number>;
  counters: Record<string, number>;
} | null;

export type RuntimeAssetReader = {
  /** Mockbird addition: lets the runtime skip optional sessions. */
  exists?: (args: { rootPath: string; relativePath: string }) => Promise<boolean>;
  readText?: (args: { rootPath: string; relativePath: string }) => Promise<string>;
  readJson?: (args: { rootPath: string; relativePath: string }) => Promise<unknown>;
  readBuffer?: (args: { rootPath: string; relativePath: string }) => Promise<Uint8Array>;
};

export declare class BrowserOnnxTtsRuntime {
  constructor(options?: {
    logger?: ((message: string) => void) | null;
    assetReader?: RuntimeAssetReader | null;
  });

  /**
   * Escape hatch: the upstream loop reads sampling/length defaults straight from
   * `browser_poc_manifest.json` (`generation_defaults`), so per-request overrides
   * are applied by mutating this object after `ensureManifestLoaded()`.
   */
  manifest: {
    generation_defaults: Record<string, number | string | boolean>;
  } | null;

  configure(args: { modelPath: string; threadCount?: number }): Promise<void>;
  ensureManifestLoaded(): Promise<void>;
  ensureSynthesisLoaded(): Promise<void>;
  ensureCodecEncodeLoaded(): Promise<void>;
  ensureTokenizerLoaded(): Promise<void>;
  warmup(): Promise<void>;

  readTextAsset(relativePath: string): Promise<string>;
  readJsonAsset<T = unknown>(relativePath: string): Promise<T>;
  readBufferAsset(relativePath: string): Promise<Uint8Array>;

  listBuiltinVoices(): BuiltinVoicePreset[];
  encodeText(text: string): Promise<number[]>;
  countTextTokens(text: string): Promise<number>;
  splitVoiceCloneText(text: string, maxTokens?: number): Promise<string[]>;
  prepareSynthesisText(text: string, options?: Record<string, unknown>): PreparedTextInfo;

  encodeReferenceAudioFromFile(file: File | ArrayBuffer): Promise<number[][]>;
  encodeReferenceAudioFromWaveform(
    waveform: Float32Array,
    waveformLength: number,
  ): Promise<number[][]>;

  synthesizeVoiceClone(options: SynthesizeVoiceCloneOptions): Promise<VoiceCloneResult>;

  setProfilingEnabled(enabled?: boolean): ProfileSnapshot;
  getProfileSnapshot(): ProfileSnapshot;
  resetProfile(): ProfileSnapshot;
}

export declare function createBrowserOnnxTtsRuntime(options?: {
  logger?: ((message: string) => void) | null;
  assetReader?: RuntimeAssetReader | null;
}): BrowserOnnxTtsRuntime;
