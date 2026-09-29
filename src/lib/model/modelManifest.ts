/**
 * Model manifest: the exact ONNX artifacts Mockbird needs, and where to get them.
 *
 * Sizes are the exact byte counts reported by the Hugging Face API; the downloader
 * treats them as hints and records the real `content-length` + `ETag` per file so
 * the OPFS cache can be revalidated cheaply.
 *
 * Directory names intentionally mirror the upstream layout
 * (`MOSS-TTS-Nano-100M-ONNX` / `MOSS-Audio-Tokenizer-Nano-ONNX`) because
 * `browser_poc_manifest.json` refers to the codec metadata with a `../` relative
 * path, which the vendored runtime resolves through the asset reader.
 */

export const TTS_REPO = "OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX";
export const CODEC_REPO = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX";

export const TTS_DIR = "MOSS-TTS-Nano-100M-ONNX";
export const CODEC_DIR = "MOSS-Audio-Tokenizer-Nano-ONNX";

/** Optional mirror, e.g. a Cloudflare R2 bucket in front of the HF CDN. */
const MODEL_BASE_URL = (process.env.NEXT_PUBLIC_MODEL_BASE_URL ?? "").replace(/\/+$/, "");

export type ModelGroupId = "tts" | "codec" | "codec-encode";

export type ModelFile = {
  /** Path relative to the OPFS `models/` root, e.g. `MOSS-TTS-Nano-100M-ONNX/tokenizer.model`. */
  path: string;
  repo: string;
  group: ModelGroupId;
  /** Approximate download size in bytes (advisory only). */
  size: number;
};

const tts = (file: string, size: number): ModelFile => ({
  path: `${TTS_DIR}/${file}`,
  repo: TTS_REPO,
  group: "tts",
  size,
});

const codec = (file: string, size: number): ModelFile => ({
  path: `${CODEC_DIR}/${file}`,
  repo: CODEC_REPO,
  group: "codec",
  size,
});

const codecEncode = (file: string, size: number): ModelFile => ({
  path: `${CODEC_DIR}/${file}`,
  repo: CODEC_REPO,
  group: "codec-encode",
  size,
});

export const MODEL_FILES: ModelFile[] = [
  // ---- TTS: autoregressive decoder + local transformer + tokenizer ----
  tts("browser_poc_manifest.json", 503_354),
  tts("tts_browser_onnx_meta.json", 4_487),
  tts("tokenizer.model", 470_897),
  tts("moss_tts_prefill.onnx", 283_305),
  tts("moss_tts_decode_step.onnx", 291_483),
  tts("moss_tts_local_decoder.onnx", 49_231),
  tts("moss_tts_local_cached_step.onnx", 53_685),
  tts("moss_tts_local_fixed_sampled_frame.onnx", 471_262),
  tts("moss_tts_global_shared.data", 440_813_568),
  tts("moss_tts_local_shared.data", 229_678_080),

  // ---- Audio tokenizer decoder: codes -> 48 kHz stereo waveform ----
  codec("codec_browser_onnx_meta.json", 17_036),
  codec("moss_audio_tokenizer_decode_full.onnx", 681_902),
  codec("moss_audio_tokenizer_decode_step.onnx", 351_400),
  codec("moss_audio_tokenizer_decode_shared.data", 44_198_912),

  // ---- Audio tokenizer encoder: reference audio -> codes (voice cloning only) ----
  codecEncode("moss_audio_tokenizer_encode.onnx", 815_775),
  codecEncode("moss_audio_tokenizer_encode.data", 44_507_136),
];

export const MODEL_GROUPS: {
  id: ModelGroupId;
  label: string;
  description: string;
  required: boolean;
}[] = [
  {
    id: "tts",
    label: "Speech decoder (MOSS-TTS-Nano, 100M)",
    description: "Autoregressive text→audio-token model plus its SentencePiece tokenizer.",
    required: true,
  },
  {
    id: "codec",
    label: "Audio tokenizer decoder",
    description: "Turns audio tokens into 48 kHz stereo waveform. Needed for every generation.",
    required: true,
  },
  {
    id: "codec-encode",
    label: "Voice cloning encoder",
    description:
      "Turns your reference recording into audio tokens. Only needed to clone a new voice — built-in voices already ship pre-computed codes.",
    required: false,
  },
];

export const REQUIRED_GROUPS: ModelGroupId[] = MODEL_GROUPS.filter(
  (group) => group.required,
).map((group) => group.id);

export function filesForGroup(group: ModelGroupId): ModelFile[] {
  return MODEL_FILES.filter((file) => file.group === group);
}

export function totalSizeForGroups(groups: ModelGroupId[]): number {
  return MODEL_FILES.filter((file) => groups.includes(file.group)).reduce(
    (sum, file) => sum + file.size,
    0,
  );
}

export function fileUrl(file: ModelFile): string {
  if (MODEL_BASE_URL) {
    return `${MODEL_BASE_URL}/${file.path}`;
  }
  return `https://huggingface.co/${file.repo}/resolve/main/${file.path.split("/").slice(1).join("/")}`;
}

export function usingMirror(): boolean {
  return Boolean(MODEL_BASE_URL);
}

/** Files that must be present for synthesis (i.e. no voice cloning). */
export function isSynthesisReady(present: Set<string>): boolean {
  return REQUIRED_GROUPS.every((group) =>
    filesForGroup(group).every((file) => present.has(file.path)),
  );
}

export function isCloningReady(present: Set<string>): boolean {
  return filesForGroup("codec-encode").every((file) => present.has(file.path));
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  // Decimal units (1 kB = 1000 B) so on-screen numbers match how model sizes are
  // quoted by Hugging Face and in the docs (the full stack is ~763 MB).
  const units = ["B", "kB", "MB", "GB"];
  const exponent = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1000)));
  const value = bytes / 1000 ** exponent;
  return `${value.toFixed(value >= 100 || exponent === 0 ? 0 : 1)} ${units[exponent]}`;
}
