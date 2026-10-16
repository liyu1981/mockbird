#!/usr/bin/env node
/**
 * Headless synthesis CLI for the WGSL TTS engine.
 *
 *   pnpm synth "your paragraph here" 0
 *
 * The first positional argument is the text, the second is the index of a
 * built-in voice (see `--list`). It writes a 48 kHz stereo WAV and prints the
 * engine log to stderr.
 *
 * Run it through the TypeScript loader because the engine lives in `src/` as
 * `.ts`:
 *
 *   node --import ./tests/gpu/register-ts.mjs tools/synth.mjs "…" 0
 *
 * Model weights are required: point `--model-dir` (or `MOCKBIRD_MODEL_DIR`) at a
 * directory that has `browser_poc_manifest.json`, the ONNX graphs + `.data`
 * sidecars and `tokenizer.model`. Both the flat layout (the oracle dump) and the
 * manifest's `MOSS-TTS-Nano-100M-ONNX/…` layout are accepted.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { create, globals } from "webgpu";

import { encodeWav } from "../src/lib/tts/audio.ts";
import { WgpuTtsRuntime } from "../src/lib/tts/engine/runtime.ts";

Object.assign(globalThis, globals);

const USAGE = `mockbird synth — run the WGSL TTS engine headlessly

Usage:
  pnpm synth <text> [voiceIndex] [--out <file.wav>] [--model-dir <dir>] [--threads <n>]
  pnpm synth --list

Examples:
  pnpm synth "你好，世界。" 0
  pnpm synth "The quick brown fox." 6 --out fox.wav

Flags:
  --text <text>       the paragraph to speak (alternative to the first positional)
  --voice <n>         built-in voice index (alternative to the second positional)
  --out <file>        output WAV path (default: synth-<voice>.wav)
  --model-dir <dir>    directory with the ONNX weights (or $MOCKBIRD_MODEL_DIR)
  --threads <n>       worker-style thread budget (informational; GPU ignores it)
  --max-frames <n>    hard cap on 12.5 Hz frames per text chunk (default 160;
                      the WGSL engine currently only decodes greedily and the
                      model rarely emits its end token, so this bounds the run)
  --no-trim           keep trailing silence (it is trimmed by default)
  --list              list the built-in voices and exit
  -h, --help          show this help`;

function parseArgs(argv) {
  const positional = [];
  const flags = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "-h" || arg === "--help") flags.help = true;
    else if (arg === "--list") flags.list = true;
    else if (arg === "--text") flags.text = argv[++i];
    else if (arg === "--voice") flags.voice = argv[++i];
    else if (arg === "--out") flags.out = argv[++i];
    else if (arg === "--model-dir") flags.modelDir = argv[++i];
    else if (arg === "--threads") flags.threads = argv[++i];
    else if (arg === "--max-frames") flags.maxFrames = argv[++i];
    else if (arg === "--no-trim") flags.noTrim = true;
    else if (arg.startsWith("-")) throw new Error(`unknown flag: ${arg}`);
    else positional.push(arg);
  }
  return { positional, flags };
}

/**
 * A filesystem-backed `AssetReader`. The engine resolves every declared model
 * file relative to the manifest, which for the oracle dump means a path like
 * `MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json` while the file
 * actually sits next to the manifest — so a missing direct hit falls back to the
 * basename.
 */
function fsAssetReader(root) {
  const locate = (relativePath) => {
    const direct = join(root, relativePath);
    if (existsSync(direct)) return direct;
    const flat = join(root, basename(relativePath));
    if (existsSync(flat)) return flat;
    return null;
  };
  const need = (relativePath) => {
    const file = locate(relativePath);
    if (!file) throw new Error(`model file not found: ${relativePath} (in ${root})`);
    return file;
  };
  return {
    async exists({ relativePath }) {
      return locate(relativePath) !== null;
    },
    async readText({ relativePath }) {
      return readFileSync(need(relativePath), "utf8");
    },
    async readJson({ relativePath }) {
      return JSON.parse(readFileSync(need(relativePath), "utf8"));
    },
    async readBuffer({ relativePath }) {
      return new Uint8Array(readFileSync(need(relativePath)));
    },
  };
}

/**
 * The SentencePiece tokenizer is the one browser-only piece of the engine: the
 * vendored bundle is an emscripten module that talks to a Web Worker. In Node we
 * evaluate it in a `vm` context with just enough of the worker surface shimmed,
 * then drive the exact same request/response protocol the browser uses.
 */
function createNodeTokenizer() {
  const bundlePath = fileURLToPath(
    new URL("../src/workers/vendor/tokenizerSandbox.js", import.meta.url),
  );
  const sourceUrl = `file://${bundlePath}`;
  const code = readFileSync(bundlePath, "utf8").replaceAll(
    "import.meta.url",
    JSON.stringify(sourceUrl),
  );

  const listeners = [];
  const sandbox = {};
  Object.assign(sandbox, {
    globalThis: sandbox,
    console,
    Buffer,
    process: Object.assign(Object.create(process), { type: "renderer", argv: process.argv }),
    self: sandbox,
    window: { addEventListener: (type, fn) => type === "message" && listeners.push(fn) },
    document: {},
    location: { href: sourceUrl },
    importScripts: () => {},
    setTimeout,
    clearTimeout,
    setInterval,
    clearInterval,
    queueMicrotask,
    performance,
    TextEncoder,
    TextDecoder,
    URL,
    WebAssembly,
    fetch,
    crypto: globalThis.crypto,
    atob: (s) => Buffer.from(s, "base64").toString("binary"),
    btoa: (s) => Buffer.from(s, "binary").toString("base64"),
  });
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox, { filename: "tokenizerSandbox.js" });

  let modelKey = null;
  const call = (action, body) =>
    new Promise((resolve, reject) => {
      const requestId = `cli-${Date.now()}-${Math.random().toString(36).slice(2)}`;
      const event = {
        data: {
          source: "nano-reader-tokenizer-sandbox",
          type: "request",
          requestId,
          action,
          ...body,
        },
        source: {
          postMessage: (message) => {
            if (!message || message.requestId !== requestId) return;
            if (message.ok) resolve(message.data);
            else reject(new Error(message.error || "tokenizer request failed"));
          },
        },
      };
      for (const listener of listeners) void listener(event);
    });

  return {
    async load(bytes) {
      modelKey = `moss::${bytes.length}`;
      await call("loadTokenizer", {
        modelKey,
        tokenizerBase64: Buffer.from(bytes).toString("base64"),
      });
      return modelKey;
    },
    async encode(key, text) {
      const ids = await call("encodeText", { modelKey: key, text: String(text ?? "") });
      return Array.isArray(ids) ? ids : [];
    },
    loadedKey: () => modelKey,
  };
}

function parseVoiceIndex(value, fallback = 0) {
  if (value === undefined || value === null || value === "") return fallback;
  const parsed = Number.parseInt(String(value), 10);
  if (!Number.isInteger(parsed) || parsed < 0)
    throw new Error(`expected a non-negative voice index, got "${value}"`);
  return parsed;
}

function concatChannels(chunks, channelCount) {
  const totals = new Array(channelCount).fill(0);
  for (const chunk of chunks)
    for (let c = 0; c < channelCount; c += 1) totals[c] += chunk[c]?.length ?? 0;
  const out = totals.map((length) => new Float32Array(length));
  const offsets = new Array(channelCount).fill(0);
  for (const chunk of chunks)
    for (let c = 0; c < channelCount; c += 1) {
      const data = chunk[c];
      if (data) {
        out[c].set(data, offsets[c]);
        offsets[c] += data.length;
      }
    }
  return out;
}

/**
 * Drop the near-silent tail the greedy decoder leaves behind (it keeps
 * generating after the sentence, without emitting the end token). A short tail
 * is kept so the last phoneme is not clipped.
 */
function trimTrailingSilence(channels, sampleRate, threshold = 0.002, keepSeconds = 0.25) {
  const frames = channels[0]?.length ?? 0;
  let last = frames - 1;
  while (
    last > 0 &&
    Math.abs(channels[0]?.[last] ?? 0) <= threshold &&
    Math.abs(channels[1]?.[last] ?? 0) <= threshold
  ) {
    last -= 1;
  }
  const end = Math.min(frames, last + 1 + Math.round(keepSeconds * sampleRate));
  return channels.map((channel) => channel.subarray(0, end));
}

async function main() {
  const { positional, flags } = parseArgs(process.argv.slice(2));
  if (flags.help) {
    console.log(USAGE);
    return;
  }

  const modelDir = resolve(flags.modelDir ?? process.env.MOCKBIRD_MODEL_DIR ?? "");
  if (!modelDir || !existsSync(join(modelDir, "browser_poc_manifest.json"))) {
    console.error(USAGE);
    throw new Error(
      `no model directory: set --model-dir or MOCKBIRD_MODEL_DIR to a folder with ` +
        `browser_poc_manifest.json`,
    );
  }

  const threadCount = flags.threads ? Number(flags.threads) : 4;
  const runtime = new WgpuTtsRuntime({
    logger: (message) => console.error(`[engine] ${message}`),
    assetReader: fsAssetReader(modelDir),
    tokenizer: createNodeTokenizer(),
    gpu: create([]),
  });
  await runtime.configure({ modelPath: modelDir, threadCount });
  await runtime.ensureManifestLoaded();

  // size the KV ring and the decode loop before the engine is built
  const maxFrames = flags.maxFrames ? Number.parseInt(flags.maxFrames, 10) : 160;
  if (!Number.isInteger(maxFrames) || maxFrames <= 0)
    throw new Error(`--max-frames must be a positive integer, got "${flags.maxFrames}"`);
  if (runtime.manifest) runtime.manifest.generation_defaults.max_new_frames = maxFrames;

  if (flags.list) {
    console.log("Built-in voices:");
    runtime.listBuiltinVoices().forEach((voice, index) => {
      console.log(
        `  ${String(index).padStart(2)}  ${voice.voice.padEnd(10)} ${voice.group.padEnd(16)} ${voice.display_name}`,
      );
    });
    return;
  }

  const text = String(flags.text ?? positional[0] ?? "").trim();
  if (!text) {
    console.error(USAGE);
    throw new Error("nothing to say — pass a paragraph of text");
  }

  const voice = parseVoiceIndex(flags.voice ?? positional[1], 0);
  const out = resolve(flags.out ?? positional[2] ?? `synth-${voice}.wav`);

  console.error(`[cli] loading engine (model dir: ${modelDir}) …`);
  await runtime.ensureSynthesisLoaded();
  await runtime.warmup();
  await runtime.ensureTokenizerLoaded();

  const voices = runtime.listBuiltinVoices();
  if (voice < 0 || voice >= voices.length)
    throw new Error(
      `voice index ${voice} is out of range (0..${voices.length - 1}); use --list`,
    );
  const selected = voices[voice];
  console.error(`[cli] voice ${voice}: ${selected.voice} (${selected.group})`);
  console.error(`[cli] synthesizing ${text.length} chars …`);

  runtime.setProfilingEnabled(true);
  const chunks = [];
  const started = Date.now();
  await runtime.synthesizeVoiceClone({
    text,
    promptAudioCodes: selected.prompt_audio_codes,
    streaming: true,
    onPreparedText: (info) => {
      console.error(`[cli] normalised: ${info.text}`);
      for (const warning of info.warnings) console.error(`[cli] warning: ${warning}`);
    },
    onAudioChunk: async (chunk) => {
      if (chunk.isPause) return; // chunked speech: drop the inter-chunk pauses
      chunks.push(chunk.chunkData);
    },
  });

  if (chunks.length === 0) throw new Error("the engine produced no audio");
  const sampleRate = 48_000;
  const raw = concatChannels(chunks, 2);
  const channels = flags.noTrim ? raw : trimTrailingSilence(raw, sampleRate);
  const frameCount = channels[0].length;
  const blob = encodeWav(channels, sampleRate);
  writeFileSync(out, Buffer.from(await blob.arrayBuffer()));

  const profile = runtime.getProfileSnapshot();
  const frames = profile?.counters?.["generation.generated_frames"] ?? 0;
  const seconds = frameCount / sampleRate;
  console.error(
    `[cli] ${frames} frames, ${seconds.toFixed(2)} s of audio in ` +
      `${((Date.now() - started) / 1000).toFixed(1)} s`,
  );
  console.error(`[cli] wrote ${out}`);
}

main().catch((error) => {
  console.error(`error: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
