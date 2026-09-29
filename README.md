# Mockbird

**Clone a voice and generate speech entirely in your browser.**

Mockbird runs [MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) — a
0.1B-parameter multilingual text-to-speech model — fully on-device using the
official ONNX export and ONNX Runtime Web. Your voice recordings never leave the
machine, there is no server inference, and the app keeps working offline after
the one-time model download.

- **Voice cloning** — record or upload 8–15 s of clean speech, get a reusable
  voice
- **18 built-in voices** — included with the model manifest, no extra download
- **20 languages**, 48 kHz stereo output
- **Streaming** — audio starts in about a second while the rest is still decoding
- **Private by design** — reference audio is encoded to audio codes on-device

---

## Quick start

```bash
pnpm install          # also copies the ONNX Runtime wasm + tokenizer worker
pnpm dev              # https://localhost:8600  (webpack, see ADR 0001)
```

Then open <https://localhost:8600>:

1. **Model** — press *Download* (≈ 763 MB, cached in your browser forever).
2. **Voices** — clone your own voice, or pick a built-in one.
3. **Studio** — type anything and generate.

The download is the slow part the first time. After that the app starts
instantly and runs offline.

> **Browser support.** Chrome/Edge 108+ is the reference target: the app needs the
> Origin Private File System to cache ~763 MB, cross-origin isolation for
> multi-threaded wasm, and `MediaRecorder` for in-browser recording. Firefox and
> Safari work with reduced functionality (no persistent weight cache / no
> in-page recording; upload a file instead).

### Why dev runs over HTTPS

`crossOriginIsolated` is what unlocks `SharedArrayBuffer`, and ONNX Runtime Web
needs it for multi-threaded wasm. Browsers only grant a secure context to
`https://` (or `http://localhost`), so opening the dev server from another device
on the LAN would silently drop the engine to a single thread. `pnpm dev`
therefore generates a self-signed certificate with
[mkcert](https://github.com/FiloSottile/mkcert) and serves HTTPS.

The CA, leaf certificate and key live in the git-ignored `mkcert/` folder and are
regenerated automatically when missing or older than 30 days.

| Command | What it does |
| --- | --- |
| `pnpm dev` | HTTPS on port 8600 (certificates generated on demand) |
| `pnpm dev:http` | plain HTTP (fine when working only on this machine) |
| `pnpm certs` | generate/refresh the certificates only |
| `pnpm certs:force` | regenerate even if they look fresh |
| `pnpm certs:trust` | add the CA to the system trust store (needs sudo) |

On first run the CA is created but adding it to the system trust store needs
root, so your browser will warn once. Either run `pnpm certs:trust` (Linux:
installs into `/usr/local/share/ca-certificates`; macOS: Keychain; Windows:
certutil) or click through the warning — the app works either way, TLS is just
encrypted-but-untrusted in the latter case. The certificate covers `localhost`,
`127.0.0.1`, `::1` and every detected LAN address; pass extra names with
`node scripts/dev-certs.mjs --host my-host.local`.

## Scripts

| Command | What it does |
| --- | --- |
| `pnpm dev` | Dev server on port **8600**, forced to webpack |
| `pnpm build` | Production build (Turbopack, as Vercel runs it) |
| `pnpm build:webpack` | Parity build with webpack (used in CI) |
| `pnpm lint` / `pnpm lint:fix` | Biome check (lint + format + import order) |
| `pnpm format` | Biome formatter |
| `pnpm typecheck` | `next typegen` + `tsc --noEmit` |
| `pnpm verify` | typecheck → lint → build |
| `pnpm e2e` | Headless browser end-to-end check (see below) |
| `pnpm sync:vendor` | Re-vendor the upstream runtime and re-apply the patches |

## Voice cloning tips

The **Voices** tab hands you a paragraph to read, in the language your voice
actually speaks (all 19 languages MOSS-TTS-Nano documents). A fixed script beats
"say anything": the same phonemes and prosody every time make the audio codes
capture the speaker rather than the topic, and two recordings become comparable.

- Read at a **natural pace** — the card shows an estimated duration (11–19 s)
- One speaker, quiet room, no music or fan, 20–30 cm from the mic
- **Hear it first** plays the paragraph through a built-in voice, so you know the
  intended phrasing before you read it

## How it works

```
Browser tab (Next.js 16, static on Vercel)
│
├─ UI (React 19 + shadcn)            jotai atoms
│   /model   weight downloader, per-file progress, threads, storage
│   /voices  recorder / uploader → codec encode → voice library
│   /studio  text composer → chunked synthesis → streaming player → WAV
│   /settings runtime diagnostics, cached files, profile, log
│
├─ AudioWorklet playback      (public/stream-player.worklet.js)
│
└─ TTS worker (module worker)
    ├─ vendored MOSS-TTS-Nano browser runtime   (src/lib/tts/vendor)
    ├─ onnxruntime-web 1.24.3 (wasm, SIMD + threads, /public/ort)
    ├─ OPFS model store + resumable HF downloader
    └─ SentencePiece tokenizer worker          (public/tokenizer.worker.js)
```

Everything expensive happens in a dedicated Web Worker, so a 670 MB weight load
or a multi-second decode loop never blocks the UI or audio playback.

### The pipeline

```
clone:  file → decodeAudioData → 48 kHz stereo → moss_audio_tokenizer_encode → audio codes

synth:  text → normalizer → SentencePiece → ≤75-token chunks
          → prefill → (decode_step ⇄ local decoder) per 12.5 Hz frame
          → streaming moss_audio_tokenizer_decode_step → PCM
          → AudioWorklet playback + WAV export
```

## Deploying to Vercel

The app is fully client-side, so the deployment is just a Next.js build:

```bash
vercel            # or connect the repo; build command: pnpm build
```

Two things are configured in `next.config.ts` and must not be dropped:

- `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp` on every route — without them
  `crossOriginIsolated` is false, `SharedArrayBuffer` is unavailable and
  ONNX Runtime silently falls back to a single thread.
- `typedRoutes: true`.

Weights are **not** part of the bundle. They stream from the Hugging Face CDN on
first run and are cached in the visitor's browser. To serve them from your own
CDN/mirror, set:

```bash
NEXT_PUBLIC_MODEL_BASE_URL=https://your-cdn.example.com/models
```

The mirror must keep the upstream directory layout
(`MOSS-TTS-Nano-100M-ONNX/…`, `MOSS-Audio-Tokenizer-Nano-ONNX/…`) and allow
cross-origin reads (`Access-Control-Allow-Origin: *`) plus HTTP range requests.

## Model weights

| File | Size | Purpose |
| --- | --- | --- |
| `moss_tts_global_shared.data` | 441 MB | prefill + decode step (fp32) |
| `moss_tts_local_shared.data` | 230 MB | local transformer decoder |
| `moss_audio_tokenizer_decode_shared.data` | 44 MB | audio codes → waveform |
| `moss_audio_tokenizer_encode.data` | 45 MB | reference audio → audio codes (**cloning**) |
| 6 × `*.onnx` graphs | 2.5 MB | graph + IO names |
| `tokenizer.model` | 0.5 MB | SentencePiece |
| `browser_poc_manifest.json` | 0.5 MB | config + 18 built-in voices with pre-computed codes |
| ONNX Runtime wasm | 12 MB | served from `public/ort/` |

A future optimisation (not implemented) is re-exporting the two big fp32 weight
files as fp16 or int8-dynamic, which would roughly halve or quarter the download.

## Architecture decisions

- [ADR 0001 — bundler parity (webpack dev, Turbopack on Vercel)](docs/adr/0001-bundler-parity.md)
- [ADR 0002 — static export vs. Vercel deployment](docs/adr/0002-static-export-vs-vercel.md)
- [ADR 0003 — vendored upstream runtime](docs/adr/0003-vendored-runtime.md)

## Upstream / licensing

Mockbird is MIT-licensed. It vendors code from **MOSS-TTS-Nano**
(Apache-2.0, © OpenMOSS) and adapts the browser port from
**MOSS-TTS-Nano-Reader**, which ships without a license file — see
[src/lib/tts/vendor/README.md](src/lib/tts/vendor/README.md) for the exact list
of local edits and the open licensing question. Please only clone voices you have
permission to use.

MOSS-TTS-Nano · arXiv · Hugging Face · ModelScope · Discord
