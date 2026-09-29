# Mockbird — Web-Only Voice Cloning & TTS App (MOSS-TTS-Nano)

**Working title:** Mockbird · 声音工作台
**Status:** ✅ **built** — see [§0 Implementation status](#0-implementation-status) for what shipped and where the plan changed.
**Stack:** **Next.js 16.3** (App Router, webpack in dev / Turbopack on Vercel) · **React 19.3** · TypeScript · Tailwind CSS v4 · shadcn/ui · jotai v3 · **Biome** · onnxruntime-web · Vercel

---

## 0. Implementation status

The app is built and working end-to-end against the real model stack. What is
verified working in a browser: weight download → OPFS, engine load, tokenizing,
streaming synthesis with built-in voices, voice cloning (upload → codec encode →
stored → reused), WAV export, offline-capable caching. See `README.md` for usage
and `docs/adr/` for the design decisions.

### What changed vs. the plan below

| Plan said | Shipped | Why |
| --- | --- | --- |
| Sidebar shell with 4 routes (`/studio` `/voices` `/model` `/settings`) | **One centred glass panel with 3 tabs**: `/` Generate, `/clone` Clone, `/settings` Settings (model + diagnostics merged; the old routes redirect) | Casual-user brief: hide the machinery, keep three decisions — download, pick a voice, press a button. Design language (zinc neutrals, violet ring, `.glass-control` / `.apple-panel` / `.ambient-bg`, animated blob background, light/dark/system toggle) copied from `~/shoufa`. Thread budget, per-file logs, capability report and profiler moved behind disclosures. |
| Port `mossTtsRuntime.ts` by hand | Vendored the upstream browser runtime + **11 auditable patches** in `scripts/vendor-patches.json`, re-appliable with `pnpm sync:vendor` | 2 400 lines of tensor-layout detail; re-implementing was the single biggest risk. License caveat tracked in §10.1. |
| Model store, downloader, worker, player, 4 pages | As planned | — |
| ESLint (flat config) | **Biome** (`biome.json`, `pnpm lint`) | Project decision: Biome for lint + format. |
| `jotai/utils` `atomWithStorage` | Local `src/state/atomWithStorage.ts` | `jotai/utils` pulls a second jotai v3 copy into the dev build ("Detected multiple Jotai instances"). |
| Worker assets via `new URL(..., import.meta.url)` | `public/stream-player.worklet.js` + `public/tokenizer.worker.js`, loaded by string URL | webpack treats a `.ts` worklet reached through `new URL` as a raw asset and the browser rejects its MIME type; `public/` behaves identically under webpack and Turbopack. See ADR 0001. |
| Encoder weights optional (45 MB) | Same intent, implemented as a **lazy ORT session** + lazy warmup | Upstream creates the encoder session during `ensureSynthesisLoaded`; patched so cloning weights are genuinely optional. |
| Cross-origin isolation via `next.config.ts` headers | Same | `output: "export"` dropped because it does not support `headers()` — see ADR 0002. |
| Sidebar shell, 4 routes (`/studio` `/voices` `/model` `/settings`) | **One centred glass panel with 3 tabs**: `/` Generate, `/clone` Clone, `/settings` Settings (model + diagnostics merged). Old routes redirect. Design language (zinc neutrals, violet ring, `.glass-control`/`.apple-panel`/`.ambient-bg`, animated blob background, theme toggle) copied from `~/shoufa`. Technical controls moved behind disclosures. | Casual-user brief: hide the machinery, keep three decisions — download, pick a voice, press a button. |
| Default 2 ORT threads | 4 (capped by the UI at `hardwareConcurrency - 1`) | Faster on multi-core machines. |

### Bugs found and fixed during the build (worth remembering)

1. Worker messages posted before React subscribed were dropped → buffered backlog
   in `TtsWorkerClient`, and `probe` now republishes the engine state.
2. `tokenizerSandboxFrame?.contentWindow` stayed in upstream code after the
   sandbox moved to a Worker; it is always falsy for a `Worker`, so a **fresh
   tokenizer Worker (and a lost model) was created on every request**.
3. The OPFS asset reader returned an `ArrayBuffer`; the runtime iterates
   `.length` when base64-encoding the tokenizer → empty model → "Tokenizer model
   is not loaded". It returns a `Uint8Array` now (as upstream does).
4. `channels[0].length` was read *after* the buffer had been transferred to the
   AudioWorklet, so the sample count was always 0.
5. The vendored runtime is rebuilt by `pnpm sync:vendor`; any hand edit must be
   turned into a patch or it is silently lost (this happened once).

### Still open

- **Time-to-first-audio** is ~0.8 s in a warm session but several seconds on a
  cold engine (session creation dominates). The profiler is wired up (Model tab →
  *Runtime profiling log*, results on the Settings page) to chase it.
- **Licensing of the Reader-derived files** (§10.1).
- **fp16 / int8 weight re-export** to cut the 763 MB download (§10.2).
- Keyboard/AT coverage pass beyond the basics.

---

## 1. Research summary: MOSS-TTS-Nano

**Source:** https://github.com/OpenMOSS/MOSS-TTS-Nano (Apache-2.0, 4.4k★, MOSI.AI / OpenMOSS)

| Property | Value |
|---|---|
| Size | 0.1B params (100M) |
| Architecture | Pure autoregressive: Audio Tokenizer (Cat) + tiny LLM, then a small local transformer decoder |
| Output format | **48 kHz, 2-channel (stereo)**, 12.5 Hz frame rate (downsample 3840) |
| Audio codes | 16 RVQ codebooks × 1024 entries → `n_vq=16`, row width 17 with text token |
| Languages | 20 (zh, en, ja, ko, de, fr, es, pt, ru, ar, fa, it, pl, cs, da, sv, el, hu, tr, …) |
| Latency | Realtime streaming, runs on 4-core CPU; smooth on a single M4 core |
| Long text | Auto chunked voice cloning (`voice_clone_max_text_tokens=75`/chunk) |
| Weights | [MOSS-TTS-Nano (HF)](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano), [MOSS-Audio-Tokenizer-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano) |

### Three runtimes ship upstream

1. **PyTorch** — `infer.py`, `app.py` (FastAPI web demo on :18083, endpoints `/api/generate`, `/api/generate-stream/start` + `/audio` + `/result`).
2. **ONNX CPU (recommended)** — `infer_onnx.py`, `app_onnx.py`, `moss-tts-nano serve --backend onnx`. No PyTorch at inference, ~2× faster. Weights: `MOSS-TTS-Nano-100M-ONNX` + `MOSS-Audio-Tokenizer-Nano-ONNX`.
3. **In-browser ONNX** — [MOSS-TTS-Nano-Reader](https://github.com/OpenMOSS/MOSS-TTS-Nano-Reader) runs the model **inside the browser** (Chrome/Edge extension) with onnxruntime-web 1.24.3 (`ort-wasm-simd-threaded.wasm`, 12.4 MB), no local service. Files: `browser_onnx_runtime.js` (2380 lines), `browser_model_store.js` (OPFS cache + HF downloader), `tokenizer_sandbox.js` (SentencePiece compiled to wasm-in-JS), 18 built-in voices with **precomputed prompt codes**.

> **This is the critical finding:** item 3 already proves the exact product we want — browser-only voice cloning + synthesis. Our app is essentially a Next.js UI + OPFS model manager wrapped around that runtime, without the extension shell.

### The inference pipeline (what we must reproduce client-side)

```
clone:  File/Blob (any format)
          → decodeAudioData (OfflineAudioContext, resample to 48k stereo)
          → moss_audio_tokenizer_encode.onnx  →  audio_codes [1, T, 16]  (T = dur × 12.5)
          → store as "voice"

synth:  text → JS robust normalizer → SentencePiece encode
          → split into ≤75-token chunks
          per chunk: build rows [user_prompt_prefix, audio_start,
                                 (audio_user_slot, 16 codes)*, …text tokens…, audio_end]
            → moss_tts_prefill.onnx  → global_hidden + KV
            → loop: moss_tts_decode_step.onnx (1 token/frame, 16 logits)
                   + moss_tts_local_decoder.onnx / local_cached_step.onnx
                   + local_fixed_sampled_frame.onnx (sampling: T=0.8, top_p=.95, top_k=25, rep=1.2)
                   until audio_end or max_new_frames (375 ≈ 30 s)
            → moss_audio_tokenizer_decode_step.onnx (streaming, cached KV) → PCM chunks
          → concatenate (inter-chunk pause inserted automatically)
          → AudioWorklet playback + WAV export
```

Streaming contract in the reference runtime: `synthesizeVoiceClone({text, voiceName|promptAudioCodes, doSample, sampleMode, streaming, onAudioChunk, isCancelled, voiceCloneMaxTextTokens, enableNormalizeTtsText, enableWeTextProcessing})`; each chunk is `{channels, sampleRate, chunkData: Float32Array[], isPause}`.

### Model payload that must reach the browser (measured from HF API)

| File | Size | Needed for |
|---|---|---|
| `moss_tts_global_shared.data` | **440.8 MB** | prefill + decode (fp32 weights) |
| `moss_tts_local_shared.data` | **229.7 MB** | local decoder |
| `moss_audio_tokenizer_encode.data` | 44.5 MB | **voice cloning** (user audio → codes) |
| `moss_audio_tokenizer_decode_shared.data` | 44.2 MB | codes → waveform |
| 6 × `*.onnx` graphs | 2.5 MB | graph + IO names |
| `tokenizer.model` (SentencePiece) | 0.47 MB | text → ids |
| `browser_poc_manifest.json` | 0.5 MB | 18 built-in voices **with precomputed prompt codes** + all tts/prompt/generation config |
| `tts_browser_onnx_meta.json`, `codec_browser_onnx_meta.json` | 0.03 MB | IO names, model_config, codec_config |
| ORT wasm (`ort-wasm-simd-threaded.wasm`) | 12.4 MB | runtime (self-host) |
| **Total** | **≈ 763 MB** (672 MB TTS + 91 MB codec) | |

**Hosting check (verified live):** `huggingface.co/.../resolve/main/<file>` → 302 → `us.aws.cdn.hf.co` with `access-control-allow-origin: *`, `accept-ranges: bytes`, strong `ETag`. So direct cross-origin browser download works, **including under COEP `require-corp`**, with HTTP-range resume.

---

## 2. Key architectural decision: 100% client-side inference

Vercel cannot host this model:
- Python runtime on Vercel Fluid has a ~250 MB function bundle limit; models are 763 MB.
- ORT/PyTorch CPU inference would take 10–60 s+ per request and burn function time on every call.
- The model is a *stateful streaming* decoder — bad fit for request/less serverless.

So: **Vercel hosts only the app; all inference runs in the user's browser; weights are fetched once from the HF CDN and cached in OPFS.** Result: zero GPU/CPU cost, user data (voice samples) never leaves the device, works offline after first load.

| Option | Verdict |
|---|---|
| **A. Browser-only ONNX + Vercel** ✅ | Chosen. Proven upstream, ~0 infra cost, privacy-first, offline after first load. Cost: 763 MB first-run download. |
| B. Python/ONNX backend on Railway/Fly + Next frontend | Fast per-request on warm instances, but needs always-on CPU, audio leaves the device, no offline. Keep as **optional later "Cloud mode"** toggle (proxy to a self-hosted `app_onnx.py`). |
| C. Vercel AI Gateway / serverless ONNX | Rejected (bundle size + cold starts). |
| D. WebGPU EP | Rejected for v1: only `executionProviders: ["wasm"]` is validated upstream. Revisit after 1.0. |

---

## 3. Tech stack (versions verified against npm + Next 16 docs)

| Layer | Choice | Version | Why |
|---|---|---|---|
| Framework | **Next.js App Router** | `16.3.6` (latest) | Requested. Turbopack is the default bundler in 16; we override for dev (below). |
| UI runtime | **React / React DOM** | `19.3.0` | Requested. Next 16's App Router ships React 19.2+ features: `Activity`, `useEffectEvent`, `<ViewTransition>`. |
| Types | `@types/react`, `@types/react-dom` | `^19` | Next 16 requires React `^19.0.0` peer. |
| Language | TypeScript | `>=5.1` | Next 16 minimum. |
| Node (dev/CI) | Node | `>=20.9` | Next 16 minimum (18 dropped). |
| Styles | **Tailwind CSS v4** | `4.3.3` | CSS-first `@theme inline`, no `tailwind.config.js` needed. |
| Components | **shadcn/ui** (CLI `4.21.0`) | — | Copied into `components/ui`; works with Tailwind v4 + React 19. |
| State | **jotai v3** | `3.0.0` | Requested. v3 = module-first ESM, ES2020; drops `atomFamily`/`loadable`/`setSelf`/CJS. We only use `atom`, `atomWithStorage`, `selectAtom`, `Provider`, `useHydrateAtoms` — all present. |
| Inference | **onnxruntime-web** (wasm, SIMD + threads) | pin exact, start at `1.24.3` (what upstream validated); re-validate before bumping | Self-host the `.wasm` in `public/ort/` — a third-party CDN would break COEP unless it sends CORP. |
| Threading | Dedicated **module Web Worker** | — | Keeps the ~670 MB weight decode + decode loop off the main thread. `ort.env.wasm.proxy = false` (as upstream). |
| Playback | **AudioWorklet** ring buffer | — | Constant-rate drain of variable-rate 48 kHz stereo chunks; the only gapless streaming option. |
| Persistence | **OPFS** (`navigator.storage.getDirectory`) + `navigator.storage.persist()` | — | 763 MB of weights + voice codes; survives reloads. Fallback: Cache Storage → IndexedDB. |
| Lint | **ESLint CLI + flat config** (`eslint.config.mjs`, `eslint-config-next`) | — | `next lint` was **removed** in Next 16. |
| PWA | `app/manifest.ts` + Serwist (Turbopack examples exist) | — | Offline app shell. Offline *inference* comes from OPFS, not the SW. |

### Bundler strategy: webpack in dev, Turbopack on Vercel

Next 16 made Turbopack the default for both `next dev` and `next build`; both can be overridden per-command.

| Task | Bundler | Command | Reason |
|---|---|---|---|
| Local dev | **webpack** | `next dev --webpack` | **Turbopack dev exceeded our RAM budget on this project.** Forced opt-out; do not remove. |
| Local/CI build (parity check) | webpack | `next build --webpack` | Reproduces the dev graph; used in CI to catch bundler-only breakages early. |
| Vercel build | Turbopack (default) | `next build` | Faster, and Vercel is where it works well. |
| Escape hatches | — | `--turbopack` / `--webpack` | If Vercel's Turbopack build ever mis-bundles the worker/worklet, set the Vercel build command to `next build --webpack`. |

**Dev/build bundler-parity guardrails** (dev = webpack, prod = Turbopack, so both must work):
1. Use only bundler-agnostic patterns for the non-React payload:
   ```ts
   const worker = new Worker(new URL('../workers/tts.worker.ts', import.meta.url), { type: 'module' })
   const workletUrl = new URL('../audio/stream-player.worklet.ts', import.meta.url)
   ```
   Both webpack 5 and Turbopack support `new URL(..., import.meta.url)` for workers/assets.
2. **No bundler-specific magic comments** (`webpackIgnore`, `turbopackIgnore`) in app code — CI greps for them so a one-off fix can't silently diverge dev and prod.
3. Bundle `onnxruntime-web` normally (pure ESM, worker-safe) and only load the `.wasm` binary from `/ort/` via `ort.env.wasm.wasmPaths`. If bundling ORT misbehaves in one bundler, fall back to a URL import with **both** ignore comments in one block — and record it in `docs/adr/0001-bundler-parity.md`.
4. **P0 spike must verify both bundlers emit** the worker chunk and the worklet asset, and that the page is `crossOriginIsolated` in dev (webpack dev serves `next.config.ts` headers fine).
5. `webpack` config in `next.config.ts` is still allowed and is **ignored by `next build` (Turbopack)** — if we ever add one, dev keeps it and prod silently loses it. Prefer none.

### Cross-origin isolation (required for wasm threads)

```ts
// next.config.ts — single source of truth; works in `next dev` and on Vercel
const nextConfig: NextConfig = {
  async headers() {
    return [{
      source: '/:path*',
      headers: [
        { key: 'Cross-Origin-Opener-Policy', value: 'same-origin' },
        { key: 'Cross-Origin-Embedder-Policy', value: 'require-corp' },
      ],
    }]
  },
}
```

Why not `output: 'export'`: **static export does not support `headers()`** (Next 16 docs list Headers as unsupported), so a pure-CDN export would force duplicated header config in `vercel.json` *and* a dev proxy. Instead we deploy a normal Next 16 app: every route here is a client component with no server data, so all pages prerender to static HTML at build time and Vercel serves them from the CDN. If we later want a pure static host, switching to `output: 'export'` + `vercel.json` headers is a config-only change.

Consequences to design around: `crossOriginIsolated === true` gates `SharedArrayBuffer` (multi-thread ORT); every third-party subresource needs CORS/CORP (HF CDN is `*` ✅); popups break under `same-origin` (use `same-origin-allow-popups` if we ever add OAuth). Note `experimental.useOffline` (connectivity detection/retry) is **experimental and explicitly not recommended for production** — we use our own `navigator.onLine` badge instead.

### React 19.3 features we actually use

| API | Use in Mockbird |
|---|---|
| `<Activity mode="hidden">` | Keeps the **engine provider, AudioContext and live waveform panels warm** while the user navigates `/model ↔ /voices ↔ /studio`. No reload of the 670 MB session, no interrupted playback. Root layout holds the provider; `<Activity>` wraps heavy always-alive panels. |
| `useEffectEvent` | Worker message handling inside effects without re-subscribing on every jotai atom change (e.g. `generationAtom` updates while streaming). |
| `<ViewTransition>` | Subtle route transitions and idle→generating→done state changes; gate on `prefers-reduced-motion`. |
| `useSyncExternalStore` (via jotai) | Fine under React 19; no manual `use`/tearing issues. |
| `useHydrateAtoms` | Restores persisted `synthParamsAtom` / settings without a flash of defaults. |

### Next 16 breaking changes that touch us (and what we do)

| Change | Impact here |
|---|---|
| Turbopack default | Handled by `--webpack` in dev (see table above). Turbopack config moved out of `experimental` → top-level `turbopack: {}`. |
| Async Request APIs only (`cookies`/`headers`/`params`/`searchParams` sync access **removed**) | No impact: zero server data fetching. Use generated `PageProps<'/studio'>` helpers from `next typegen` if we add params later. |
| `next lint` removed | Use `eslint .` with flat config; `npm run lint` in CI. |
| `middleware` → `proxy` (file rename) | Not used. COOP/COEP come from `headers()`. |
| Runtime configuration removed | All config via env vars baked at build (`NEXT_PUBLIC_MODEL_BASE_URL`, …). |
| `images.domains` deprecated, `qualities`/`imageSizes`/`minimumCacheTTL` defaults changed, local-IP images blocked | No `next/image` use for user audio/waveforms (canvas + object URLs). Only static icons/OG. |
| `next dev` writes a `nextjs-agent-rules` block into `AGENTS.md` | Expected; commit `AGENTS.md` so the tree stays clean. |
| AMP removed, `experimental_ppr` route config removed | N/A. |
| Browser baseline: Chrome 111+, Edge 111+, Firefox 111+, Safari 16.4+ | Matches the WASM/SIMD/OPFS floor we need; still feature-detect OPFS + `crossOriginIsolated` + AudioWorklet. |

---

## 4. Architecture

```
┌──────────────────────── Browser tab (Next.js 16 app on Vercel) ───────────────────────────────┐
│                                                                                              │
│  UI (React 19 + shadcn)         jotai v3 atoms                                               │
│  ├─ /model     Model Lab        ├─ modelStatusAtom        (idle|downloading|ready|error)     │
│  ├─ /voices    Voice Library    ├─ modelFilesAtom / downloadProgressAtom  (per-file bytes)   │
│  ├─ /studio    Studio           ├─ voicesAtom             (OPFS-backed list)                  │
│  └─ /settings  Settings         ├─ voiceDraftAtom         (recording/upload)                  │
│                                ├─ synthParamsAtom         (persisted)                          │
│  root layout (always mounted)   └─ generationAtom         (idle|running|paused|error+stats)   │
│   └─ <EngineProvider>                                                              ▲          │
│        <Activity mode="hidden">  keeps AudioContext + waveform warm   ┌──────────┴────────┐    │
│                                                                 │   postMessage RPC    │    │
│  AudioWorklet  ← PCM chunks (transferables) ────────────────────┘                      │    │
│  playback ring buffer                          ▲                                       │    │
│  ┌────────────────────────────────────────────┴────────────────────────────────────┐     │    │
│  │ Web Worker  src/workers/tts.worker.ts  (module worker)                        │     │    │
│  │  ┌──────────────────────────────────────────────┐                             │     │    │
│  │  │ mossTtsRuntime.ts  (ported from Apache-2.0   │                             │     │    │
│  │  │ reference: browser_onnx_runtime.js +          │                             │     │    │
│  │  │ ort_cpu_runtime.py) — pure TS module          │                             │     │    │
│  │  │  · text normalizer · SentencePiece bridge      │                             │     │    │
│  │  │  · prefill/decode loop · frame sampling        │                             │     │    │
│  │  │  · streaming codec decode                      │                             │     │    │
│  │  └──────────────────────────────────────────────┘                             │     │    │
│  │  ┌──────────────────┐  ┌─────────────────────────────┐                         │     │    │
│  │  │ onnxruntime-web  │  │ modelStore.ts (OPFS)        │                         │     │    │
│  │  │ wasm SIMD ×N     │  │ fetch HF CDN w/ resume,     │                         │     │    │
│  │  │ wasmPaths:/ort/  │  │ ETag revalidate, persist()  │                         │     │    │
│  │  └──────────────────┘  └─────────────────────────────┘                         │     │    │
│  │  ┌──────────────────────────────┐                                              │     │    │
│  │  │ tokenizer.worker.ts  SentencePiece wasm (0.47 MB model)                     │     │    │
│  │  └──────────────────────────────┘                                              │     │    │
│  └───────────────────────────────────────────────────────────────────────────────┘     │    │
│                          OPFS: /models/{tts,codec}/*  ·  /voices/<id>/*                     │    │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Data flow for a generation (clone voice → speak):**
1. User records/uploads reference audio on `/voices` → worker `encodeReferenceAudio` (codec encode) → `audio_codes` saved to OPFS with metadata → voice appears in library; a preview can be synthesized instantly ("test voice").
2. On `/studio` the user picks a voice (built-in preset codes or cloned voice) and types text.
3. Worker normalizes + tokenizes + chunks → AR loop → per-frame streaming decode → PCM chunk posted to the main thread → AudioWorklet plays while the rest is still generating (realtime first audio).
4. Final concatenated buffer is offered as WAV download + kept in generation history (metadata only; audio is re-synthesizable).

---

## 5. Project structure

```
mockbird/
├─ next.config.ts                  # headers() COOP/COEP (dev + Vercel), typedRoutes, no webpack cfg
├─ eslint.config.mjs               # flat config (next lint is gone in 16)
├─ AGENTS.md                       # next dev writes nextjs-agent-rules here; commit it
├─ package.json                    # "dev": "next dev --webpack", postinstall → copy-ort
├─ tsconfig.json                   # paths @/* → src/*
├─ scripts/copy-ort.mjs            # node_modules/onnxruntime-web/dist/*.{mjs,wasm} → public/ort/
├─ public/
│  ├─ ort/                         # ort.wasm.min.mjs, ort-wasm-simd-threaded.{mjs,wasm} (12.4 MB)
│  └─ tokenizer/tokenizer.model    # vendored (0.47 MB) for offline tokenization
├─ docs/adr/
│  ├─ 0001-bundler-parity.md       # dev=webpack / prod=Turbopack rules & escape hatch
│  └─ 0002-static-export-vs-vercel.md
└─ src/
   ├─ app/
   │  ├─ layout.tsx                # <html lang> + <EngineProvider> + sidebar shell
   │  ├─ page.tsx                  # redirect → /studio (or landing → /model if not ready)
   │  ├─ manifest.ts               # PWA manifest (Next 16 metadata convention)
   │  ├─ model/page.tsx            # first-run downloader, per-file progress, thread picker, warmup
   │  ├─ voices/page.tsx           # library: voice grid, record/upload, import/export
   │  ├─ studio/page.tsx           # text + voice + params + streaming player
   │  └─ settings/page.tsx         # threads, cache mgmt, storage usage, reset, engine info
   ├─ components/
   │  ├─ ui/                       # shadcn (button, card, dialog, slider, select, tabs,
   │  │                            #  progress, scroll-area, sonner, badge, textarea, …)
   │  ├─ engine/                   # EngineProvider (client), EngineStatusPill, CrossOriginGate
   │  ├─ model/                    # ModelDownloadPanel, FileProgressRow, WarmupPanel, StorageMeter
   │  ├─ voice/                    # VoiceCard, VoiceRecorder (MediaRecorder + level meter),
   │  │                            # VoiceUploader, CodecPreview, VoiceImportExport
   │  ├─ studio/                   # TextComposer, VoiceSelector, GenerationParams,
   │  │                            # StreamingPlayer, WaveformCanvas, GenerationHistory
   │  └─ layout/                   # AppShell, Sidebar, Header
   ├─ lib/
   │  ├─ tts/
   │  │  ├─ mossTtsRuntime.ts      # ⭐ ported inference core (prefill/decode/sample/stream)
   │  │  ├─ textNormalizer.ts      # JS robust normalizer (WeTextProcessing not available in browser)
   │  │  ├─ tokenizerClient.ts     # worker bridge: encodeText / countTokens / splitChunks
   │  │  ├─ sampling.ts            # temperature / top-k / top-p / repetition-penalty
   │  │  ├─ ort.ts                 # ORT bootstrap (wasmPaths=/ort/, numThreads, proxy=false)
   │  │  └─ audio.ts               # decodeAudioData→48k stereo, resampler, WAV writer
   │  ├─ model/
   │  │  ├─ modelManifest.ts       # file lists, HF URLs, expected sizes/etags
   │  │  ├─ opfsStore.ts           # OPFS read/write/persist/quota
   │  │  └─ downloader.ts          # ranged, resumable, progress, abort, 3× retry
   │  ├─ worker/
   │  │  ├─ protocol.ts            # typed RPC messages (main ⇄ worker)
   │  │  └─ client.ts              # main-thread promise-based proxy over postMessage
   │  └─ utils.ts, cn.ts
   ├─ state/
   │  ├─ modelAtoms.ts             # modelStatus, files, progress, threads, warmup, storage
   │  ├─ voiceAtoms.ts             # voices[], activeVoiceId, draft/recording state
   │  ├─ studioAtoms.ts            # text, synthParams (atomWithStorage), generation, history
   │  └─ derived.ts                # selectors: isReady, totalBytes, canGenerate, estDuration
   ├─ workers/
   │  ├─ tts.worker.ts             # entry: owns runtime + modelStore
   │  └─ tokenizer.worker.ts       # SentencePiece
   └─ audio/
      ├─ stream-player.worklet.ts  # ring buffer + Float32 channel playback
      └─ useStreamPlayer.ts        # AudioContext, worklet wiring, playback controls
```

### `package.json` scripts

```jsonc
{
  "scripts": {
    "dev": "next dev --webpack",        // RAM: Turbopack dev is too heavy for this project
    "build": "next build",              // Turbopack — matches Vercel
    "build:webpack": "next build --webpack", // parity check in CI
    "start": "next start",
    "lint": "eslint .",
    "typecheck": "next typegen && tsc --noEmit",
    "postinstall": "node scripts/copy-ort.mjs"
  }
}
```

---

## 6. Core modules to build (priority order)

### 6.1 `mossTtsRuntime.ts` — the port (highest risk, do first)
Port of `browser_onnx_runtime.js` (or, licensing-safe alternative, re-implement from the Apache-2.0 Python `ort_cpu_runtime.py` + the published ONNX graphs and their `*_meta.json` IO names). Responsibilities:
- `load()`: read both meta JSONs + manifest; create 6 ORT sessions with correct `externalData` sidecars; bind tokenizer worker.
- `encodeReferenceAudio(blob)` → `{codes, sampleRate, duration, numQuantizers}`.
- `synthesize({text, voice, params, onChunk, signal})` → chunked AR loop + streaming decode.
- `warmup()` — one tiny prefill/decode/decode_step pass so the first real utterance isn't slow.
- Sampling utilities + text chunking (`voice_clone_maxTextTokens`, default 75).
- Guardrails: max text length, frame cap, cancellation via `AbortSignal`.

**Licensing note (flag for decision):** MOSS-TTS-Nano is Apache-2.0, but MOSS-TTS-Nano-Reader ships **no LICENSE file**. Safest path: implement from the Apache-2.0 Python reference + published ONNX metadata (all IO names/shapes are in `*_meta.json`), attributing OpenMOSS. Faster path: vendor the Reader's JS with an attribution header — confirm licensing with upstream first.

### 6.2 `modelStore` + downloader
- Manifest: two repos, exact `requiredFiles` (verified above), `https://huggingface.co/<repo>/resolve/main/<file>`.
- Per file: check OPFS record `{size, etag}`; skip if present (instant "ready" on revisit); else ranged download with `Range: bytes=<have>-`, exponential-backoff retry, byte-progress events, per-file `AbortController`. 3–4 files in parallel.
- `navigator.storage.persist()` + `navigator.storage.estimate()` → show free quota; warn below ~1.5 GB.
- `NEXT_PUBLIC_MODEL_BASE_URL` env override to self-host weights on a cheaper/faster CDN.
- Pre-flight gates in `CrossOriginGate`: `crossOriginIsolated`, `SharedArrayBuffer`, OPFS, AudioWorklet, `deviceMemory` — thread slider capped by `hardwareConcurrency` (1 if not isolated).
- **Size reduction (Phase 5, optional):** re-export `moss_tts_global_shared.data` / `moss_tts_local_shared.data` as fp16 (→ ~335 MB) or int8-dynamic (→ ~170 MB) with an output-parity test against the fp32 graphs.

### 6.3 Voice cloning UX
- Record via `MediaRecorder` (input level meter, target 48 kHz, max 60 s) **or** upload any audio/video file.
- Guidance card: clean, single speaker, 8–15 s, no music/noise/reverb — most faithful with a clean sample.
- Validation: 2–30 s, decode success; quality hints for clipping / very low RMS.
- After encode: store `{id, name, createdAt, source, sampleRate, durationSec, numQuantizers, codes}` in OPFS; rename, re-encode, delete, **export `.mossvoice`** (JSON + base64 codes ≈ T×16×2 bytes → ~400 KB for 15 s) and **import**.
- 18 built-in voices are free: their `prompt_audio_codes` ship inside `browser_poc_manifest.json` (0.5 MB) → instant preview, and the user can skip the 45 MB codec-encode download entirely.

### 6.4 Synthesis + streaming playback
- Text: textarea with live char/token count, language hint (20 langs), normalization toggle, "splits at 75 tokens" indicator.
- Params (from `generation_defaults`): sample mode (`fixed`/greedy), audio temperature 0.8, top_p 0.95, top_k 25, repetition penalty 1.2, max frames 375 (~30 s), chunk token budget 75, seed. Advanced panel behind a disclosure.
- Streaming: worker posts `{chunkData: [Float32Array, Float32Array], sampleRate, isPause}` (transferables) → `useStreamPlayer` pushes into the AudioWorklet ring buffer; UI shows a live waveform, elapsed/RTF and generated-frame counter (the runtime already exposes a `profileState` we can surface).
- Controls: play/pause/stop, seek within the generated buffer, download **WAV (48 kHz stereo 16-bit)**, copy text, regenerate with a new seed, "continue" for long scripts.
- History: last N generations (params + text + voice) in `localStorage` via `atomWithStorage`; re-render on click.

### 6.5 Robustness / a11y
- Feature-detection banner (isolated / OPFS / AudioWorklet / threads) with explicit degradation messages, not silent failure.
- Actionable errors: network stall, quota exceeded, unsupported audio, OOM (suggest threads=1 / shorter text), missing COOP/COEP (the classic "why is it slow" case).
- Keyboard-accessible recorder, visible focus rings, `prefers-reduced-motion`, `aria-live` for generation status, dark/light theme.

---

## 7. Milestones

| Phase | Scope | Done when |
|---|---|---|
| **P0 · Scaffold + bundler parity** ⭐ | Next 16.3.6 + React 19.3 + TS, Tailwind v4, shadcn init, jotai v3 atoms, AppShell/sidebar/theme, `headers()` COOP/COEP, `copy-ort` script, ESLint flat config, `AGENTS.md` committed. **Verify both bundlers** emit the worker + worklet and that dev is `crossOriginIsolated` | `npm run dev` (webpack) and `npm run build` (Turbopack) both succeed; a test page spawns the module worker and loads the AudioWorklet; green banner in dev and on a Vercel preview |
| **P1 · Model Lab** ⭐ risk | `modelManifest`, `opfsStore`, resumable downloader, worker bootstrap, ORT sessions load, `warmup()`, progress UI + storage meter + thread picker | Fresh browser: "Download models" → 763 MB with progress → ready; reload is instant from OPFS; codec-encode can be skipped if the user won't clone |
| **P2 · Voice cloning** | Record/upload, 48 kHz stereo resampler, codec encode, voice store + library page, import/export, 18 built-ins, "test voice" preview | Record 10 s → voice card appears → preview speaks in that voice |
| **P3 · Studio** | Text composer + language + params, chunked synthesis, AudioWorklet streaming player, live waveform, WAV export, cancel/regenerate | 500-word zh/en script streams with no UI jank, first audio < ~1.5 s, downloadable WAV |
| **P4 · Polish** | History, PWA offline (app shell + voices with no network), i18n (zh/en), opt-in analytics (no audio/text), a11y pass, error hardening, README + deploy docs | Lighthouse a11y ≥ 95; offline reload synthesizes from cache |
| **P5 · Optional** | fp16/int8 weight re-export (~2× smaller download); "Cloud mode" toggle to a self-hosted `app_onnx.py` behind a `SynthesisEngine` interface; WebGPU EP experiment | Download < 400 MB with parity-verified output |

**Sequencing note:** P1 is the riskiest and gates everything; do a 1–2 day spike on `mossTtsRuntime.ts` + ORT session creation **before** any UI work, validating quality against `assets/audio/zh_1.wav` and the manifest's built-in voices (known-good upstream references).

---

## 8. Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **763 MB first-run download** | Adoption killer | OPFS persistent cache (one-time), HTTP-range resume, parallel downloads, `NEXT_PUBLIC_MODEL_BASE_URL` mirror, honest size+ETA messaging, optional fp16/int8 re-export (P5) |
| **dev/prod bundler split** (webpack dev vs Turbopack build) | "Works on my machine" bundling bugs in the worker/worklet | Bundler-agnostic `new Worker(new URL(...))` / `new URL(...)` patterns only; no magic comments (CI grep); dual builds in CI; documented escape hatch to `--webpack` on Vercel |
| **Turbopack dev RAM** | Dev machine unusable | `next dev --webpack` (done). If webpack dev also grows, cap with `NODE_OPTIONS=--max-old-space-size=4096` and split heavy pages via `dynamic()` |
| **Licensing of the browser runtime JS** | Legal | Implement from Apache-2.0 Python reference + published ONNX metadata; attribute OpenMOSS; confirm with upstream before shipping |
| **Chrome-only APIs** (OPFS in worker, AudioWorklet, `crossOriginIsolated` threads) | Firefox/Safari degraded | Feature-detect and degrade (threads=1, Cache Storage fallback, MediaRecorder differences); banner explaining best experience in Chromium |
| **Memory pressure** (~670 MB fp32 weights + KV cache; 2–4 GB total) | Tab crash on low-RAM devices | Thread slider, `deviceMemory` pre-flight, warn on long text, offer reload-after-download |
| **CPU-only speed** (RTF > 1 on weak machines) | Poor UX | Show RTF/ETA, cap default length, streaming means audio starts early, default threads = `min(4, hardwareConcurrency)` |
| **COEP breaks embeds/popups** | Third-party breakage | `same-origin-allow-popups` if needed; self-host fonts/icons; document the constraint |
| **WeTextProcessing unavailable in browser** | Numbers/symbols in zh read slightly worse | Ship the ported JS robust normalizer, expose the toggle, note the parity caveat in the UI |
| **Voice-clone misuse** | Ethical/legal | Consented-use checkbox + notice on record/upload; audio never leaves the device; terms-of-use page; nothing is uploaded or rate-limited |
| **HF CDN availability / rate limits** | Download failures | Resume + retry + mirror; ETag revalidation so repeat visits cost nothing |
| **jotai v3 is a fresh major** (released 2026-09-08) | API surprises | We use only stable primitives; if v3 churn bites, pin `jotai@2.20.3` (identical API for our surface) |

---

## 9. Alternatives considered (and why not)

- **Python backend on Vercel** — bundle-size and cold-start blockers; forfeits privacy/offline.
- **Always-on ONNX service (Fly/Railway) + thin web UI** — better latency, but cost + audio leaves device; kept as a future "Cloud mode".
- **Port to WebGPU/WebNN** — big speed win, but upstream only validates the `wasm` EP; revisit after v1.
- **Reimplement in-browser from the PyTorch runtime (transformers.js)** — not viable for a 0.1B custom multimodal AR model + 16-codebook audio tokenizer.
- **Other TTS models (Piper/Coqui/XTTS)** — MOSS-TTS-Nano wins on quality-per-parameter, 20 languages, realtime streaming, CPU-only, maintained ONNX export.
- **`output: 'export'` static hosting** — rejected for v1 because it disables `headers()`, which we need for cross-origin isolation. Documented as a config-only later option (ADR 0002).

---

## 10. Open questions

1. Vendor the Reader's browser runtime (fast, licensing question) or re-implement from the Apache-2.0 Python reference + ONNX metadata (safe, slower)? — recommend re-implement, spike first.
2. Ship the 763 MB fp32 bundle, or invest ~1 week in an fp16/int8 re-export with a parity test before public launch?
3. Include the 18 built-in voices (0.5 MB manifest, no codec-encode download) as a zero-clone on-ramp? — recommend yes.
4. Keep Vercel on Turbopack for the build, or standardize on webpack everywhere for guaranteed dev/prod parity at the cost of build speed? (Default: Turbopack on Vercel + CI parity build.)
5. Any server at all (analytics, voice sharing)? Default: no — 100% local; add only if asked.

---

## Appendix — verified facts & sources

**MOSS-TTS-Nano (all verified against the repos/HF today)**
- Specs, 20 languages, ONNX-CPU 2× claim: [README](https://github.com/OpenMOSS/MOSS-TTS-Nano#readme)
- Browser inference, ORT 1.24.3, 18 voices, OPFS store, SentencePiece sandbox: [MOSS-TTS-Nano-Reader/extension](https://github.com/OpenMOSS/MOSS-TTS-Nano-Reader)
- `codec_config`: `{sample_rate: 48000, channels: 2, downsample_rate: 3840, num_quantizers: 16}`
- `tts_config`: `{n_vq: 16, vocab_size: 16384, audio_codebook_sizes: [1024×16], audio_pad_token_id: 1024, …}`
- `generation_defaults`: `{max_new_frames: 375, do_sample: true, sample_mode: "fixed", text_temperature: 1.0, text_top_p: 1.0, text_top_k: 50, audio_temperature: 0.8, audio_top_p: 0.95, audio_top_k: 25, audio_repetition_penalty: 1.2}`
- Built-in voices (18): Junhao, Zhiming, Weiguo, Xiaoyu, Yuewen, Lingyu, Trump, Ava, Bella, Adam, Nathan, Soyo, Saki, Mortis, Umiri, Mei, Anon, Arisa
- HF CDN responses: `access-control-allow-origin: *`, `accept-ranges: bytes`, ETag present (verified with `curl -I`)

**Next.js 16 (verified against docs, version 16.3.6)**
- Turbopack default for dev+build; `next dev --webpack` / `next build --webpack` opt-outs: [/docs/app/guides/upgrading/version-16](https://nextjs.org/docs/app/guides/upgrading/version-16), [/docs/app/api-reference/cli/next](https://nextjs.org/docs/app/api-reference/cli/next)
- Static export does not support `headers()`: [/docs/app/guides/static-exports](https://nextjs.org/docs/app/guides/static-exports)
- React 19.2 features (`Activity`, `useEffectEvent`, View Transitions) + stable `reactCompiler` (+ experimental `turbopackRustReactCompiler`): version-16 guide
- `next lint` removed, `middleware`→`proxy`, async Request APIs only, runtime config removed: version-16 guide
- PWA/`app/manifest.ts` + Serwist, and `experimental.useOffline` (not recommended for production): [/docs/app/guides/progressive-web-apps](https://nextjs.org/docs/app/guides/progressive-web-apps)

**Package versions checked on npm:** `next@16.3.6` · `react@19.3.0` · `tailwindcss@4.3.3` · `shadcn@4.21.0` · `jotai@3.0.0` (v2.20.3 as fallback) · `onnxruntime-web@1.30.0` latest (start pinned at `1.24.3`, the version upstream validated)
