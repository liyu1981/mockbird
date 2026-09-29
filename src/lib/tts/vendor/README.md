# Vendored upstream runtime

These files are **not** authored by Mockbird. They are adapted copies of the
MOSS-TTS-Nano browser runtime, re-synced with:

```bash
git clone --depth 1 https://github.com/OpenMOSS/MOSS-TTS-Nano-Reader /tmp/reader
pnpm sync:vendor
```

| File | Upstream | Upstream license |
| --- | --- | --- |
| `mossTtsRuntime.js` | `MOSS-TTS-Nano-Reader/extension/browser_onnx_runtime.js` | Reader repo ships **no LICENSE file** (MOSS-TTS-Nano itself is Apache-2.0) |
| `../../workers/vendor/tokenizerSandbox.js` | `MOSS-TTS-Nano-Reader/extension/tokenizer_sandbox.js` | same |

`mossTtsRuntime.d.ts` is ours: it types the surface Mockbird uses so the port
stays type-checked without rewriting 2 400 lines of JavaScript.

## Every local edit

All edits are stored in **`scripts/vendor-patches.json`** as exact
`{find, replace}` pairs, each validated against the upstream file (exactly one
match required, otherwise the sync fails loudly). `pnpm sync:vendor` re-copies
the sources and re-applies all of them, so the delta is always reproducible and
reviewable:

```bash
pnpm sync:vendor
```

| # | Patch id | Area | Why Mockbird needs it |
| --- | --- | --- | --- |
| 1 | `ort-import` | dependencies | Use the pinned `onnxruntime-web` package instead of a vendored ORT copy; only the `.wasm` is served, from `/ort/`. |
| 2 | `ort-wasm-paths` | dependencies | Self-hosted wasm path instead of the extension's `chrome.runtime.getURL`. |
| 3 | `tokenizer-guard` | worker context | The runtime runs in a Web Worker, where `document`/`chrome` do not exist. |
| 4 | `tokenizer-iframe-to-worker` | worker context | Host the SentencePiece bundle in a nested `Worker` instead of a hidden `<iframe>`. |
| 5 | `tokenizer-drop-window-listener` | worker context | The response handler moved into `ensureTokenizerSandboxFrame`. |
| 6 | `tokenizer-post-to-worker` | worker context | `postMessage` goes to the nested `Worker`. |
| 7 | `tokenizer-frame-is-a-worker` | worker context | A `Worker` has no `.contentWindow`; keeping upstream's check spawned a **fresh tokenizer Worker (losing the loaded model) on every request**. |
| 8 | `lazy-codec-encode-session` | optional weights | The 45 MB audio-tokenizer *encoder* is only needed for cloning, so its ORT session is created lazily. |
| 9 | `codec-encode-helper` | optional weights | Companion `maybeCodecEncodeSessionSpec()` method. |
| 10 | `warmup-skip-missing-encoder` | optional weights | Warmup must tolerate the encoder session being absent. |
| 11 | `encode-from-waveform` | worker context | `AudioContext` is unavailable in Workers: accept pre-decoded 48 kHz PCM transferred from the main thread. |

Everything else (prefill/decode loop, local decoder, sampling, streaming codec
decode, text normalization, voice presets, profiling hooks) is upstream code,
unmodified.

## The tokenizer bundle

`tokenizerSandbox.js` is copied to `public/tokenizer.worker.js` at install/build
time by `scripts/copy-tokenizer-worker.mjs`, with two edits:

| # | Upstream | Mockbird | Why |
| --- | --- | --- | --- |
| 1 | `window.addEventListener("message", …)` | `self.addEventListener("message", …)` | Runs in a Worker, not a document. |
| 2 | `event.source?.postMessage({…}, "*")` | `self.postMessage({…})` | Same. |

The SentencePiece wasm is embedded in this bundle as base64, so the worker only
needs the 0.47 MB `tokenizer.model`, which comes from the OPFS model cache.

## Licensing follow-up

Because the Reader repository has no license file, a re-implementation from the
Apache-2.0 sources (`ort_cpu_runtime.py` in MOSS-TTS-Nano plus the published
`*_meta.json` IO metadata) is tracked as an open question in `PLAN.md` (§10.1).
Until that is resolved, Mockbird keeps these files in a clearly marked `vendor/`
directory with attribution and does not relicense them.
