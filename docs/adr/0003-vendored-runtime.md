# ADR 0003 — Vendoring (and patching) the upstream browser runtime

**Status:** accepted, with one open question

## Context

MOSS-TTS-Nano ships ONNX weights and two Python runtimes. The reference *browser*
implementation lives in a separate repository, **MOSS-TTS-Nano-Reader**, which
packages OpenMOSS's extension build of the runtime:

| File | Lines | Role |
| --- | --- | --- |
| `extension/browser_onnx_runtime.js` | 2 380 | prefill/decode loop, frame sampling, streaming codec decode, text normalization, profiling |
| `extension/tokenizer_sandbox.js` | 5 960 | SentencePiece compiled to wasm-in-JS |
| `extension/browser_model_store.js` | 381 | OPFS cache + Hugging Face downloader |

Reimplementing the decode loop from scratch would be a large, risky effort: it
encodes tensor layout details (17-wide rows, 24 KV tensors, RoPE, per-channel
cached steps) that are only fully specified by the code and the `*_meta.json`
sidecars.

## Decision

Vendor the runtime, patch it, and make every patch auditable and reproducible.

- `src/lib/tts/vendor/mossTtsRuntime.js` and
  `src/workers/vendor/tokenizerSandbox.js` are copies of the upstream files.
- All local edits live in **`scripts/vendor-patches.json`** as exact
  `{find, replace}` pairs, each validated against upstream (exactly one match,
  or the sync fails).
- `pnpm sync:vendor` re-copies the files and re-applies every patch in one
  command, so the delta is never lost and always reviewable.
- A hand-written `.d.ts` (`mossTtsRuntime.d.ts`) types the surface Mockbird uses,
  so the port stays type-checked without rewriting 2 400 lines of JavaScript.
- `npm run copy:ort` / `copy:tokenizer` stage the runtime assets that must live
  outside the module graph (see ADR 0001).

## The eleven patches

| # | Area | Why |
| --- | --- | --- |
| 1–2 | ONNX Runtime import + `wasmPaths` | use the pinned npm package; serve the wasm from `/ort/` |
| 3–6 | Tokenizer sandbox | upstream hosts SentencePiece in a sandboxed `<iframe>`; the runtime runs in a Web Worker here, so it becomes a nested `Worker` served from `public/` |
| 7 | Frame identity | with a `Worker` there is no `.contentWindow`; keeping upstream's check spawned a fresh tokenizer Worker — losing the loaded model — on **every** request |
| 8–9 | Audio tokenizer encoder | cloning weights are optional here, so the encode session (and its warmup pass) is created lazily only when cached |
| 10 | Reference audio decoding | `AudioContext` does not exist in a Worker; the main thread decodes/resamples and transfers PCM |
| 11 | (see `vendor-patches.json`) | remaining fix required to run in a Worker |

## Consequences

- Upstream improvements are picked up by re-running `pnpm sync:vendor` and
  resolving patch conflicts — a deliberate, visible cost.
- Licensing is the open question: MOSS-TTS-Nano is Apache-2.0, but the Reader
  repository ships **no LICENSE file**. The vendored files stay clearly marked in
  `vendor/` with attribution until upstream confirms terms. A re-implementation
  from the Apache-2.0 Python reference (`ort_cpu_runtime.py` + the published
  `*_meta.json`) remains the safe long-term path.
- Anything the runtime cannot do in a Worker (audio decoding) is done on the main
  thread and handed over as plain PCM — a clean boundary that also keeps the
  worker API surface small.
