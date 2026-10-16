# `mossTtsRuntime.js` — architecture analysis & re-implementation plan (no ONNX Runtime)

**Status:** analysis only, no code changed. Written for the "replace the ONNX Runtime
dependency" discussion (PLAN.md §10.1 open question, ADR 0003).

## 0. Method & evidence

Everything below is derived from sources in this repo plus two meta files and five
`.onnx` graphs fetched from Hugging Face for verification:

| Source | Used for |
| --- | --- |
| `src/lib/tts/vendor/mossTtsRuntime.js` (2 376 lines) | the runtime itself |
| `src/lib/tts/vendor/README.md`, `scripts/vendor-patches.json`, `docs/adr/0003` | local delta vs upstream |
| `upstream/MOSS-TTS-Nano/ort_cpu_runtime.py` (Apache-2.0) | the reference CPU runtime this file is a port of |
| `upstream/MOSS-TTS-Nano/onnx/export_moss_tts_browser_onnx.py` (Apache-2.0) | **exact** graph semantics: layer-by-layer forward, tensor layouts, IO names |
| HF `tts_browser_onnx_meta.json`, `codec_browser_onnx_meta.json` | shapes, layer counts, IO names, streaming-cache specs |
| HF `moss_tts_decode_step.onnx`, `moss_tts_local_fixed_sampled_frame.onnx`, `moss_audio_tokenizer_{encode,decode_full,decode_step}.onnx` | read with a throwaway pure-Python protobuf reader (`/tmp`, not committed) to get the op histogram and the full initializer inventory (name, shape, byte offset into the `.data` sidecar) |

No dev server, no browser, no ORT session was executed. Line numbers refer to the
current vendored file.

---

## 1. What the file actually is

One ES module, ~2 400 lines, that is a **concatenation of four upstream sources**
(the `// src/…` banners are still in the file):

| Lines | Block | ORT-dependent? |
| --- | --- | --- |
| 1–3 | `import * as ort from "onnxruntime-web/wasm"` | yes |
| 4–189 | `tts_robust_normalizer.js` — text normalisation (CJK/Latin spacing, markdown stripping, URL/email protection, terminal punctuation) | **no** |
| 190–1018 | runtime utilities: profiling, path resolution, asset reads, tensor flattening, sampling, tokenizer sandbox, prompt-row builders, text chunking, stream-pacing helpers, `CodecStreamingDecodeSession` | mostly no (ORT only for `ort.Tensor` + `session.run`) |
| 1019–2369 | `BrowserOnnxTtsRuntime` class — session lifecycle, prefill/decode loop, codec decode, `synthesizeVoiceClone` | yes |
| 2370–2376 | factory + `export` | — |

The class is the whole public surface Mockbird uses (11 methods, typed by
`mossTtsRuntime.d.ts`).

**The single most important structural fact for the re-implementation question:**
the file cleanly separates into a *policy* layer and a *compute* layer.

* **Policy (no ORT, keep verbatim):** text normalisation (186 lines), sentence/clause
  chunking + token-budget splitting, prompt row construction, sampling semantics
  (argmax / repetition penalty / top-k / top-p / fixed-random-u), the
  stream-pacing heuristic, inter-chunk pause rules, profiling keys, cancellation
  and event-loop yielding. ≈ 700 lines, all pure JS, all independently testable.
* **Compute (ORT, replace):** `new ort.Tensor(...)`, `session.run(feeds)`,
  graph construction, session loading, KV-cache plumbing, output slicing. ≈ 900 lines
  interleaved with the policy layer inside the same methods.

A re-implementation is therefore *not* "rewrite 2 400 lines" — it is "keep 700,
replace 900 with kernels, and re-glue".

---

## 2. Model architecture (from the meta files + export script)

### 2.1 Global transformer (the "LLM")

* 12 layers, `d_model = 768`, 12 heads, head_dim 64, vocab 16 384.
* GPT-2 blocks (`ln_1 → attn.c_attn(2304) → RoPE → causal MHA → c_proj(768) → +residual`,
  `ln_2 → mlp.fc_in(3072) → GELU → fc_out(768) → +residual`), final `ln_f`.
* Learned `wte` (16 384×768) + `wpe`? — the export script adds `transformer.wpe`
  only when `position_embedding_type == "absolute"`; attention uses **RoPE**, so
  the position embedding is *not* in the parameter list (confirmed: the decode
  graph's initializers contain `wte`, 16 × `audio_embeddings.k`, 12 × layer
  weights, `ln_f` — no `wpe`).
* Input embedding = `wte[text_id] + Σ_k audio_embeddings[k][code_k]`, where
  `audio_pad_token_id` channels contribute 0 (`_build_inputs_embeds`).
* Row layout: 17 int32 = 1 text token + 16 audio codes (`row_width = n_vq + 1`).
* Reserved ids: `pad=3, im_start=4, im_end=5, audio_start=6, audio_end=7,
  user_slot=8, assistant_slot=9, audio_pad=1024`; codebooks are 16 × 1024.

### 2.2 Local transformer (the "frame head")

* **1 layer**, `d_model = 768`, 12 heads, head_dim 64 (≈ 2.4 MFLOP-level per step,
  but it is re-run 17× per audio frame in the cached-step mode).
* Same GPT-2 block shape; RoPE; `ln_f`.
* Three heads: `text_lm_head` (768×16 384), and 16 × `audio_lm_heads[k]` (768×1024).
* Two exported variants:
  * `moss_tts_local_decoder.onnx` — stateless. Input `[global_hidden, text_token_id,
    audio_prefix_token_ids(15)]` → builds an 17-token sequence
    `[global, wte(text), emb_0(prefix0) … emb_14(prefix14)]`, runs one full
    non-causal pass, reads logits at position 0 (text) and k+1 (channel k).
  * `moss_tts_local_cached_step.onnx` — incremental, with its own KV cache
    (`local_past_key_i/value_i`). Input selects the embedding by
    `step_type` (0 = global_hidden, 1 = text token, 2 = audio token of
    `channel_index`), i.e. one channel per session call.

### 2.3 Audio tokenizer (codec)

* 48 kHz stereo, 16 RVQ codebooks × 1024 entries, 12.5 Hz frame rate
  (`downsample_rate = 3840`).
* Encoder graph (`encode.onnx`, 44.5 MB sidecar): 32 Convs, 24 LayerNorms,
  `ArgMax` ×16 with `Pow/ReduceSum/Clip/ReduceL2` (the RVQ nearest-codebook
  search), `Pad`, i.e. a strided-conv encoder + RVQ quantiser.
* Decoder graph (`decode_full.onnx` and the streaming twin `decode_step.onnx`,
  shared 44.2 MB sidecar) — identical op histogram except the masking ops
  (`Less/And/Where` vs cached-KV gather), because the streaming variant keeps
  attention state in explicit IO:
  * 16 × `Conv(512, 8, 1)` codebook projections + `output_proj Conv(768, 512, 1)`
    (1×1 convs only — no `ConvTranspose` anywhere).
  * 4 hierarchical transformer stages `model.decoder.{1,3,5,7}.transformer.layers.*`
    with **4 / 2 / 2 / 4** layers, `d_model = 256`, SwiGLU FFN
    (`ffn.0` 256→1024, `Erf·Div` gate, `ffn.2` 1024→256), `layer_scale_1/2`
    (LayerScale), RoPE attention.
  * 12 attention caches grouped 4/4/4 with contexts **500 / 1696 / 2480 / 3248**
    positions, `cache_shape [1,4,ctx,64]`, plus `cached_positions [1,ctx]`
    (int32) and an int32 `attn_offset` per layer — a **ring buffer with explicit
    absolute positions**, so RoPE stays correct across wrap-around.
  * 4 int32 `transformer_offset_*` counters (one per stage).
  * Final head `decoder.8` → waveform: the last projection is `MatMul[256, 240]`,
    then reshape/Gather/Mul/Concat overlap-add, `Div 2.0`, `Sub 1.0`, `Where`
    (a tanh-style final activation) and `Reshape → [1, 2, T]`.
  * `If` node: if the requested code length is 0 the graph returns an empty audio
    tensor (the runtime's `audio_lengths == 0` check depends on this).
* Op set overall: `MatMul, Conv(1×1), LayerNormalization, Softmax, Gather,
  Reshape/Transpose/Concat, elementwise, Cast`. **No convolution > 1×1, no
  transposed convolution, no attention beyond standard MHA.** This matters a lot
  for §5.

---

## 3. Logic flow

### 3.1 Session / asset lifecycle

```
configure({modelPath, threadCount})
  → key = "<root>::<threads>"; if unchanged and prepared → no-op
  → drops manifest/meta/sessions/tokenizer, resetRuntimeCaches()

ensureManifestLoaded()          browser_poc_manifest.json (3 candidate paths) + tts_meta + codec_meta
ensureOrtConfigured()           ort.env.wasm.{numThreads, wasmPaths="/ort/", simd=true, proxy=false}
ensureSynthesisLoaded()         creates, in order:
                                 prefill, decode, codecDecode(full),
                                 [codecEncode if cached], localGreedyFrame,
                                 localFixedSampledFrame,
                                 localCachedStep | localDecoder, codecDecodeStep
warmup()                        one prefill + one local step + one frame + one full codec decode
                                (+ one streaming codec step, + encoder warmup) to JIT-warm
                                ORT's wasm and to page in weights
```

Thread count (1334): `crossOriginIsolated ? min(threadCount, hardwareConcurrency) : 1`.
Because the app sets COOP/COEP, threads are available; without them ORT silently
drops to 1.

Model bytes are read through `assetReader` (OPFS in Mockbird) or `fetch` of
`file://`/`http(s)://` (`readBufferFromLocalPath`). External weight sidecars
(`*.data`) are passed to ORT as `sessionOptions.externalData = [{path, data}]`
with a same-basename fallback — this is how 440 MB / 229 MB / 44 MB blobs get
attached without a filesystem (1371–1403).

### 3.2 Cloning path

```
main thread: File → decodeAudioData + OfflineAudioContext resample to 48 kHz stereo
            → transfer Float32Array (channel-major) to worker          (patch 11)
worker:      encodeReferenceAudioFromWaveform()
            → codecEncode.run({waveform [1,2,L], input_lengths})
            → audio_codes [1, L/3840, 16] → stored in OPFS as uint16
```

`ensureCodecEncodeLoaded()` is **lazy** (patch 8/9): if `assetReader.exists()` says
the 45 MB encoder is not cached, no session is created and `warmup` skips it
(patch 10). That is why cloning is optional in the download UI.

### 3.3 Synthesis path

```
synthesizeVoiceClone({text, promptAudioCodes, doSample, sampleMode, streaming, …})
  prepareSynthesisText()             robust normaliser (+ a warning: no WeTextProcessing in JS)
  splitVoiceCloneText(text, 75)      sentence → clause → binary-search token budget → repack
  for each chunk:
     synthesizeSingleChunk()
       buildVoiceCloneRequestRows()  ← the prompt layout, below
       generateAudioFrames()         prefill + 12.5 Hz loop
       codec decode                  streaming (default) or decode_full + fallback
     insert inter-chunk pause       0.24 s (≤4 words) or 0.40 s otherwise
```

**Prompt row layout** (766) — the whole conditioning scheme in six lines:

```
[ user_prompt_prefix… , audio_start ,          ← text rows (col0 = token id)
  (audio_user_slot, code0…code15) × T,          ← audio prefix rows, col0 = 8
  audio_end , user_prompt_after_reference… ,
  <text tokens…> , assistant_prompt_prefix… , audio_start ]   ← suffix text rows
```
All rows padded to width 17 with `audio_pad_token_id`; `attention_mask` is all
ones, and `past_valid_length` is initialised to the row count.

### 3.4 The frame loop (1965) — three interchangeable local-frame strategies

All three consume the same `global_hidden` (last position of `prefill` or
`decode_step`) and produce one 16-token frame + a stop flag:

| Strategy | Session | Calls / frame | Sampling |
| --- | --- | --- | --- |
| `greedy` | `local_greedy_frame` | **1** | argmax with an in-graph `repetition_seen_mask` [1,16,1024] |
| `fixed` (default) | `local_fixed_sampled_frame` | **1** | in-graph TopK(25) → top-p 0.95 → inverse-CDF against a `u` vector drawn in JS (`audio_random_u[16]`, `assistant_random_u`) |
| `full` | `local_cached_step` | **17** (1 text + 16 channels) | JS-side `sampleFromScores` (temperature/top-k/top-p/repetition penalty) |
| fallback | `local_decoder` | 17 | same, but stateless (only loaded when `local_cached_step` is absent) |

Per frame the loop then feeds the frame back into the global transformer as a
single row `[assistant_slot, code0…code15]`, runs `decode_step` with
`past_valid_lengths`, and takes the last hidden state as the next frame's
conditioning. `max_new_frames` (default 375 ≈ 30 s) bounds it; the local head's
stop flag ends it early.

Cooperative scheduling: `await yieldToBrowserEventLoop()` at the top of every
frame, after `decode`, and every 4 channels inside the `full` path
(`CONTROL_EVENT_LOOP_YIELD_CHANNEL_INTERVAL = 4`); `isCancelled()` is polled at
the same points.

### 3.5 Codec streaming + pacing (912, 2142)

`CodecStreamingDecodeSession` is a pure bookkeeping wrapper around
`codec_decode_step`: it owns the 4 offset tensors and 12×(offset, keys, values,
positions) state tensors, feeds them back on the next call, and returns
`{audio, audio_lengths}`. `reset()` reallocates zeroed (positions filled with −1).

Pacing: the runtime tracks **lead time** = `emittedSamples/sampleRate − wallclock
since first audio`, and only flushes frames when the buffer is at risk:

| lead | frames per codec call |
| --- | --- |
| < 0.2 s | 1 |
| < 0.55 s | 2 |
| < 1.1 s | 4 |
| ≥ 1.1 s | up to 8 |

plus a hard rule "decode if `leadSeconds < 0`" and "decode if lead < 0.45 s".
This keeps ~0.2–1 s of audio buffered so the AudioWorklet never starves while
still not decoding further ahead than the renderer can absorb.

Non-streaming mode runs `codec_decode_full` on the whole frame list, and on
`bad_alloc` falls back to `decodeFullAudioIncremental` (8-frame batches through
the streaming session) — an explicit OOM escape hatch for long generations.

### 3.6 Sampling code (456–642)

`sampleFromScores` is textbook: temperature → top-k by *threshold on the sorted
array* (ties handled by `< threshold`) → top-p with the standard "keep the token
that crosses p" shift → softmax in `Float64Array` → single-pass inverse-CDF draw
(`Math.random()`, falling back to argmax if the CDF under-runs).
`applyRepetitionPenalty` is the HuggingFace convention (× penalty if logit < 0,
÷ otherwise) over the *set* of previously emitted codes per channel.

Note the two sampling implementations are **not** numerically equivalent: the JS
top-p keeps the token that crosses the threshold, the graph's
`sample_from_topk_topp_with_random_u` keeps `prev_cumsum < p` on the *already
top-k-truncated* distribution. Different `sample_mode` therefore means slightly
different prosody, not just a different RNG.

---

## 4. How it uses ONNX Runtime (the part we would replace)

**Sessions** (6–8 `InferenceSession`s, 763 MB of weights, `graphOptimizationLevel: "all"`,
`executionProviders: ["wasm"]` only). `prefill` and `decode` share one 440 MB sidecar,
the three local graphs share one 229 MB sidecar, the two codec decoders share 44 MB.

**Tensor discipline.** ORT-web requires real typed arrays, so the runtime keeps a
"reusable feed" object per session and mutates the underlying `Int32Array`
in place (1091–1184): `reusableDecodeRowData` (17), `reusableLocalCached*` (1 each),
`reusableLocalGreedySeenMaskData` (16×1024, refilled per frame),
`reusableLocalFixedSampled*RandomData` (1 + 16), `reusableLocalDecoderPrefixData` (15).
This avoids per-step allocation in JS, but **not** in the wasm heap.

**KV caches are passed by value, not by reference.** `decode_step` returns
`present_key_i/present_value_i` as *new* tensors; the loop copies them into
`reusableDecodeFeeds` for the next call. Every step therefore
(a) concatenates past+current inside the graph, and
(b) allocates 24 fresh float buffers sized `[1, N, 12, 64]`.

At 12.5 steps/s this is the dominant cost of a long generation:

```
KV bytes per sequence position = 12 layers × 2 × 12 heads × 64 × 4 B = 73.7 kB
per step ≈ 2 × N × 73.7 kB (concat + copy)  (+2× the local 1-layer cache, tiny)
N = 600 (≈ 20 s prompt + 30 s generated) → ≈ 88 MB moved **per frame**
                                          → ≈ 1.1 GB/s of pure memcpy
```

This is the concrete, quantified argument for a hand-written engine: a
pre-allocated ring buffer with an offset removes it entirely, turning per-step
global cost from O(N) into O(1) + O(1 matmul).

**Other ORT touchpoints:** `extractLastHiddenTensor` (440) wraps the last slice of
`[1, S, 768]` in a new `ort.Tensor` over `subarray` (free); `sliceChannelMajorAudio`
(413) + `mergeFloat32Chunks` (425) unpack `[1, C, T]` into per-channel
`Float32Array`s for the worker→main transfer; `ort.env.wasm.numThreads` is set
from `crossOriginIsolated && hardwareConcurrency`; `ort.env.versions.web` is
reported to the UI as the engine version.

**Not ORT:** SentencePiece runs in a nested `Worker` (`public/tokenizer.worker.js`,
patches 4–7) and is driven by request/response promises (`postTokenizerSandboxRequest`).
The tokenizer model is shipped as base64 over `postMessage` — a 0.47 MB copy per
`ensureTokenizerLoaded`, not per request.

---

## 5. Findings / sharp edges (all verified in the current file)

1. **`nextTextToken` is not defined** (line 2106). The `localDecoder` fallback
   branch of `generateAudioFrames` calls `runLocalDecoder(global_hidden, nextTextToken, frame)`
   with an undeclared identifier, and it also skips the `step_type=0` text-token
   step that the Python reference performs (`ort_cpu_runtime.py:779-789`).
   The branch is dead for the shipped model (which ships `local_cached_step`),
   so it has never thrown — but it is a real `ReferenceError` waiting for any
   model variant that lacks the cached-step graph. It is upstream code, so it
   belongs in `vendor-patches.json` if we fix it.
2. **Profile snapshots are typed as `Record<string, number>` but are objects.**
   `cloneProfileSnapshot` emits `{totalMs, count, minMs, maxMs, averageMs}` per
   key; `mossTtsRuntime.d.ts` and `src/lib/worker/protocol.ts` say `number`, and
   the worker papers over it with `as never`. `src/app/settings/page.tsx:352-364`
   then calls `ms.toFixed(1)` on an object → **render-time TypeError** whenever
   the profiler is enabled. The "Runtime profiling log" has therefore never
   displayed. One-line fix in the UI + one in the two type files.
3. **Missing profile timing on the non-streaming path** (2142): the early
   `return {frames, chunks}` skips `finishProfileTiming("pipeline.synthesize_single_chunk")`,
   so non-streaming generations under-report. Streaming records it.
4. **Two sampling implementations** (JS vs in-graph) as described in §3.6 — worth
   pinning down which one the "expected" prosody baseline came from before any
   re-implementation, because it changes the output distribution.
5. **`decodeFullAudioIncremental` ignores `audio_lengths` continuity** and simply
   concatenates per-call chunks; correct for this codec, but it is an implicit
   assumption, not an invariant check.
6. **`ort.env.wasm.numThreads` is set in `ensureOrtConfigured`, i.e. after the ORT
   module is imported.** Fine today (sessions are created later), but it is an
   ordering dependency: any code path that creates a session before
   `ensureSynthesisLoaded` would silently run single-threaded.
7. `reusableLocalGreedySeenMaskData.fill(0)` clears 16 k ints per frame — 200 kB/s
   of writes at 12.5 fps, negligible, but it shows the pattern: any state that
   could be maintained incrementally is instead rebuilt.
8. `splitTextByTokenBudget` binary-searches the *character* prefix with a
   `countTextTokens` call per probe — O(log n) tokenizer round-trips per chunk,
   each a `postMessage` to the nested worker. For a 10 000-character paste this
   is the slowest part of `analyzeText` before synthesis even starts.

---

## 6. Re-implementation without ONNX Runtime

### 6.1 What "no ORT" buys, honestly

| | Today (ORT-web 1.24.3, wasm, SIMD+threads) | Hand-written engine |
| --- | --- | --- |
| Dependency | 12.4 MB `ort-wasm-simd-threaded.wasm` + ~1.5 MB JS glue, `copy-ort` step, COOP/COEP headers required for threads | none (or a ~30–80 KB hand-written `.wasm` for the hot GEMMs) |
| Licensing | Reader repo has **no LICENSE** → the vendored runtime cannot be relicensed (§10.1) | only Apache-2.0 sources (`ort_cpu_runtime.py`, export script, published `*_meta.json`) → problem closed |
| Per-step cost | O(N) KV copy ≈ 1.1 GB/s at N=600 | O(1) with a ring buffer |
| Numerics | reference | must be *proven* equal, not assumed (see 6.5) |
| Effort | — | ~2–4 engineer-weeks for TTS + streaming codec; +2–3 weeks for the encoder (cloning) |
| Failure modes | well-trodden | new: every kernel is ours |

### 6.2 Reading the weights without ORT

The `.onnx` files are protobuf. A ~200-line reader (I wrote one during this
analysis, in Python, in ~30 min) yields every initializer's **name, dims, dtype
and byte range inside the `.data` sidecar**. Verified inventory:

* `moss_tts_decode_step.onnx` (291 kB) + `moss_tts_global_shared.data` (440.8 MB):
  `wte [16384,768]`, 16 × `audio_embeddings.k [1024,768]`, and per layer
  `c_attn [768,2304]`, `c_proj [768,768]`, `fc_in [768,3072]`, `fc_out [3072,768]`,
  `ln_1/ln_2/ln_f [768]`. All 440.8 MB is referenced; no dead tensors.
* `moss_tts_local_fixed_sampled_frame.onnx` (471 kB) + `moss_tts_local_shared.data` (229.7 MB):
  `wte` again, 16 × audio embeddings, one local layer, `text_lm_head [768,16384]`,
  16 × `audio_lm_heads.k [768,1024]`.
* Codec: 178 tensors / 44.2 MB, all referenced.

Because Mockbird already holds these bytes in OPFS, the loader is
`new Float32Array(opfsBuffer, offset, count)` — **zero copy, zero decode**, and
`alignFloat32` only if an offset is not 4-byte aligned (they are). No need to keep
ORT's external-data machinery at all.

### 6.3 Kernel inventory (the actual work)

Everything the graphs need, and nothing else:

| Kernel | Where | Notes |
| --- | --- | --- |
| `gemm_f32` (M×K×N) | global attn/MLP, all lm heads, codec transformer | the only thing worth writing in wasm/SIMD; shapes are tiny at decode (M=1..17) → **GEMV-bound, not GEMM-bound** |
| `layer_norm` (ε from graph) | 12 global + 12 codec layers | ε = 1e-5 (`0x3727c5ac`) |
| `rope` (interleaved even/odd, `rotate_half` = `cat(-x[1::2], x[::2])`) | every attention | 12 global + 12 codec |
| `softmax` (row-wise, f32) | attention | |
| `gelu_tanh` | MLPs (both models) | graph shows `Tanh`+`Pow 3`+`0.044715` |
| `swiglu` (`Erf·Div` gating) | codec FFN | |
| `embedding_gather` + masked add | input embeds | |
| `concat_rope_kv` / ring write | KV cache | pre-allocated, no alloc |
| `topk_topp_inverse_cdf` | fixed sampling | already in JS, feed `u` in |
| `repetition_penalty_mask` | greedy/fixed | already in JS |
| `argmax` (greedy), `top-k` (full) | sampling | already in JS |
| `nearest_codebook` ×16 with residual | codec **encode** (cloning) | `‖r−c‖²` over 1024 codes, 8-dim → trivial |
| overlap-add / windowing / final activation | codec decode head | must be reverse-engineered from `decoder.8` (the one piece I could not fully confirm statically) |

No `ConvTranspose`, no strided conv in the decode path, no fused QKV, no
paged attention, no quantization ops. This is a *small* kernel set — that is the
single biggest argument that the re-implementation is tractable.

### 6.4 Performance strategy (in order of expected payoff)

1. **Pre-allocated KV ring buffers** for the global decoder (and the codec's
   12 caches with their `attn_offset`/`cached_positions` semantics preserved
   verbatim). Removes the O(N²) behaviour. Expect the single largest win.
2. **Keep weights f32, keep the download as-is** for the first milestone; a
   f16 re-export (PLAN.md §10.2, 763 MB → ~390 MB) is a separate, independent
   win and can be layered on later since the loader is index-based.
3. Measure plain `Float32Array` JS kernels first: at M=1 the arithmetic intensity
   is ~1 FLOP/byte for `c_attn`, i.e. memory-bandwidth-bound, and the weights are
   already in cache-line-friendly row-major order. A hand-written wasm GEMV is
   worth it only if profiling says so.
4. Threads: either (a) shard layers across N workers (prefill is embarrassingly
   parallel, decode needs a per-step barrier), or (b) one worker + a small
   pthread wasm. Layer sharding also gives a natural place to keep COOP/COEP-free
   operation.
5. Optional, **not ORT**: a WGSL compute path for prefill (large M) where a GPU
   actually pays off, while decode (M=1) stays on the CPU kernels. This is the only
   sensible shape for "GPU" here — a 100 M AR decoder at 12.5 steps/s is
   latency-bound, not throughput-bound. **This is a question for the review (§7).**

### 6.5 The real risk: discrete outputs

Floating-point differences do not stay small. A `±1e-6` logit difference is
harmless until two candidates are within 1e-6, at which point a frame token
flips, the KV cache diverges, and the rest of the utterance is different audio
(not "slightly different audio"). Therefore:

* **Milestone 0 — parity harness before any kernel work.** For a fixed input
  (one builtin voice, one 75-token chunk, greedy mode), record from the current
  ORT path: per-frame token ids, per-step `global_hidden` last-vector checksum,
  text-stop step, and the decoded PCM hash. The new engine must reproduce the
  **token sequence exactly** under greedy mode before it is allowed to serve
  users; PCM equality is checked to a tolerance, token equality exactly.
  (`tests/fixtures/reference-zh.wav` exists but is an audio fixture, not a
  token-level golden file — we need to add one.)
* Ship the two engines side by side behind a flag, run both on the same input,
  and only delete ORT once token parity holds over a corpus (all 18 builtin
  voices × N texts × greedy) and "close enough" holds over the rest.
* Keep the encoder (cloning) on ORT in phase 1 if needed: it is a one-shot
  offline path, its graph is the most intricate (strided convs, RVQ search), and
  it has no latency constraint. "No ORT for synthesis, ORT for cloning" is a
  legitimate milestone-1 outcome; "no ORT at all" is milestone 3.

### 6.6 Suggested phasing

| Phase | Scope | Exit criterion |
| --- | --- | --- |
| 0 | Parity harness + golden token fixtures (no new engine code) | reproducible token/PCM baselines from today's build |
| 1 | Protobuf weight reader + global prefill/decode kernels (greedy, no sampling) | token-exact parity on the harness; prefill TTFA ≤ today |
| 2 | Local frame head in all three sample modes + streaming codec decoder (ring caches) | full-utterance parity, streaming TTFA ≤ today, no O(N) KV growth |
| 3 | Drop `onnxruntime-web`, `copy:ort`, COOP/COEP (if unused elsewhere), `vendor-patches.json` sync for the runtime | build without ORT; `vendor/` reduced to the tokenizer worker or removed |
| 4 | Optional: codec encoder (cloning) kernels; f16 weight re-export; WGSL prefill | — |

Phase 1+2 is where the licence question (§10.1) is effectively resolved, because
the only remaining ORT code would be the encoder.

### 6.7 What stays untouched either way

`src/lib/tts/vendor/README.md` provenance rules, `scripts/vendor-patches.json`,
the OPFS store, the downloader, the worker protocol, the AudioWorklet player,
the WAV export, the normaliser. The re-implementation should be a **new module**
(`src/lib/tts/engine/…`) behind the same `synthesizeVoiceClone` contract, with
the vendored runtime kept as the reference implementation (and the parity
oracle) until phase 3.

---

## 7. Follow-up: feasibility of a **WGSL-only** re-implementation (no ORT)

Review questions 1 and 2, taken together: *drop ORT entirely and put the whole
engine in WGSL/WebGPU*. Answer: **yes, with a bandwidth caveat, and the caveat is
quantified below.** This section is the argument; the numbers come from the real
weight shapes extracted in §2 and read out of the graphs.

### 7.1 The engine is bandwidth-bound, not FLOP-bound

| Quantity per audio frame (12.5 frames/s) | Value |
| --- | --- |
| Global decode step (12 layers, `d=768`): weights read once | 340 MB |
| Global KV cache read at `N=600` positions (12×2×12×64 fp32) | 43 MB |
| Local head as currently graphed (`local_fixed_sampled_frame`, 17 unrolled local steps) | 582 MB |
| — of which the 28.3 MB local layer is re-read **17×** | 481 MB |
| — of which `text_lm_head [768,16384]` is read for **2** logits | 50 MB |
| Codec streaming decode (44 MB weights + up to 65 MB ring caches) | 109 MB |
| **Total, today's ORT path** | **≈ 1 075 MB/frame → 13.4 GB/s sustained** |
| **Total, fused kernels + f16 weights** | **≈ 286 MB/frame → 3.6 GB/s** |
| FLOPs, global step only | 0.17 GFLOP → **2.1 GFLOP/s** |
| FLOPs, prefill `S=200` | **34 GFLOP** (once per text chunk) |

Two conclusions:

1. **The decoder needs ~0.17 GFLOP but must move 340 MB per step.** Arithmetic
   intensity ≈ 0.4 FLOP/byte. Everything that matters is bandwidth, latency and
   kernel fusion; nothing matters about peak FLOPS. This is why a WGSL engine is
   *not* absurd — the problem is small — and equally why a *generic* graph
   interpreter in WGSL would be a waste: the only thing to get right is "read each
   weight byte exactly once per step".
2. **The current ORT path is leaving a large win on the table.** The fused
   `local_fixed_sampled_frame` graph re-streams the 28.3 MB local layer 17 times
   per frame, and computes a 16 384-wide `text_lm_head` to use **2** columns.
   *Caveat added after §9 analysis:* this is a **CPU-path** win, not a universal
   one. A modern desktop GPU has 32–72 MB of L2, so re-reading a 28.3 MB layer
   17× back-to-back is largely **L2-resident** and nearly free — the thing that is
   *not* free on any backend is the 50 MB text head reduced to 2 columns (an
   unavoidable read, or a 2-column gather instead). So the GPU win comes from
   elsewhere: f16 weights, prefill compute, and the absence of the
   JS↔wasm↔ORT-graph boundary. Verify with the existing profiler
   (`local_transformer.local_fixed_sampled_frame_session` timing) before
   committing to any optimisation story.

### 7.2 Is 3.6–13.4 GB/s reachable?

| Target | Effective HBM/unified bandwidth | Verdict at 3.6 GB/s (fused+f16) | Verdict at 13.4 GB/s (unfused f32) |
| --- | --- | --- | --- |
| Desktop dGPU (RTX 3060+, RX 6700+) | 250–900 GB/s | ~1.5% duty — trivial | ~5% — fine |
| Apple M-series (unified) | 100–400 GB/s | trivial | fine |
| Modern laptop iGPU (Arc, Iris Xe, Apple M1) | 30–150 GB/s | 2–12% — fine | 9–45% — workable, thermal/power sensitive |
| Old iGPU / Intel UHD, thin-and-light | 10–25 GB/s | 15–36% — tight but plausible | **over budget** |
| Phone (Adreno/Mali) | 10–30 GB/s | borderline | **no** |

So: **WGSL-only is viable on desktop and modern laptop GPUs, and not viable on
low-power/mobile GPUs.** The honest statement is that a WGSL engine is *hardware
eligible for most of the audience*, not *portable*. Two device questions decide
it, and both are cheap to answer: (a) `adapter.limits.maxStorageBufferBindingSize`
and `maxBufferSize` (see 7.3), and (b) a 30-minute micro-benchmark of a
340 MB streaming-read kernel on the actual target machines.

### 7.3 WGSL-specific constraints (the ones that will bite)

| Constraint | Default limit | Consequence |
| --- | --- | --- |
| `maxStorageBufferBindingSize` | 128 MiB | the 440 MB global blob needs ≥4 chunks **unless** you `requestDevice({requiredLimits})`; many desktop adapters report 1–2 GB, mobile much less |
| `maxBufferSize` | 256 MiB | same: a single "one big weight buffer" design (cleanest: index arithmetic in-shader, no bind-group churn) needs an explicit limit request and a fallback |
| `maxStorageBuffersPerShaderStage` | 8 | forbids bindless; argues for **one packed buffer + offsets**, chunked by hand if the adapter can't |
| `shader-f16` feature | optional | halves the 3.6 GB/s to 1.8 GB/s; must be feature-detected, with an f32 fallback path |
| `maxComputeInvocationsPerWorkgroup` | 256 | GEMV/GEMM tiling must fit 256 threads/wg; standard |
| dispatch overhead | ~5–20 µs | ~48 kernels/frame if one-per-op, so **fuse to one kernel per transformer block** (12 dispatches for the global step, not 48) |
| no f64 | — | irrelevant; the graphs are fp32/fp16 |
| WebGPU in workers | ✅ `navigator.gpu` exists in `DedicatedWorkerGlobalScope` | engine stays in `src/workers/tts.worker.ts`; no architecture change |
| COOP/COEP | not required for WebGPU | can be dropped once ORT is gone — but *only* if nothing else needs threads |
| GPU→CPU readback | `mapAsync` is a sync point | keep token ids on the GPU. The existing in-graph `fixed` sampling path already does this (random `u` in → token out); re-implement inverse-CDF sampling in WGSL with a seed, and read back only `should_continue`/token ids (a few bytes) |
| waveform streaming | 30 KB/frame out | trivial; but needs double-buffered readback or the worklet starves |

### 7.4 What "totally WGSL" adds on top of the kernel work

* **Convolutions for the encoder.** The decode graphs contain only 1×1 convs (which
  are just GEMMs). The **encode** graph has 32 real convs with strides/padding plus
  16 nearest-codebook searches. WGSL conv kernels (im2col+GEMM or direct) are a
  separate, self-contained ~1–2 week project. Nothing forces you to do it on the
  GPU: cloning is a one-shot, non-latency-critical path.
* **No CPU fallback.** Firefox < 141, older Safari, locked-down enterprise browsers,
  and every low-bandwidth GPU get *nothing* unless you write a second backend.
  Options: (i) ship CPU-JS/wasm as the fallback (2 backends to validate), (ii) keep
  ORT only as the fallback (then you have not removed the dependency or the
  licensing question), (iii) require WebGPU and show a clear "unsupported browser"
  screen (simplest, but a real UX regression for a web-only app whose selling point
  is "it just works in a browser").
* **Prefill is where GPU compute genuinely pays.** 34 GFLOP at `S=200` is ~1–3 s on
  the current CPU path and ~50–100 ms on any real GPU. That is a 1–2 s cut in
  time-to-first-audio per text chunk — larger than any other single win available,
  and the strongest *user-visible* argument for the WGSL route.

### 7.5 Effort estimate (one engineer, fluent in WGSL)

| Work item | WGSL | JS/wasm (CPU) |
| --- | --- | --- |
| protobuf weight reader + packing | 3–5 d | 3–5 d |
| global prefill (batched GEMM) | 3–5 d | 3–5 d |
| global decode (fused block, ring KV, RoPE) | 5–8 d | 5–8 d |
| local head, 3 sample modes, fused 17-step frame | 3–5 d | 3–5 d |
| streaming codec decoder (4 stages, 12 ring caches, `decoder.8` head — **unverified**) | 8–12 d | 8–12 d |
| clone encoder (convs + RVQ argmin) | 8–12 d | 8–12 d |
| parity harness + corpus | 5 d | 5 d |
| fallback backend (only if required) | +8–12 d | — |
| **Total to replace ORT entirely** | **≈ 8–12 weeks** | **≈ 6–8 weeks (+ fallback) / 4–5 without cloning** |

The kernels are the same work in both columns — the difference is that WGSL needs
its own conv stack and has no safety net, while the CPU route can be validated
against ORT side by side the whole way.

### 7.6 Recommendation

**Do both, in this order, behind one interface:**

1. Build the parity harness first (Phase 0 of §6.6) and **measure the current
   per-frame session timings** to confirm the 582 MB local-head claim. If the
   local head really dominates, a CPU-fused engine already buys most of the win.
2. WGSL engine as the **primary** path, but structured as
   `EngineBackend { prefill, decodeStep, localFrame, codecStep, encode }` so a
   second backend is a drop-in.
3. **Fallback decision is a product decision, not a technical one** — see the
   question below. My recommendation: CPU-JS backend (no ORT) for the ~15% of
   sessions without WebGPU, because it keeps the "zero ORT, licence-clean"
   property that is the *actual* reason for this project (§10.1). If you are
   willing to keep ORT purely as a fallback, the schedule halves but so does the
   point of the exercise.

## 8. Questions for the review

1. **Scope of "no ONNX GPU runtime"** — answered as (a) *no ORT at all* plus
   (b) *WGSL-only compute*, see §7. Confirm that reading.
2. **Fallback browser policy** (§7.4/§7.6). WGSL-only leaves Firefox < 141,
   older Safari and low-bandwidth GPUs with no path to audio. Choose: a second
   CPU-JS backend (recommended, keeps the licence win), ORT-as-fallback-only
   (halves the schedule, keeps the dependency), or a hard WebGPU requirement
   with an "unsupported browser" screen. Which do you want?
2b. **Is the clone (encoder) path in scope for GPU?** It is the only part that
   needs real convolution kernels in WGSL, it is a one-shot offline operation, and
   it accounts for ~2 of the 8–12 weeks (§7.5). Keep it on the CPU, or move it?
3. **Weight precision.** Do we keep fp32 (763 MB) for parity simplicity, or is
   the f16 re-export (PLAN.md §10.2) on the table for this work? f16 halves the
   download and the bandwidth, but adds a numerics variable to the parity work.
4. **Cloning (encoder).** See 2b. If a CPU backend exists, the encoder lives
   there and the GPU backend only ever sees TTS + codec decode.
5. **The `decoder.8` waveform head** is the one piece I could not confirm
   statically (op-level reverse engineering of the final reshape/overlap-add).
   Do you want me to pin it down empirically (a one-off local run of the codec
   graph outside the app) before we commit to the codec re-implementation?
6. **Bugs 1 and 2 in §5** — do you want them fixed now (bug 2 is a real
   user-visible crash on the Settings page when profiling is on), independently
   of the re-implementation?
7. **`vendor/` fate.** If phases 1–3 land, do we delete
   `src/lib/tts/vendor/mossTtsRuntime.js` and the 11 patches, or keep them
   frozen as the reference/parity oracle?

---

## 9. Decode-first design: a WGSL engine with no ONNX Runtime

Decisions taken: **no ORT**, **WGSL-only** (no CPU fallback, legacy platforms out
of scope), **decode first** (prefill and the clone encoder are follow-ons).

### 9.1 Facts extracted from the graphs (do not re-derive these)

Verified by reading the published `.onnx` protobuf structures.

**Global transformer, `moss_tts_decode_step.onnx`**

* One fused QKV projection per layer, `c_attn [768, 2304]`, laid out
  `q = xW[0:768]`, `k = xW[768:1536]`, `v = xW[1536:2304]` (the export splits with
  `Concat` on axis 1, i.e. `[B,S,3,768]`).
* Attention: `qᵀ [0,2,1,3]`, `kᵀ [0,2,3,1]` (this is the perm the exporter
  patches in), `scores = q·kᵀ`, `× 0.125` (= `1/√64`, confirmed as a literal
  constant), mask = `key_pos < past_valid_lengths + 1  ∧  key_pos ≤ query_pos`,
  masked value `-3.4028235e38` (f32 min), `Softmax(axis=-1)`, `@ V`, merge heads,
  `c_proj`, residual; then `ln_2 → fc_in[768,3072] → GELU(tanh) → fc_out → +residual`,
  `ln_f`. All ONNX `MatMul` initializers are **`[K, N]` row-major**, which is
  exactly the layout a `y = W·x` GEMV wants: **no transposes needed anywhere.**
* RoPE constants: `inv_freq = 10000^(−2i/64)`, i = 0..31 (read out of the graph and
  verified against the formula), `cos/sin` produced by `Einsum(pos, inv_freq)`.
* The rotary *pairing* is the one unresolved detail: the graph builds `cos` by
  `Tile [1,1,1,2]` (duplicated halves) while the exporter's `rotate_half` is
  interleaved (`cat(−x[1::2], x[::2])`). Those two conventions are mathematically
  incompatible, so one of my two static readings is wrong. **Resolve empirically
  (see 9.6) before writing `k_rope`.** Do not guess.

**Codec streaming decoder, `moss_audio_tokenizer_decode_step.onnx`**

Per attention layer (verified on `decoder.1.transformer.layers.0`, `C = 500`):

```
cur_pos      = attn_offset_i                                   # int32 [1]
slot_pos     = cur_pos + arange(C)                             # absolute position per ring slot
stored_pos   = concat(cached_positions_i [C], slot_pos [C])    # [2C]
delta        = slot_pos − stored_pos
valid        = (stored_pos ≥ 0) ∧ (delta ≥ 0) ∧ (delta < C)    # Constants 31/32/33 = 0/0/500
K_all        = concat(cached_keys_i [C], rope(k, cur_pos))     # rotate on WRITE
V_all        = concat(cached_values_i [C], v)
scores       = softmax((q·K_allᵀ)·0.125  masked with −inf where ¬valid) @ V_all
cached_keys_i'   = last C entries of K_all
cached_values_i' = last C entries of V_all
cached_positions_i' = last C entries of stored_pos
```

So the "ring" is a shift-append in the graph. The equivalent, cheaper WGSL form is
an explicit ring: `slot = pos % C`, rotate on write, mask with
`0 ≤ cur − stored < C`. Contexts: **500 / 1696 / 2480 / 3248** for the four groups
of 3 layers. `transformer_offset_i` is the per-stage frame counter.

* Head: `input_proj [768,512,1]` (1×1 conv = GEMM), 16 × `Conv(512,8,1)` codebook
  projections → `output_proj [768,512,1]`, four stages at `d=256`
  (`in_proj [256,768]`, `out_proj [256,256]`, SwiGLU `ffn.0 [256,1024]` /
  `ffn.2 [1024,256]`, LayerScale), then `decoder.8` → waveform.
* The graph's tail is `Reshape → [1,2,T] → Cast f32` with a
  `Where(x/2 − 1, x/2)` epilogue and windowed `Gather/Mul/Concat` (overlap-add)
  driven by `/decoder.8/Shape*` and `Div` by the *input length*. **The exact
  window length, hop and overlap ratio are not recoverable from the graph alone
  without running it — resolve empirically (9.6).** 48 kHz ÷ 12.5 Hz = 3840
  samples/frame/channel, and the last projection is `MatMul[256, 240]`.

### 9.2 Weight layout: keep the ONNX bytes, don't convert

`[K, N]` row-major, fp32, contiguous, is already optimal for a GEMV. The loader is
therefore trivial and **zero-copy**: read the initializer's `(offset, length)`
from the `.onnx` protobuf, and — since Mockbird already holds the `.data` sidecar
in OPFS — `new Float32Array(buffer, offset, count)`. No reordering, no packing
step, no `f16` conversion in v1.

The engine keeps **one** small metadata buffer: for each tensor,
`{offsetElems, dims[4], kind, layerIndex}` plus RoPE constants, ε values, reserved
token ids. All kernels address weights as `base + layerOffset + k*N + n` through
this table, so there is a single storage buffer and therefore a single bind group
per pipeline.

Consequence: `maxStorageBufferBindingSize` (default 128 MiB) is the binding
constraint — the 440 MB sidecar must be **chunked into ~4–8 storage buffers** and
the metadata table carries which chunk. That is a uniform-array lookup, not a
bind-group change, so it costs nothing at dispatch time.

### 9.3 Buffers

| Buffer | Size (fp32) | Notes |
| --- | --- | --- |
| `weights` (4–8 chunks) | 670 MB | immutable after upload; the only thing uploaded |
| `kvGlobal` | `12·2·1024·768·4` = 75 MB | ring, cursor = `past_valid_length`; grows to `maxN = prefill + max_new_frames` |
| `kvLocal` | < 1 MB | 1 layer × 17 positions |
| `kvCodec` | 65 MB | 12 caches, `Σ C = 31 696` positions, keys stored **post-RoPE** |
| `positionsCodec` | 127 KB | int32 |
| `scratch` (arena) | 2–4 MB | fixed sub-ranges; bind group never changes |
| `staging` (mapped) | 2 × 64 KB | PCM out + token/flag readback, double-buffered |
| `uniforms` | 4 KB | per-dispatch: offsets, cursor, sampling params, seed, flags |

`max_new_frames = 375`, so `maxN = 1024` is generous; sizing it from the manifest
(`prefill_rows + max_new_frames`) makes the allocation exact.

### 9.4 The per-frame schedule

`generateAudioFrames` becomes: prefill once, then per frame —

```
k_embed            1 dispatch   wte[assistant_slot] + Σ_k audio_emb_k[code_k]   → [768]
k_global_step     48 dispatches 12 layers × 4, no weight byte read twice
   ├ qkv_gemv        c_attn 7.1 MB + RoPE(k) → ring write
   ├ attn + proj     KV ring N·6 kB + c_proj 2.4 MB → residual
   ├ fc_in + GELU    9.4 MB
   └ fc_out          9.4 MB → residual
k_local_frame     17 dispatches 17 sequential steps; 2 text-head columns, 16 audio heads
k_sample           1 dispatch   top-k 25 → top-p 0.95 → inverse-CDF vs u (on GPU)
k_codec_frame     ~16 dispatches RVQ dequant + 4 stages + decoder.8 → 30 KB PCM
k_readback         copy         16 tokens + should_continue (+ PCM when due)
```

≈ 83 dispatches/frame ≈ 1–2 ms of dispatch overhead against an 80 ms budget
(12.5 fps) — acceptable, and it is *why* the layer is split into 4 rather than
fused into 1: core WGSL has no grid-wide barrier, so a single-dispatch block would
need one workgroup for the whole layer and would leave the GPU ~99% idle.

**Token ids never leave the GPU.** Sampling runs on-GPU against a `u` drawn from a
seed in the uniforms (this is exactly what the existing `local_fixed_sampled_frame`
graph does, which is presumably why that mode exists and is the default). Only
16 ids + 1 flag + PCM cross the boundary.

**Stop / cancel / pace** stay in JS, unchanged from the vendored runtime:
`should_continue` and the lead-time heuristic (§3.5) are pure logic. Cancellation
is a flag in the uniform buffer, checked at each layer boundary.

### 9.5 The sampling kernels (on GPU, matching current semantics)

```
penalised[j] = seen[j] ? (logit[j] < 0 ? logit[j]*p : logit[j]/p) : logit[j]   // HF convention
topk         = 25 largest of penalised
probs        = softmax(topk)
keep         = prev_cumsum < 0.95            // graph semantics, NOT the JS shift variant
u            = f32(clamp(rand, 0, 0.99999994))
token        = topk[ #{ i : cumsum[i] < u } ]
```

One workgroup per channel, 1024 candidates — trivial. `greedy` mode replaces it
with a masked argmax; `full` mode (JS sampling) is the one mode that *would* need
a readback per step, and it can be dropped in v1 because `greedy` and `fixed`
cover the product (the UI default is `fixed`).

### 9.6 Validation gate 0: a headless golden-trace oracle — **built, see §10**

Everything above had three unknowns that static reading could not settle. The
oracle is now implemented and run; §10 records what it settled, the one item it
left open (§10.3) and the exact plan to close it.

Cost: `onnxruntime-node` installed in `/tmp` (never in the repo) + the decode
weights: `moss_tts_global_shared.data` 440.8 MB, `moss_tts_local_shared.data`
229.7 MB, `moss_audio_tokenizer_decode_shared.data` 44.2 MB → **≈ 715 MB, one
time.** Emits `tests/fixtures/golden/*.json`: per-frame token ids, the stop step,
`global_hidden` checksums per step, and the decoded PCM hash.

With that file, a kernel is "done" when it reproduces the golden trace. Without
it, every kernel is a guess and a frame-token flip is invisible until a human
listens to a WAV. **I recommend this before writing any WGSL.**

### 9.7 Build order

| # | Deliverable | Gate |
| --- | --- | --- |
| 0 | golden-trace oracle + resolve the 3 unknowns | tokens + PCM reproduce the current build |
| 1 | device init, limits report, weight loader, metadata table, `k_embed` | input vector matches golden checksum |
| 1a | ✅ **done:** `src/lib/tts/engine/onnxWeights.ts` (loader) + `kernels.ts` (gemv / embed / layernorm / rope / gelu / residual) | loader validated on the real 440.8 MB graph: 248 tensors, zero-copy, 12×4 layer-major matrices, Identity aliases resolved (`tests/gpu/loader.test.mjs`) |
| 2 | `k_global_step` (prefill S=1 seed + greedy only) | 12-layer hidden matches; 40+ steps token-exact |
| 3 | `k_local_frame` greedy/fixed + `k_sample` | full utterance token-exact (all 18 voices) |
| 4 | `k_codec_frame` + `decoder.8` + ring caches | per-frame PCM within tolerance, streaming parity |
| 5 | prefill (batched, `S>1`) — the first real GEMM | TTFA ≤ today; this is where f16 pays off |
| 6 | f16 / int8 weights, PGO-ish kernel tuning | ≤ 30% of today's per-frame bytes |
| 7 | clone encoder kernels (the only conv work) | prompt codes match the codec encoder |
| 8 | delete `vendor/mossTtsRuntime.js`, `onnxruntime-web`, `copy:ort`, COOP/COEP | `pnpm verify` green with no ORT in the graph |

Steps 1–4 are the decode path the review asked to focus on, and they are
self-contained: no prefill batching, no quantization, no encoder.

### 9.8 Open items for this round

1. **Oracle download (~715 MB into `/tmp`, `onnxruntime-node` in `/tmp`)** — go or
   no-go? Without it I can write the design but the first three kernels are
   unverifiable guesses.
2. **Headless WGSL test harness.** WGSL kernels can only run in a browser without
   a native binding. If you want kernel-level tests in CI, that means a Node
   WebGPU binding (`webgpu`/Dawn) in `devDependencies`; otherwise validation goes
   through the browser e2e you already own. Which?
3. ~~Sample modes~~ — **settled: greedy only.** The published model ships no
   `local_greedy_frame`, so the engine implements the 17-step `local_cached_step`
   chain with GPU-resident argmax + repetition penalty. No top-k/top-p kernel.
4. ~~KV ring size~~ — **settled: exact from the manifest**, i.e.
   `prefill_rows + max_new_frames` (570 for the committed fixture).
5. **Next step:** close §10.3 (index-based graph surgery to pin the stored KV
   layout) and then start step 1 (device + weight loader + `k_embed`) against the
   golden trace? Or do you want the codec's `decoder.8` head pinned first, since
   it is the other unresolved piece and it gates step 4?

---

## 10. Oracle results — what is now *verified* (step 0 complete)

Decisions taken: WGSL-only, no fallback, **greedy only**, KV ring sized from the
manifest. The oracle was built and run: `onnxruntime-node` 1.30.0 in `/tmp`, the
full decode weight set (715 MB) fetched from Hugging Face, and the greedy path
replayed headlessly. A golden trace is committed at
`tests/fixtures/golden/greedy-junhao-zh.json`.

### 10.1 Facts confirmed against the real graphs

| Fact | Value | How verified |
| --- | --- | --- |
| Prefill rows for voice `Junhao` + `text_samples[0]` | **195** (98 prompt frames) | manifest template assembly |
| KV ring size | `prefill_rows + max_new_frames` = **195 + 375 = 570** | decision (4) — allocate exactly this |
| Global model | 12 layers, `d=768`, 12 heads, head_dim 64, vocab 16 384 | initializers + `tts_browser_onnx_meta.json` |
| Attention scale | literal `0.125` (`1/√64`) in the graph | constant decode |
| Masked value | f32 min `-3.4028235e38`, causal mask confirmed | `query=3` support `{0,1,2,3}`, row sum 1.0 |
| LayerNorm | ε = `1e-5`, **scale ≡ 1.0** (all layers) | LN scale tensors read as all-ones |
| RoPE base / freqs | `10000^(−2i/64)`, 32 freqs | graph constant matches the formula to 6 digits |
| **RoPE pairing** | **interleaved** `rotate_half`: partner `d^1`, sign `−` for odd `d` | graph's own `Concat_3` has shape `[1,195,12,32,2]` and its values are `[-v₁, v₀, -v₃, v₂, …]` |
| Attention rank | exactly 64 per head (127-key block) ⇒ standard MHA | Gaussian elimination on the score matrix |
| Weights are `[K,N]` row-major | `c_attn[768,2304]` etc. | verified: `y[n]=Σₖx[k]·W[k·N+n]` reproduces `present_value_0` to **1.4e-6** |
| Greedy path used by the app | **17 × `local_cached_step`** + host argmax | `local_greedy_frame` is **absent** from the published model |
| Codec frame size | **3840 samples/channel/frame** exactly (12 × 3840 = 46 080; 40 × 3840 = 153 600) | decode output lengths |
| Streaming ≠ full codec graph | rms ≈ 0.12, max ≈ 1.25 | `codec.streamingVsFull` in the golden trace |

Two of these change the plan: **greedy-only deletes the top-k/top-p kernel
entirely** (argmax + repetition penalty only), and the **streaming codec graph is
not bit-equivalent to the full graph**, so the engine must reproduce the
*streaming* graph — that is what the app plays.

### 10.2 Loader requirements discovered (these would have broken a naive loader)

1. **Weights live in three places**, not one: external initializers (`.data`
   sidecar, `offset`/`length`), inline `Constant` node attributes, and
   `Identity` aliases. Every LayerNorm scale except layer 0's arrives via
   `Identity` → an anonymous initializer. All 1 199 parameters resolve only when
   all three are handled.
2. **Inline protobuf payloads are not 4-byte aligned** — `new Float32Array(buf,
   off, n)` throws `start offset … should be a multiple of 4`. Copy-then-view.
3. External views *are* alignable, so the sidecar path stays zero-copy.
4. `g.get(...)` must resolve through the alias chain; `onnx::MatMul_*` names are
   anonymous, so **layer order is recovered from sidecar byte offsets**
   (layer-major: `c_attn, c_proj, fc_in, fc_out` per layer, ascending).

### 10.3 One unresolved item — **SOLVED in §10.5**

`present_key_0` (the stored key cache) is **not** reproduced by any rotation of
any of the three projection slices under 30+ tested conventions
(head-major/dim-major × {none, interleaved, half-split, half-split-sign-flipped}
× {`d>>1`, `d&31`} × flat-768 variants × position deltas −1/0/+1), even though
the control (`present_value_0` = slice 3) matches to 1.4e-6 and the *pairing* is
proven interleaved.

Root cause identified: **node names in these traced graphs are not unique.** The
same `/Concat_5`, `/Reshape_1` … labels appear in multiple layers, so name-based
graph surgery silently binds to the wrong node (ORT itself reported a type
mismatch for a `/Concat_5` that my dump showed as the attention K).

Resolved in §10.5: the two causes were (a) node names repeat across
instantiated submodules and the file is not in topological order, so every
name-addressed probe was reading a different layer than intended, and (b) an
**inverted sign** in my `rotate_half` partner rule. Fixing the tooling (index
addressing + end-to-end rename) and reading the rotation factors back out of the
graph's own `Mul` nodes produced the convention to 1.7e-7.

### 10.4 Headless WGSL harness (step 1 infra, in the repo)

`pnpm test:gpu` runs `tests/gpu/device.test.mjs` + `tests/gpu/kernels.test.mjs`
against a real adapter via the `webgpu` (Dawn) binding — **no browser needed**,
which is what makes kernel-level CI possible for a runtime with no CPU fallback.

Device facts this box reports (the harness asserts them):

| Limit | Value | Consequence |
| --- | --- | --- |
| `maxStorageBufferBindingSize` | **134 217 728 (128 MiB)** | the 440 MB sidecar **must** be chunked; confirmed mandatory, not defensive |
| `maxBufferSize` | 4 GiB | a single packed weight buffer *is* possible if requested |
| `maxStorageBuffersPerShaderStage` | 16 | one packed buffer + a metadata table beats 8–16 chunked bindings |
| `shader-f16` | present | halves the 3.6 GB/s figure to ~1.8 GB/s |
| `timestamp-query`, `float32-filterable` | present | profiling and cheap packed formats available |

Kernel conformance (synthetic weights, **real shapes**, JS reference):

```
ok   layernorm d=768 eps=1e-5                       maxErr=2.38e-7
ok   gemv [K=768, N=2304] c_attn (abs)              maxErr=1.19e-6   (rel 1.7e-4)
ok   rope interleaved rotate_half d=64 base=10000   maxErr=4.23e-6
ok   attention vs ring cache N=512 heads=12         maxErr=7.45e-8
ok   argmax over 1024 logits (tie -> lowest id)     maxErr=0
```

The harness earned its keep immediately: the first run **failed 3 of 5** kernels,
all with the same root cause — a 768/1024-element reduction performed with only
256 lanes and no strided loop, silently dropping the tail. That is precisely the
class of bug that would have produced "almost right" audio in a browser.

### 10.5 The rotary convention — resolved by measurement

The §10.3 open item is **closed**. The blocker was tooling, not the model: these
graphs are **not in topological file order and tensor names repeat across
instantiated submodules**, so name-addressed graph surgery silently reads the
wrong node. `scripts`-worthy fix: address nodes by **index**, and *rename* the
probe tensor end-to-end (node outputs, every consumer's input, and If/Loop branch
graphs, which live in `AttributeProto.g` = field 6).

Once the probes bound the right nodes, every convention fell out of the data:

```
split0 = qkv[0:768]    → Q  → RoPE → score query
split1 = qkv[768:1536] → K  → RoPE → stored in the key cache
split2 = qkv[1536:2304]→ V  → raw, stored in the value cache
```

(the `Split` node at index 281 is what makes this unambiguous).

**RoPE, measured** — `cos = Mul(split · …)`, so the factors were divided straight
out of the graph:

```
measured cos[pos=7] = 0.75390 0.75390 0.51145 0.51145 -0.70043 -0.70043 …
                       └ pairs (2i, 2i+1) share one frequency
cos(pos·10000^(−2(d>>1)/64))  matches to 1.7e-7   (the "duplicated halves"
                                                  variant d&31 is wrong by 1.98)
```

**rotate_half** — the graph's own `Concat_3` has shape `[1,S,12,32,2]` and its
values are `[-v₁, v₀, -v₃, v₂, …]`, i.e. standard GPT-J interleaved with the
negation on the **even** element:

```
out[d] = v[d]·cos(d>>1) + (−v[d+1] if d even else +v[d−1])·sin(d>>1)
```

Head layout is **[1, S, 12, 64] head-major** (verified: the graph's own
post-RoPE Q·K reproduces its score matrix to 1.44e-6, head-major; dim-major is
off by 39).

| Verified quantity | Error |
| --- | --- |
| Q / K post-RoPE vs recomputed | 2.4e-6 / 1.3e-6 |
| value slice vs `present_value_0` | 9.6e-7 |
| attention context vs `/MatMul_1` | 6.6e-7 |
| layer-0 residual vs `Add_19` | 7.5e-7 |
| **full 195-row prefill `global_hidden`** | **2.1e-6** |
| **40-frame greedy token stream** | **40/40 exact** |

`ref-decode.mjs` is now a complete CPU reference for the decode path
(prefill + 12 global layers + 17-step local head + greedy sampling) that
reproduces the oracle's tokens exactly. It is the thing the WGSL kernels are
ported from and diffed against.

Three bugs this comparison caught, all of which would have produced "almost
right" audio:

1. **Softmax never normalised** — `exp` was used un-divided by the sum. Harmless
   at position 0 (one key), wrong everywhere else.
2. **Layer/MLP not interleaved** in the decode step: all 12 attention layers ran,
   *then* all 12 MLPs, instead of attention→MLP per layer.
3. **Text-head columns confused with hidden values** — the assistant/end decision
   needs 2 columns of the 16 384-wide `text_lm_head`, not `hidden[7]`/`hidden[9]`.

### 10.6 The codec decoder — head and upsampler resolved

`decoder.8` is **not** a learned expansion. Measured shapes (T = 8 input frames):

| Stage | input | output | attention cache |
| --- | --- | --- | --- |
| RVQ + front | 8 × 16 codes | `[1, 32, 768]` = 4T | — |
| 1 (4 layers) | `[1, 4T, 256]` | `[1, 4T, 768]` | 500 |
| 2 (2 layers) | `[1, 8T, 256]` | `[1, 8T, 768]` | 1696 |
| 3 (2 layers) | `[1, 16T, 256]` | `[1, 16T, 768]` | 2480 |
| 4 (4 layers) | `[1, 32T, 256]` | `[1, 32T, 240]` | 3248 |
| head | `[1, 32T, 240]` | `[1, 2, 120·32T]` | — |

**The head is a pure layout transform** — the two transposes in it cancel, and
the only real work is an interleaved de-interleave of the channels:

```
audio[c][t·120 + j] = latent[t][2j + c]        # verified: maxErr 0.000e+0
```

**Each stage upsamples ×2 in time by splitting its 768 channels into an
*interleaved* pair of 384-wide sub-bands that become consecutive time steps**:

```
new[2t + r, m] = old[t, 2m + r]                 # verified: maxErr 0.000e+0
input_proj: 384 → 256   (so the 768 = 2×384 is a sub-pixel upsample,
                         NOT a nearest-neighbour repeat — hypothesis A was 5.6e-2 off)
```

So the whole codec decoder is: RVQ dequant → 4 × (attention over a ring cache →
output_proj → interleaved ×2 sub-pixel) → one reshape. The attention is plain
64-dim MHA at `d=256` with SwiGLU, LayerScale and RoPE, and the ring cache is
exactly the shift-append measured in §9.1, which in WGSL becomes
`slot = pos % C` with `valid = 0 ≤ cur − stored < C`.

### 10.7 The RVQ front — the last unmeasured piece, resolved

The ×4 is *inside* the quantiser output, not after it. Measured: the 16-quantiser
sum plus `output_proj` yields `[1, 768, T]` (channel-major), and `decoder.0`
expands it:

```c
// codes [1, T, 16]
for (k = 0; k < 16; k++)                       // 16 codebooks, 1024 x 8 each
    e = codebook_k[codes[:, k]];                // [T, 8]
    acc[:, t] += W_k[512, 8] · e[t, :] + b_k;    // summed over k -> [512, T]
g[:, t] = W_out[768, 512] · acc[:, t] + b_out;  // -> [768, T]
stage_in[4t + s, m] = g[4m + s, t];             // x4, INTERLEAVED sub-pixel  (m < 192)
h = input_proj(stage_in)                        // 192 -> 256, then stage 1
```

verified `maxErr = 0.000e+0`.

**Everything in the decode path is now measured.** The complete, verified
specification is:

```c
// ---- per text chunk -------------------------------------------------------
rows  = [user_prefix…, audio_start, (user_slot, 16 codes) × Tprompt,
         audio_end, after_reference…, text…, assistant_prefix…, audio_start]   // 17 wide
prefill(rows)   // 12 x { LN, QKV[768,2304], RoPE, causal attn x0.125, proj, LN, MLP[3072] }, ln_f
for each frame:
    local head (17 steps, fresh KV each frame):
        step 0: embed=global_hidden -> heads -> argmax(text_lm_head[:, {9, 7}])  // 9 = keep, 7 = stop
        step 1: embed=wte[9]        -> audio_lm_heads[0] -> argmax(penalty 1.2)
        step 2+i: embed=audio_emb[i-1][prev] -> audio_lm_heads[i] -> argmax(penalty 1.2)
    global decode_step(row = [9, frame])   // appends to the 24 KV tensors
    codec frame:
        RVQ front (§10.7) -> 4 x { attention over a ring cache, output_proj, x2 sub-pixel }
        audio[c][t*120 + j] = latent[t][2j + c]
```

Conventions that took measurement rather than reading: the rotary pairing and
its sign (§10.5), the head-major KV layout, the interleaved sub-pixel upsamplers
(×4 with 192-wide sub-bands at the front, ×2 with 384-wide sub-bands between
stages), the interleaved channel de-interleave in the codec head, and that the
whole `decoder.{0,2,4,6,8}` family is a reshape/transpose pair whose two
transposes cancel.
