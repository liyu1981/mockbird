# ADR 0004 — Long texts are spoken chunk by chunk, scheduled on the main thread

**Status:** accepted

## Context

MOSS-TTS-Nano only speaks ~75 SentencePiece tokens at a time
(`voice_clone_max_text_tokens`). Anything longer has to be split and spoken
piecemeal, and the vendored runtime already does that internally:
`synthesizeVoiceClone()` calls `splitVoiceCloneText()` and loops over the
result, emitting a pause chunk between the pieces.

Relying on that loop gave a poor experience on the Generate tab:

- The whole text was **one** worker request, so the UI had no idea how far along
  it was — a 40-paragraph essay looked exactly like a one-sentence sample.
- A failure in the second chunk surfaced as a single `generation-error` with no
  hint of which chunk died or how much audio had already been produced.
- Playback was the weak point. The decoder runs several times faster than
  realtime and the worklet ring buffer was 30 s, so anything long enough
  outran the speaker and the *oldest* audio was silently dropped (the
  "playback overflowed" badge), while the user heard the text disappear.

## Decision

Own the chunk loop on the **main thread**, in `EngineProvider`, and ask the
worker for one chunk at a time.

| Concern | Where it lives |
| --- | --- |
| Splitting into sentence-aligned chunks | worker: new `analyze-text` request → `prepareSynthesisText()` + `splitVoiceCloneText()` (the tokenizer is already warm there) |
| Scheduling, pacing, cancel, progress | `EngineProvider` run controller |
| Synthesis of a single chunk | worker: one `synthesize` request per chunk, with `chunkIndex` / `chunkCount` echoed back |
| Playback + accumulation for one WAV | `useStreamPlayer`, unchanged (append, flush once per run) |

Concretely:

1. **Plan.** Speak → `analyze-text` returns `{ tokens, chunks }`. The composer
   runs the same request (debounced) so the user sees "4 chunks · ≈ 38s" *before*
   pressing anything.
2. **Run.** Chunks are dispatched in order, each as its own `synthesize`
   request carrying `chunkIndex`/`chunkCount`. The provider awaits the chunk's
   `generation-done` / `generation-error` before dispatching the next one.
3. **Pacing.** Before each chunk the scheduler waits until the player's ring
   buffer has drained to a lead of `clamp(1.5 × last chunk's generation time,
   2 s, 40 s)`. That is enough to cover the next chunk's compute time (no
   audible gap) while bounding memory and keeping the ring buffer away from its
   overflow behaviour. The worklet's capacity went from 30 s to 90 s as slack.
   For this to work the worklet must report its **fill level while it plays**
   (~10 msgs/s from `process()`); reporting it only on `append` leaves a stale
   value and the scheduler waits forever. The wait is also hard-capped so a
   missing level report can never stall a run.
4. **Breathing room.** `interChunkGapSeconds()` inserts 240 ms (400 ms for very
   short pieces) of silence between chunks, mirroring upstream's inter-chunk
   pause, so the result sounds like one passage.
5. **One clip.** The player is flushed once per *run* (not per chunk) and keeps
   every chunk plus the gaps, so *Save audio* exports the full text as a single
   WAV. A cancelled or failed run keeps whatever was produced.
6. **Errors.** A failing chunk aborts the run with
   `Chunk 3 of 7 failed: …`, and the partial audio is still downloadable.

The worker caches the resolved prompt audio codes (`promptAudioCodesCache`,
keyed by cloned/builtin id), so chunks 2..n do not re-read and re-materialise the
reference codes from OPFS on every request.

## Consequences

- The vendored runtime is used for *one* chunk per call. `splitVoiceCloneText()`
  still runs inside it and is a no-op for a piece that already fits, so no patch
  to `mossTtsRuntime.js` was needed — `scripts/vendor-patches.json` is unchanged.
- Prefill runs once per chunk instead of once per text. That is the same cost as
  the upstream loop and it is what buys progress, pacing and per-chunk errors.
- Memory scales with the clip: ~380 MB of `Float32Array` PCM per 10 minutes of
  audio is held until the run is replaced. The composer warns above ~4 minutes.
- Progress, pacing and cancel all live in one place (`generate()` in
  `EngineProvider.tsx`), which is the file to read first when a long text
  misbehaves.

## Follow-ups

- `scripts/e2e.mjs` still targets the pre-redesign routes (`/studio`, `/model`)
  and the old DOM markers, so it cannot validate this path yet. It needs a
  rewrite (long-text case: N chunks in, one WAV out) before it is trustworthy.
- Spoken-duration estimates in `src/lib/tts/chunking.ts` are heuristics
  (CJK chars ÷ 4.5, words ÷ 2.8); the UI should prefer the measured duration of
  the previous run once enough data exists.