# Test fixtures

## `reference-zh.wav`

A short Mandarin speech clip copied from the upstream
[MOSS-TTS-Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano) repository
(`assets/audio/zh_1.wav`, Apache-2.0, © OpenMOSS) — the same sample the project's
own documentation uses for voice cloning.

`scripts/e2e.mjs` uploads it through the Voices page to exercise the real
codec-encode path (reference audio → audio codes → stored voice → synthesis).

Override the fixture with `MOCKBIRD_REFERENCE_WAV=/path/to/clip.wav`.

## `golden/greedy-junhao-zh.json`

The **decode-path golden trace** for the WGSL engine (ADR 0005). Produced by
running the published ONNX graphs headlessly under `onnxruntime-node` and
recording what a correct implementation must reproduce:

| Field | Meaning |
| --- | --- |
| `frames[].frame` | 16 audio codes per 12.5 Hz frame, greedy — must match **exactly**; a single flipped token diverges the rest of the utterance |
| `frames[].hidden` | min/max/sum/FNV-1a of the 768-d `global_hidden` feeding that frame |
| `prefillHidden` | same statistics for the 195-row prefill output |
| `stopStep` | step at which the local head emits `audio_end_token_id` (`-1` = hit the frame cap) |
| `codec.pcm` | decoded 48 kHz stereo, planar (`[all left…, all right…]`): exact sample count, plus min/max/sum/FNV-1a |

Input: built-in voice `Junhao` (98 prompt frames) + `text_samples[0]`
(20 pre-tokenised text ids), so no tokenizer is needed to replay it.
`codec.streamingVsFull` records that the streaming and full codec graphs are
bit-equivalent to fp32 noise (rms ≈ 2e-7) — the app plays the *streaming* graph,
and this is what confirms the ring-cache mask reproduces the chunked graph.

Regenerating requires the decode weight set (~715 MB from Hugging Face); see
`docs/moss-tts-runtime-analysis.md` §10 for the oracle recipe.
