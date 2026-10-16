# ONNX probe (analysis tooling)

Minimal, dependency-free ONNX reader/surgery used to pin down the MOSS-TTS-Nano
graph conventions recorded in `docs/moss-tts-runtime-analysis.md` §10.

The published graphs are **not in topological file order** and their **tensor
names repeat across instantiated submodules** (the codec decoder contains four
copies of the same `decoder.8` module). Any tooling that addresses nodes by name
silently reads the wrong layer — which is exactly how a first pass produced a
rotary convention that matched nothing in 30+ variants. Everything here is
addressed by **topological node index** instead, and injected probe tensors are
**renamed end to end** (producer output, every consumer input, and `If`/`Loop`
branch graphs, which live in `AttributeProto.g` = field 6).

```bash
# structural listing / forward trace from a node
python3 tools/onnx-probe/onnxgraph.py list <model.onnx> --from 0 --to 40
python3 tools/onnx-probe/onnxgraph.py trace <model.onnx> --from 277

# expose node N's output as a graph result (use index, not name)
python3 tools/onnx-probe/onnxgraph.py patch <in.onnx> <out.onnx> 279 295 309
python3 tools/onnx-probe/onnxgraph.py patch <in.onnx> <out.onnx> 3254:3277:3288
#   the `:i32` suffix declares a non-float output; the renamed tensor is
#   reported as __probe_0__, __probe_1__, … in the order you listed it
```

Model files are not committed; point it at a directory holding the published
graphs (see the oracle recipe in the analysis doc).

## `ref-decode.mjs` — the CPU reference for the decode path

The port target for the WGSL engine: a complete, dependency-free JS
implementation of the MOSS-TTS-Nano greedy decode loop (prefill + 12 global
layers + the 17-step local head + greedy sampling), written against the measured
conventions in `docs/moss-tts-runtime-analysis.md` §10.5.

It reads weights through the **shipped loader** (`src/lib/tts/engine/onnxWeights.ts`,
which Node runs natively via type stripping), so there is no second weight
reader to drift.

```bash
MOCKBIRD_MODEL_DIR=/path/to/weights node tools/onnx-probe/ref-decode.mjs 40
# -> prefill rows = 195
# -> EXACT frame matches: 40/40  ✅   (against tests/fixtures/golden/greedy-junhao-zh.json)
```

The prefill output is cached next to the weights as `ref-prefill.bin` so that
iterating on the local head does not pay the 75 s prefill each time. A WGSL
kernel is finished when it matches this reference, not when it sounds plausible —
a single flipped frame token diverges the rest of the utterance.
