/**
 * Asset-path conformance: the engine must resolve the published manifest's
 * relative references against the *manifest's own* directory, not the model
 * root. Mockbird's OPFS layout (src/lib/model/modelManifest.ts) is
 *
 *   MOSS-TTS-Nano-100M-ONNX/{browser_poc_manifest.json, tts_browser_onnx_meta.json,
 *                            moss_tts_decode_step.onnx, moss_tts_global_shared.data, …}
 *   MOSS-Audio-Tokenizer-Nano-ONNX/{codec_browser_onnx_meta.json, …}
 *
 * and the manifest refers to the codec metadata with a `../` reference. Getting
 * this wrong produced `Model file missing from cache: moss_tts_decode_step.onnx`.
 */

import { CODEC_DIR, TTS_DIR } from "../../src/lib/model/modelManifest.ts";
import { resolveAssetPath } from "../../src/lib/tts/engine/policy.ts";

// the values from the published manifest (see docs §10.1)
const MANIFEST_FILES = {
  tts_meta: "tts_browser_onnx_meta.json",
  codec_meta: "../MOSS-Audio-Tokenizer-Nano-ONNX/codec_browser_onnx_meta.json",
  tokenizer_model: "tokenizer.model",
};

let failed = 0;
const check = (name, detail, ok) => {
  if (!ok) failed += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${name.padEnd(44)} ${detail}`);
};
const expect = (label, actual, wanted) => {
  const ok = actual === wanted;
  if (!ok) failed += 1;
  console.log(`  ${ok ? "ok  " : "FAIL"} ${label.padEnd(44)} ${actual}`);
};

console.log(`manifest dir = ${TTS_DIR}`);
expect(
  "tts_meta",
  resolveAssetPath(TTS_DIR, MANIFEST_FILES.tts_meta),
  `${TTS_DIR}/tts_browser_onnx_meta.json`,
);
expect(
  "codec_meta (parent reference)",
  resolveAssetPath(TTS_DIR, MANIFEST_FILES.codec_meta),
  `${CODEC_DIR}/codec_browser_onnx_meta.json`,
);
expect(
  "tokenizer_model",
  resolveAssetPath(TTS_DIR, MANIFEST_FILES.tokenizer_model),
  `${TTS_DIR}/tokenizer.model`,
);

// the graphs sit next to tts_browser_onnx_meta.json, so they resolve via its dir
const ttsMetaPath = resolveAssetPath(TTS_DIR, MANIFEST_FILES.tts_meta);
const dir = ttsMetaPath.slice(0, ttsMetaPath.lastIndexOf("/"));
expect(
  "decode_step graph",
  resolveAssetPath(dir, "moss_tts_decode_step.onnx"),
  `${TTS_DIR}/moss_tts_decode_step.onnx`,
);
expect(
  "global weight sidecar",
  resolveAssetPath(dir, "moss_tts_global_shared.data"),
  `${TTS_DIR}/moss_tts_global_shared.data`,
);

// every file the manifest declares must resolve the same way — the tokenizer was
// the one that got missed, producing "Model file missing from cache:
// tokenizer.model"
const declared = ["tts_meta", "codec_meta", "tokenizer_model"];
for (const key of declared) {
  const resolved = resolveAssetPath(TTS_DIR, MANIFEST_FILES[key]);
  const ok = resolved.startsWith(`${TTS_DIR}/`) || resolved.startsWith(`${CODEC_DIR}/`);
  check(`model_files.${key}`, ok, resolved);
}

// degenerate inputs must not throw or produce stray separators
expect("empty base", resolveAssetPath("", "a/b.json"), "a/b.json");
expect("leading slash tolerated", resolveAssetPath(TTS_DIR, "/a.json"), `${TTS_DIR}/a.json`);
expect(
  "dot segments dropped",
  resolveAssetPath(`${TTS_DIR}/./`, "./a/./b.json"),
  `${TTS_DIR}/a/b.json`,
);
expect("no parent escape", resolveAssetPath(TTS_DIR, "../../a.json"), "a.json");

console.log(failed ? `\n${failed} path case(s) failed` : "\nall asset-path cases passed");
process.exit(failed ? 1 : 0);
