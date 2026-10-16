/**
 * Conformance test for the TypeScript text normaliser against the upstream
 * Apache-2.0 corpus (46 cases, extracted from `TEST_CASES` in
 * `tts_robust_normalizer_single_script.py`), including idempotence.
 *
 *   node tests/engine/textNormalize.test.mjs
 */
import { readFileSync } from "node:fs";
import { normalizeTtsText } from "../../src/lib/tts/engine/textNormalize.ts";

const corpus = JSON.parse(
  readFileSync(new URL("../fixtures/golden/text-normalizer.json", import.meta.url), "utf8"),
);

let failed = 0;
for (const { name, input, expected } of corpus.cases) {
  const actual = normalizeTtsText(input);
  if (actual !== expected) {
    failed += 1;
    console.log(`FAIL ${name}`);
    console.log(`  input   : ${JSON.stringify(input)}`);
    console.log(`  expected: ${JSON.stringify(expected)}`);
    console.log(`  actual  : ${JSON.stringify(actual)}`);
    continue;
  }
  const second = normalizeTtsText(actual);
  if (second !== actual) {
    failed += 1;
    console.log(`FAIL ${name} (not idempotent)`);
    console.log(`  first : ${JSON.stringify(actual)}`);
    console.log(`  second: ${JSON.stringify(second)}`);
  }
}

console.log(
  failed
    ? `\n${failed}/${corpus.cases.length} normaliser case(s) failed`
    : `all ${corpus.cases.length} normaliser cases passed (incl. idempotence)`,
);
process.exit(failed ? 1 : 0);
