/**
 * End-to-end verification for Mockbird, driven over Chrome DevTools Protocol
 * against the dev server.
 *
 * Covers, with the real 763 MB model stack:
 *   1. first-run weight download into OPFS (+ resume/progress assertions)
 *   2. engine load, built-in voices, token counter
 *   3. synthesis from a built-in voice: streaming chunks, stats, waveform
 *   4. WAV export (captured to disk and validated)
 *   5. voice cloning: upload a reference clip, encode, then synthesize with it
 *
 * The dev server is expected on https://localhost:8600 (`pnpm dev`); certificate
 * errors are ignored because the cert is self-signed.
 *
 * The script leaves nothing running: the browser is killed on exit and the only
 * state it creates lives in /tmp/mockbird-e2e*.
 *
 * Usage:
 *   node scripts/e2e.mjs                 # reuse the cached profile (fast)
 *   node scripts/e2e.mjs --fresh         # delete the profile first (downloads ~763 MB)
 *   node scripts/e2e.mjs --port 8600
 */
import { execFileSync, spawn } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const fresh = args.includes("--fresh");
const port = Number(args[args.indexOf("--port") + 1] ?? 8600) || 8600;
// The dev server runs over HTTPS with a self-signed mkcert cert, so the headless
// browser must ignore certificate errors.
const base = `https://localhost:${port}`;
const PROFILE = "/tmp/mockbird-e2e-profile";
const DOWNLOADS = "/tmp/mockbird-e2e-downloads";
/**
 * Reference clip for the cloning test. Vendored from the MOSS-TTS-Nano
 * repository (Apache-2.0) so the check does not depend on a local clone; see
 * tests/fixtures/README.md. Override with MOCKBIRD_REFERENCE_WAV.
 */
const REFERENCE_WAV =
  process.env.MOCKBIRD_REFERENCE_WAV ?? resolve(root, "tests/fixtures/reference-zh.wav");
const CDP_PORT = 9401;

mkdirSync(DOWNLOADS, { recursive: true });
if (fresh) execFileSync("rm", ["-rf", PROFILE]);

const results = [];
const check = (name, ok, detail = "") => {
  results.push({ name, ok, detail });
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
};

const chrome = spawn(
  "chromium-browser",
  [
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--autoplay-policy=no-user-gesture-required",
    // The dev server uses a self-signed mkcert certificate.
    "--ignore-certificate-errors",
    `--remote-debugging-port=${CDP_PORT}`,
    `--user-data-dir=${PROFILE}`,
    "about:blank",
  ],
  { stdio: "ignore" },
);
const cleanup = () => {
  try {
    chrome.kill();
  } catch {
    /* already gone */
  }
};
process.on("exit", cleanup);
process.on("SIGINT", () => process.exit(130));

await sleep(3000);

const targets = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/list`)).json();
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
let id = 0;
const pending = new Map();
const pageErrors = [];
ws.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  if (message.id && pending.has(message.id)) {
    const cb = pending.get(message.id);
    pending.delete(message.id);
    cb(message);
    return;
  }
  if (message.method === "Runtime.exceptionThrown") {
    pageErrors.push(
      (
        message.params.exceptionDetails.exception?.description ??
        message.params.exceptionDetails.text ??
        ""
      ).slice(0, 300),
    );
  }
  if (message.method === "Log.entryAdded" && message.params.entry.level === "error") {
    pageErrors.push(message.params.entry.text.slice(0, 300));
  }
});
await new Promise((resolve) => ws.addEventListener("open", resolve));

const send = (method, params = {}, timeoutMs = 12000) =>
  new Promise((resolve) => {
    const n = ++id;
    pending.set(n, resolve);
    ws.send(JSON.stringify({ id: n, method, params }));
    setTimeout(() => {
      if (pending.has(n)) {
        pending.delete(n);
        resolve({ __timeout: true });
      }
    }, timeoutMs);
  });

const ev = async (expression) => {
  const response = await send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (response.__timeout) return "<<timeout>>";
  const result = response.result?.result;
  if (!result) return null;
  return typeof result.value === "string" ? result.value : JSON.stringify(result.value);
};

const navigate = async (route) => {
  await send("Page.navigate", { url: `${base}${route}` });
  await sleep(3500);
};

const click = (...labels) =>
  ev(`(() => {
    const wanted = ${JSON.stringify(labels.map((label) => label.toLowerCase()))};
    const buttons = [...document.querySelectorAll('button')];
    for (const label of wanted) {
      const button = buttons.find((node) => node.textContent.trim().toLowerCase().includes(label));
      if (button) {
        if (button.disabled) return 'disabled:' + button.textContent.trim().slice(0, 30);
        button.click();
        return 'clicked';
      }
    }
    return 'no-button';
  })()`);

const setNativeValue = (selector, value) =>
  ev(`(() => {
    const node = document.querySelector(${JSON.stringify(selector)});
    if (!node) return 'no-node';
    const setter = Object.getOwnPropertyDescriptor(
      node instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype,
      'value',
    ).set;
    setter.call(node, ${JSON.stringify(value)});
    node.dispatchEvent(new Event('input', { bubbles: true }));
    return 'ok';
  })()`);

const version = await (await fetch(`http://127.0.0.1:${CDP_PORT}/json/version`)).json();
const browserWs = new WebSocket(version.webSocketDebuggerUrl);
await new Promise((resolve) => browserWs.addEventListener("open", resolve));
let browserId = 0;
const browserSend = (method, params = {}) =>
  new Promise((resolve) => {
    const n = ++browserId;
    const handler = (event) => {
      const message = JSON.parse(event.data);
      if (message.id === n) {
        browserWs.removeEventListener("message", handler);
        resolve(message.result ?? message.error ?? null);
      }
    };
    browserWs.addEventListener("message", handler);
    browserWs.send(JSON.stringify({ id: n, method, params }));
    setTimeout(() => resolve(null), 5000);
  });
await browserSend("Browser.setDownloadBehavior", {
  behavior: "allowAndName",
  downloadPath: DOWNLOADS,
  eventsEnabled: true,
});

await send("Runtime.enable");
await send("Log.enable");
await send("Page.enable");

const phase = () =>
  ev(`document.querySelector('[data-engine-phase]')?.dataset.enginePhase ?? null`);

// ---------------------------------------------------------------------------
// 1. model download
// ---------------------------------------------------------------------------
console.log("\n── 1. model weights ─────────────────────────────────────────");
await navigate("/model");
check("cross-origin isolated", (await ev("crossOriginIsolated")) === "true");
check(
  "OPFS available",
  (await ev("typeof navigator.storage?.getDirectory === 'function'")) === "true",
);

const startPhase = await phase();
if (fresh) {
  await ev(`(() => {
    const button = [...document.querySelectorAll('button')].find((node) =>
      node.textContent.includes('Voice cloning encoder'));
    button?.click();
    return Boolean(button);
  })()`);
  await sleep(300);
}
if (startPhase !== "ready") {
  const clicked = await click("download", "resume");
  check("download started", clicked === "clicked", clicked);
  const startedAt = Date.now();
  let last = "";
  let percent = "";
  while (Date.now() - startedAt < 22 * 60 * 1000) {
    const current = await phase();
    percent = (await ev(`(document.body.innerText.match(/([\\d.]+)%/) ?? [''])[0]`)) || percent;
    if (current !== last) {
      console.log(
        `      [${((Date.now() - startedAt) / 1000) | 0}s] phase=${current} ${percent}`,
      );
      last = current ?? "";
    }
    if (current === "ready" || current === "error") break;
    await sleep(5000);
  }
  check("engine ready after download", (await phase()) === "ready");
} else {
  check("engine ready from OPFS cache", startPhase === "ready");
}

// ---------------------------------------------------------------------------
// 2. engine + voices
// ---------------------------------------------------------------------------
console.log("\n── 2. engine ───────────────────────────────────────────────");
await navigate("/settings");
const settings = await ev(`document.body.innerText`);
check(
  "engine phase ready",
  /Engine phase\s+ready/.test(settings),
  /Engine phase\s+(\w+)/.exec(settings)?.[1],
);
check(
  "ONNX Runtime version reported",
  /ONNX Runtime Web\s+[\d.]+/.test(settings),
  /ONNX Runtime Web\s+([^\s]+)/.exec(settings)?.[1] ?? "missing",
);
check("threads isolated", /Cross-origin isolated\s+yes/.test(settings));
const sessionLog = await ev(`(() => {
  const text = document.body.innerText;
  return text.includes('ORT session ready') && text.includes('warmed up');
})()`);
check("ORT sessions created + warmed up", sessionLog);

await navigate("/studio");
const voiceLabel = await ev(
  `document.querySelector('[role=combobox]')?.textContent?.trim() ?? ''`,
);
check(
  "a built-in voice is selected",
  voiceLabel.length > 0 && !voiceLabel.startsWith("Select"),
  voiceLabel,
);

// ---------------------------------------------------------------------------
// 3. synthesis (built-in voice)
// ---------------------------------------------------------------------------
console.log("\n── 3. synthesis ────────────────────────────────────────────");
await setNativeValue(
  "textarea",
  "Hello from Mockbird. This is an end to end test of local voice cloning.",
);
await sleep(1200);
const tokenCount = await ev(`(() => {
  const match = document.body.innerText.match(/(\\d+) tokens now/);
  return match ? Number(match[1]) : 0;
})()`);
check("token counter responds", tokenCount > 0, `${tokenCount} tokens`);

const startClick = await click("generate");
check("generate clicked", startClick === "clicked", startClick);

const synthStart = Date.now();
let synth = {};
while (Date.now() - synthStart < 6 * 60 * 1000) {
  synth = JSON.parse(
    (await ev(`JSON.stringify({
      status: document.querySelector('[data-samples]')?.dataset.status ?? 'unknown',
      samples: Number(document.querySelector('[data-samples]')?.dataset.samples ?? 0),
      firstAudio: (document.body.innerText.match(/first audio (\\d+) ms/) ?? [])[1] ?? null,
      error: ([...document.querySelectorAll('p')].map(n => n.textContent.trim())
        .find(t => /Error|failed|not loaded/i.test(t))) ?? null,
    })`)) ?? "{}",
  );
  if (synth.status === "done" || synth.status === "error") break;
  await sleep(2000);
}
check(
  "generation completed",
  synth.status === "done",
  `status=${synth.status} ${synth.error ?? ""}`,
);
check("PCM streamed to the player", synth.samples > 48000, `${synth.samples} samples`);
check(
  "first audio under 4 s",
  synth.firstAudio !== null && Number(synth.firstAudio) < 4000,
  `${synth.firstAudio} ms`,
);

const painted = Number(
  JSON.parse(
    (await ev(`(() => {
      const canvas = document.querySelector('canvas');
      if (!canvas) return JSON.stringify({ painted: 0 });
      const { data } = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
      let painted = 0;
      for (let i = 3; i < data.length; i += 4) if (data[i] > 8) painted += 1;
      return JSON.stringify({ painted });
    })()`)) ?? "{}",
  ).painted ?? 0,
);
check("waveform painted", painted > 500, `${painted} pixels`);

// ---------------------------------------------------------------------------
// 4. WAV export
// ---------------------------------------------------------------------------
console.log("\n── 4. WAV export ───────────────────────────────────────────");
const wavClick = await click("wav");
await sleep(6000);
let wavFile = "";
try {
  const list = execFileSync("bash", ["-c", `ls -t ${DOWNLOADS} | head -5`])
    .toString()
    .trim()
    .split("\n")
    .filter(Boolean);
  for (const name of list) {
    const path = `${DOWNLOADS}/${name}`;
    const head = readFileSync(path).subarray(0, 4).toString("latin1");
    if (head === "RIFF") {
      wavFile = path;
      break;
    }
  }
} catch {
  /* ignore */
}
check("WAV downloaded", wavClick === "clicked" && wavFile.length > 0, wavFile);
if (wavFile) {
  const size = statSync(wavFile).size;
  const buffer = readFileSync(wavFile);
  const riff = buffer.subarray(0, 4).toString("latin1");
  const wave = buffer.subarray(8, 12).toString("latin1");
  const sampleRate = buffer.readUInt32LE(24);
  const channels = buffer.readUInt16LE(22);
  check(
    "WAV header valid",
    riff === "RIFF" &&
      wave === "WAVE" &&
      sampleRate === 48000 &&
      channels === 2 &&
      size > 50_000,
    `${size} B, ${sampleRate} Hz, ${channels}ch`,
  );
}

// ---------------------------------------------------------------------------
// 5. voice cloning
// ---------------------------------------------------------------------------
console.log("\n── 5. voice cloning ────────────────────────────────────────");
if (!existsSync(REFERENCE_WAV)) {
  check("reference clip available", false, `missing ${REFERENCE_WAV}`);
  finish();
}
await navigate("/model");
const cloningReady = await ev(`document.body.innerText.includes('encoder missing')`);
if (cloningReady === "true") {
  // The optional encoder is not cached yet: download just that group.
  await ev(`(() => {
    const button = [...document.querySelectorAll('button')].find((node) =>
      node.textContent.includes('Voice cloning encoder'));
    button?.click();
    return Boolean(button);
  })()`);
  const encoderClick = await click("download", "resume");
  check("encoder download started", encoderClick === "clicked", encoderClick);
  const startedAt = Date.now();
  while (Date.now() - startedAt < 8 * 60 * 1000) {
    const state = await ev(`/Voice cloning\\s+ready/.test(document.body.innerText)`);
    if (state === "true") break;
    await sleep(4000);
  }
}
const cloningState = await ev(`(() => {
  const text = document.body.innerText;
  return /Voice clonings+ready/.test(text) ? 'ready'
    : /encoder missing/.test(text) ? 'missing' : 'unknown';
})()`);
check("voice cloning encoder ready", cloningState === "ready", cloningState);

await navigate("/voices");
const referenceBase64 = readFileSync(REFERENCE_WAV).toString("base64");
const uploaded = await ev(`(async () => {
  const bytes = Uint8Array.from(atob(${JSON.stringify(referenceBase64)}), (c) => c.charCodeAt(0));
  const file = new File([bytes], "reference.wav", { type: "audio/wav" });
  const dataTransfer = new DataTransfer();
  dataTransfer.items.add(file);
  const input = document.querySelector('input[type=file]');
  if (!input) return 'no-input';
  input.files = dataTransfer.files;
  input.dispatchEvent(new Event('change', { bubbles: true }));
  return 'uploaded';
})()`);
check("reference audio uploaded", uploaded === "uploaded", uploaded);

await sleep(6000);
const draftReady = await ev(`document.body.innerText.includes('48 kHz stereo')`);
check("reference decoded to 48 kHz PCM", draftReady === "true");

await setNativeValue("#voice-name", "E2E Test Voice");
await sleep(300);
const createClick = await click("create voice");
check("create voice clicked", createClick === "clicked", createClick);

const cloneStart = Date.now();
let voiceCreated = false;
while (Date.now() - cloneStart < 3 * 60 * 1000) {
  voiceCreated = (await ev(`document.body.innerText.includes('E2E Test Voice')`)) === "true";
  if (voiceCreated) break;
  await sleep(2500);
}
check("voice stored in OPFS", voiceCreated);

if (voiceCreated) {
  await navigate("/studio");
  const selected = await ev(`(() => {
    const select = document.querySelector('[role=combobox]');
    if (!select) return 'no-select';
    select.click();
    return 'opened';
  })()`);
  await sleep(800);
  const picked = await ev(`(() => {
    const option = [...document.querySelectorAll('[role=option], [role=menuitem]')]
      .find((node) => node.textContent.includes('E2E Test Voice'));
    if (!option) return 'no-option';
    option.click();
    return 'picked';
  })()`);
  check("cloned voice selectable in Studio", picked === "picked", `${selected}/${picked}`);
  await sleep(800);

  await setNativeValue(
    "textarea",
    "这是一段用自己录制的声音生成的语音。 This sentence is spoken by a cloned voice.",
  );
  await sleep(1200);
  await click("generate");
  const cloneSynthStart = Date.now();
  let cloneSynth = {};
  while (Date.now() - cloneSynthStart < 6 * 60 * 1000) {
    cloneSynth = JSON.parse(
      (await ev(`JSON.stringify({
        status: document.querySelector('[data-samples]')?.dataset.status ?? 'unknown',
        samples: Number(document.querySelector('[data-samples]')?.dataset.samples ?? 0),
        promptFrames: (document.body.innerText.match(/(\\d+) prompt frames/) ?? [])[1] ?? null,
        error: ([...document.querySelectorAll('p')].map(n => n.textContent.trim())
          .find(t => /Error|failed|not loaded/i.test(t))) ?? null,
      })`)) ?? "{}",
    );
    if (cloneSynth.status === "done" || cloneSynth.status === "error") break;
    await sleep(2000);
  }
  check(
    "synthesis with the cloned voice",
    cloneSynth.status === "done" && cloneSynth.samples > 48000,
    `status=${cloneSynth.status} samples=${cloneSynth.samples} promptFrames=${cloneSynth.promptFrames} ${cloneSynth.error ?? ""}`,
  );
  check(
    "prompt audio frames from the clone",
    Number(cloneSynth.promptFrames) > 0,
    `${cloneSynth.promptFrames} frames`,
  );
}

function finish() {
  console.log("\n── page errors ─────────────────────────────────────────────");
  console.log(pageErrors.length ? pageErrors.slice(0, 10).join("\n") : "(none)");
  check("no page errors", pageErrors.length === 0, `${pageErrors.length} errors`);

  const failed = results.filter((result) => !result.ok);
  console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
  mkdirSync("/tmp/mockbird-e2e", { recursive: true });
  writeFileSync(
    "/tmp/mockbird-e2e/report.json",
    JSON.stringify({ results, pageErrors: pageErrors.slice(0, 20) }, null, 2),
  );
  ws.close();
  cleanup();
  process.exit(failed.length === 0 ? 0 : 1);
}

// ---------------------------------------------------------------------------
// summary
// ---------------------------------------------------------------------------
finish();
