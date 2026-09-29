/**
 * `pnpm dev` — starts the Next.js dev server on **https://localhost:8600**.
 *
 * HTTPS is not cosmetic here. `crossOriginIsolated` is what unlocks
 * `SharedArrayBuffer`, and ONNX Runtime Web needs it for multi-threaded wasm.
 * Browsers only grant a secure context to https:// (or http://localhost), so the
 * moment you open the dev server from another device on the LAN the engine
 * quietly falls back to a single thread.
 *
 * Certificates are generated on demand by scripts/dev-certs.mjs (mkcert, CA kept
 * in the git-ignored `mkcert/` folder). Use `pnpm dev:http` for a plain-HTTP run.
 */
import { spawn } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { ensureCertificates } from "./dev-certs.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
const port = process.env.PORT ?? "8600";

let certs = null;
let certError = null;
try {
  certs = ensureCertificates();
} catch (error) {
  certError = error;
}

if (!certs) {
  console.error(
    `\n[certs] ${certError instanceof Error ? certError.message : String(certError)}\n`,
  );
  console.error("[dev] falling back to http://localhost — that only works on this machine.\n");
  const fallback = spawn(
    process.platform === "win32" ? "npx.cmd" : "npx",
    ["next", "dev", "--webpack", "-p", port],
    { cwd: root, stdio: "inherit" },
  );
  fallback.on("exit", (code) => process.exit(code ?? 0));
} else {
  const args = [
    "next",
    "dev",
    // webpack, not Turbopack: Turbopack's dev server exceeded our RAM budget.
    // See docs/adr/0001-bundler-parity.md.
    "--webpack",
    "-p",
    port,
    "--experimental-https",
    "--experimental-https-cert",
    certs.cert,
    "--experimental-https-key",
    certs.key,
  ];
  if (certs.ca) {
    args.push("--experimental-https-ca", certs.ca);
  }

  const dev = spawn(process.platform === "win32" ? "npx.cmd" : "npx", args, {
    cwd: root,
    stdio: "inherit",
  });
  dev.on("exit", (code) => process.exit(code ?? 0));
}
