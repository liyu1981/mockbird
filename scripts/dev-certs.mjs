/**
 * Generates project-local self-signed certificates with mkcert.
 *
 *   pnpm certs
 *
 * Why this exists: the app needs a *secure context* to work. `crossOriginIsolated`
 * (and therefore `SharedArrayBuffer`, and therefore multi-threaded ONNX Runtime
 * wasm) is only granted over HTTPS — or over http://localhost. As soon as you
 * test from another device on the LAN (phone, tablet, a second machine) the dev
 * server is not a secure context any more and the engine silently drops to a
 * single thread.
 *
 * Everything (CAROOT, the CA, the leaf certificate) lives in `mkcert/`, which is
 * git-ignored. `mkcert -install` needs root on Linux to add the CA to the system
 * trust store; when it is not available the script says exactly what to run by
 * hand and still produces usable certificates.
 */
import { execFileSync, execSync } from "node:child_process";
import { existsSync, mkdirSync, statSync } from "node:fs";
import { arch, networkInterfaces, platform } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "..");
export const CERT_DIR = join(root, "mkcert");
export const CERT_FILE = join(CERT_DIR, "localhost.pem");
export const KEY_FILE = join(CERT_DIR, "localhost-key.pem");
export const CA_FILE = join(CERT_DIR, "rootCA.pem");

/** Certificates are regenerated when older than this. */
const MAX_AGE_DAYS = 30;

function resolveMkcert() {
  // Prefer a project-local binary (mkcert/bin/mkcert) if someone vendored one,
  // otherwise whatever is on PATH.
  const local = join(CERT_DIR, "bin", "mkcert");
  if (existsSync(local)) return local;
  try {
    execSync("command -v mkcert", { stdio: ["ignore", "pipe", "ignore"] });
    return "mkcert";
  } catch {
    throw new Error(
      [
        "mkcert was not found.",
        "",
        "Install it with one of:",
        "  macOS:   brew install mkcert",
        "  Debian:  sudo apt install mkcert",
        "  Arch:    sudo pacman -S mkcert",
        "  or drop the binary at mkcert/bin/mkcert",
      ].join("\n"),
    );
  }
}

/**
 * Non-internal IPv4 addresses, ordered so the *likely* LAN address comes first
 * (docker bridges, WSL adapters and other virtual interfaces sort last).
 */
function lanAddresses() {
  const found = [];
  for (const [name, entries] of Object.entries(networkInterfaces())) {
    for (const entry of entries ?? []) {
      if (entry.family !== "IPv4" || entry.internal) continue;
      const virtual = /^(docker|br-|veth|virbr|vmnet|vboxnet|zt|tun|tap)/i.test(name);
      found.push({ address: entry.address, virtual });
    }
  }
  return found
    .sort((a, b) => Number(a.virtual) - Number(b.virtual))
    .map((item) => item.address);
}

/** Hosts requested via `--host <name-or-ip>` (repeatable). */
function extraHosts() {
  const hosts = [];
  const args = process.argv.slice(2);
  for (let i = 0; i < args.length; i += 1) {
    if (args[i] === "--host" && args[i + 1]) hosts.push(args[i + 1]);
  }
  return hosts;
}

function certIsFresh() {
  if (!existsSync(CERT_FILE) || !existsSync(KEY_FILE)) return false;
  const days = (Date.now() - statSync(CERT_FILE).mtimeMs) / 86_400_000;
  if (days > MAX_AGE_DAYS) return false;
  // Also honour the certificate's own expiry when we can read it.
  try {
    const endDate = execSync(`openssl x509 -enddate -noout -in ${JSON.stringify(CERT_FILE)}`, {
      stdio: ["ignore", "pipe", "ignore"],
    })
      .toString()
      .replace("notAfter=", "")
      .trim();
    if (endDate && new Date(endDate).getTime() - Date.now() < 7 * 86_400_000) return false;
  } catch {
    // openssl missing: the mtime check above is enough.
  }
  return true;
}

function runMkcert(args, env) {
  return execFileSync(resolveMkcert(), args, {
    env: { ...process.env, ...env },
    stdio: ["ignore", "pipe", "pipe"],
  }).toString();
}

/**
 * Ensures `mkcert/localhost.pem` + `mkcert/localhost-key.pem` exist and cover
 * localhost plus every LAN address. Returns the resolved paths.
 */
export function ensureCertificates({ force = false, quiet = false } = {}) {
  mkdirSync(CERT_DIR, { recursive: true });

  // Keep the CA inside the project so it can be git-ignored and thrown away.
  const env = { CAROOT: CERT_DIR };

  // Create/refresh the CA. `mkcert -install` writes rootCA.pem first and *then*
  // tries to add it to the system trust store, which needs root on Linux. A
  // failure there is not fatal: the certificates are still produced, the browser
  // just shows a warning until the CA is trusted.
  try {
    runMkcert(["-install"], env);
  } catch (error) {
    const detail = String(error?.stderr ?? error?.message ?? error).trim();
    if (!quiet) {
      console.warn(
        `[certs] CA not added to the system trust store (needs sudo): ${detail.split("\n").at(-1)}`,
      );
    }
  }
  if (!existsSync(CA_FILE)) {
    throw new Error(`mkcert did not create ${CA_FILE}`);
  }

  if (!force && certIsFresh()) {
    if (!quiet) console.log("[certs] certificates are up to date");
    return { cert: CERT_FILE, key: KEY_FILE, ca: CA_FILE, hosts: [] };
  }

  const hosts = [
    ...new Set(["localhost", "127.0.0.1", "::1", ...lanAddresses(), ...extraHosts()]),
  ];
  // CAROOT is an environment variable, not a flag (mkcert >= 1.4).
  runMkcert(["-cert-file", CERT_FILE, "-key-file", KEY_FILE, ...hosts], env);

  if (!quiet) {
    console.log(`[certs] wrote ${CERT_FILE}`);
    console.log(`[certs] wrote ${KEY_FILE}`);
    console.log(`[certs] valid for: ${hosts.join(", ")}`);
  }
  return { cert: CERT_FILE, key: KEY_FILE, ca: CA_FILE, hosts };
}

/** Prints how to trust the CA manually when `mkcert -install` was not possible. */
function printTrustHint() {
  console.log(
    [
      "",
      "[certs] The CA is at mkcert/rootCA.pem but is not in your system trust store.",
      "[certs] Browsers will warn on https://<lan-ip>:8600 until you trust it:",
      "",
      `  sudo cp ${CA_FILE} /usr/local/share/ca-certificates/mockbird-dev-ca.crt`,
      "  sudo update-ca-certificates",
      "",
      "[certs] Chromium can also skip the check for a quick test:",
      "  chromium --ignore-certificate-errors https://<lan-ip>:8600",
      "",
      `[certs] (${platform()}/${arch()})`,
    ].join("\n"),
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    const result = ensureCertificates({ force: process.argv.includes("--force") });
    console.log(
      [
        "",
        `[certs] done — dev server will use https on port 8600`,
        `[certs] local:   https://localhost:8600`,
        result.hosts.length > 3 ? `[certs] network: https://${result.hosts[3]}:8600` : "",
        "",
      ]
        .filter(Boolean)
        .join("\n"),
    );
    printTrustHint();
  } catch (error) {
    console.error(`[certs] failed: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
  }
}
