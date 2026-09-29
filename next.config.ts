import { networkInterfaces } from "node:os";

import type { NextConfig } from "next";

/**
 * Mockbird — cross-origin isolation is mandatory for the MOSS-TTS-Nano browser
 * runtime: onnxruntime-web needs SharedArrayBuffer, which browsers only expose
 * when the document is cross-origin isolated.
 *
 * This is the single source of truth: it applies in `next dev` and on Vercel.
 * (Static export is deliberately not used, because `output: "export"` does not
 * support `headers()` — see docs/adr/0002-static-export-vs-vercel.md.)
 */
const crossOriginIsolationHeaders = [
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
];

/**
 * Next 16 blocks cross-origin access to dev resources (the HMR websocket,
 * chunk loading) unless the requesting host is allow-listed. When the dev server
 * is opened from another device on the LAN — which is exactly why it runs over
 * HTTPS — those requests come from the machine's own IP and would be blocked,
 * leaving a page that loads but never hydrates.
 */
function localDevOrigins(): string[] {
  const origins = new Set<string>(["localhost", "127.0.0.1", "[::1]"]);
  for (const entries of Object.values(networkInterfaces())) {
    for (const entry of entries ?? []) {
      if (entry.family === "IPv4" && !entry.internal) origins.add(entry.address);
    }
  }
  return [...origins];
}

const nextConfig: NextConfig = {
  reactStrictMode: true,
  typedRoutes: true,
  allowedDevOrigins: localDevOrigins(),
  async headers() {
    return [
      {
        source: "/:path*",
        headers: crossOriginIsolationHeaders,
      },
      {
        // The ONNX Runtime wasm binary is fetched by the worker with
        // `credentials: "omit"`; keep the CORP header explicit for safety.
        source: "/ort/:path*",
        headers: [
          ...crossOriginIsolationHeaders,
          { key: "Cross-Origin-Resource-Policy", value: "same-origin" },
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
    ];
  },
};

export default nextConfig;
