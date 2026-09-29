# ADR 0002 — Deploy as a normal Next.js build, not a static export

**Status:** accepted

## Context

The app has no server: every route is a client component with no data fetching, no
Server Actions and no API routes. `output: "export"` would therefore produce a
pure static bundle — attractive for a CDN-only deployment.

The blocker is cross-origin isolation. ONNX Runtime Web's multi-threaded wasm
backend needs `SharedArrayBuffer`, which browsers only expose to documents served
with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Next.js documents `headers()` as **unsupported** with `output: "export"`.

## Options

**A. Static export + `vercel.json` headers (rejected).**
Works, but the header configuration then exists twice: `vercel.json` for
production and `next.config.ts` for `next dev` (a static-export build errors if
`headers()` is present, so dev would need a separate proxy). Two sources of truth
for a setting the app cannot work without is exactly the failure mode we want to
avoid.

**B. Normal Next.js build on Vercel (adopted).**
Every route prerenders to static HTML at build time and Vercel serves it from the
edge; nothing runs per-request, so the cost profile matches a static export. A
single `headers()` block in `next.config.ts` covers dev and production.

**C. `output: "export"` with no isolation headers (rejected).**
Single-threaded wasm: roughly the speed difference between a smooth demo and a
frustrating one, for a benefit nobody asked for.

## Consequences

- `next.config.ts` is the single source of truth for headers; dev and prod behave
  identically.
- If a pure-CDN host is ever required, switching is a config-only change: set
  `output: "export"`, move the two header entries to the host's config
  (`vercel.json` for Vercel, `_headers` for Netlify/Cloudflare Pages), and give
  `next dev` a header-injecting proxy. No application code changes.
- Everything in `src/` is already client-side, so that migration path stays open.
