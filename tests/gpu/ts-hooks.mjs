/**
 * Resolve hook so Node can import the engine's TypeScript modules directly.
 *
 * The app's source uses bundler-style extensionless imports (correct for
 * webpack/Turbopack), which Node's ESM resolver rejects. Rather than distort the
 * source, retry failed resolutions with the `.ts` / `/index.ts` variants.
 */
const CANDIDATES = ["", ".ts", "/index.ts", ".mjs"];

export async function resolve(specifier, context, nextResolve) {
  // bare specifiers are real packages — never rewrite them
  if (!specifier.startsWith(".") && !specifier.startsWith("/")) {
    return nextResolve(specifier, context);
  }
  let lastError;
  for (const suffix of CANDIDATES) {
    try {
      return await nextResolve(`${specifier}${suffix}`, context);
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError;
}
