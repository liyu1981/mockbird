"use client";

/**
 * A tiny `atomWithStorage` replacement.
 *
 * We deliberately avoid `jotai/utils` here: with jotai v3 (ESM-only,
 * module-first) importing that subpath in a Next dev build pulls in a second
 * copy of the package and triggers
 * "Detected multiple Jotai instances. It may cause unexpected behavior with the
 * default store." This local version keeps a single instance and gives us the
 * semantics Mockbird needs:
 *
 *  - the initial value is whatever the atom was created with (SSR-safe),
 *  - `hydrateAtomsFromStorage()` adopts persisted values once, after mount,
 *  - writes are tolerant of storage failures (private mode, quota, disabled).
 */

import { atom, getDefaultStore, type PrimitiveAtom } from "jotai";

type StorageLike = Pick<Storage, "getItem" | "setItem" | "removeItem">;

function getStorage(): StorageLike | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

export function atomWithStorage<T>(
  key: string,
  initialValue: T,
  options?: {
    serialize?: (value: T) => string;
    reviver?: (raw: string) => T | null;
  },
): PrimitiveAtom<T> {
  const serialize = options?.serialize ?? ((value: T) => JSON.stringify(value));
  const baseAtom = atom<T>(initialValue);

  return atom(
    (get) => get(baseAtom),
    (get, set, update: T | ((previous: T) => T)) => {
      const next =
        typeof update === "function" ? (update as (previous: T) => T)(get(baseAtom)) : update;
      set(baseAtom, next);
      const storage = getStorage();
      if (!storage) return;
      try {
        storage.setItem(key, serialize(next));
      } catch {
        // Storage may be unavailable or full; preferences are best-effort.
      }
    },
  );
}

/**
 * The atom's value type is intentionally open here: a hydration entry is keyed by
 * storage key, and callers build the list inline, so pinning the generic would
 * only force casts at every call site.
 */
type HydrationEntry = {
  // biome-ignore lint/suspicious/noExplicitAny: open by design, see above
  atom: PrimitiveAtom<any>;
  key: string;
  // biome-ignore lint/suspicious/noExplicitAny: open by design, see above
  reviver?: (raw: string) => any;
};

/** Applies persisted values to atoms. Call once, on the client, after mount. */
export function hydrateAtomsFromStorage(entries: HydrationEntry[]): void {
  if (typeof window === "undefined") return;
  const storage = getStorage();
  if (!storage) return;
  const store = getDefaultStore();
  for (const entry of entries) {
    let raw: string | null = null;
    try {
      raw = storage.getItem(entry.key);
    } catch {
      continue;
    }
    if (raw === null) continue;
    try {
      const value = JSON.parse(raw) as unknown;
      if (value !== null && value !== undefined) {
        store.set(entry.atom, value);
      }
    } catch {
      // Corrupt payload: keep the default value.
    }
  }
}
/* eslint-enable @typescript-eslint/no-explicit-any */

export function clearPersistedAtom(key: string): void {
  getStorage()?.removeItem(key);
}
