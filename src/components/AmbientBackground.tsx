/**
 * Animated ambient blob field behind the whole app.
 *
 * Copied from the Shoufa project (~/shoufa) — it gives the frosted-glass surfaces
 * something to blur. Respects prefers-reduced-motion and prefers-reduced-
 * transparency, and re-inits when the theme class flips.
 */
"use client";

import { useEffect, useRef } from "react";

interface Blob {
  x: number;
  y: number;
  radius: number;
  hue: number;
  vx: number;
  vy: number;
  phase: number;
}

const HUES = [200, 260, 320, 170, 30, 355];
const LIGHT_PALETTE = { sat: 95, light: 74, alpha: 0.34, count: 12 };
const DARK_PALETTE = { sat: 95, light: 56, alpha: 0.32, count: 12 };

function paletteFor(isDark: boolean, reducedTransparency: boolean) {
  const base = isDark ? DARK_PALETTE : LIGHT_PALETTE;
  return {
    ...base,
    alpha: reducedTransparency ? base.alpha * 0.55 : base.alpha,
  };
}

function createBlob(w: number, h: number): Blob {
  const minDim = Math.min(w, h);
  return {
    x: Math.random() * w,
    y: Math.random() * h,
    radius: (0.07 + Math.random() * 0.18) * minDim,
    hue: HUES[Math.floor(Math.random() * HUES.length)] + (Math.random() - 0.5) * 24,
    vx: (Math.random() - 0.5) * 0.3,
    vy: (Math.random() - 0.5) * 0.3,
    phase: Math.random() * Math.PI * 2,
  };
}

function isDarkMode(): boolean {
  return document.documentElement.classList.contains("dark");
}

function prefersReducedTransparency(): boolean {
  return window.matchMedia("(prefers-reduced-transparency: reduce)").matches;
}

function prefersReducedMotion(): boolean {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export default function AmbientBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animFrame: number;
    let blobs: Blob[] = [];
    let lastTime = performance.now();

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const initBlobs = () => {
      const palette = paletteFor(isDarkMode(), prefersReducedTransparency());
      blobs = Array.from({ length: palette.count }, () =>
        createBlob(window.innerWidth, window.innerHeight),
      );
    };

    const wrap = (val: number, max: number, margin: number): number => {
      if (val < -margin) return max + margin;
      if (val > max + margin) return -margin;
      return val;
    };

    const render = (time: number) => {
      const dt = (time - lastTime) / 1000;
      lastTime = time;
      const w = window.innerWidth;
      const h = window.innerHeight;
      const palette = paletteFor(isDarkMode(), prefersReducedTransparency());

      ctx.clearRect(0, 0, w, h);
      ctx.globalCompositeOperation = "lighter";

      for (const blob of blobs) {
        // Sine wobble + drift
        blob.x += blob.vx + Math.sin(blob.phase + time * 0.0005) * 0.2;
        blob.y += blob.vy + Math.cos(blob.phase + time * 0.0007) * 0.15;
        blob.hue += 1.2 * dt; // hue drift

        // Wrap around viewport
        blob.x = wrap(blob.x, w, 200);
        blob.y = wrap(blob.y, h, 200);

        const gradient = ctx.createRadialGradient(
          blob.x,
          blob.y,
          0,
          blob.x,
          blob.y,
          blob.radius,
        );
        gradient.addColorStop(
          0,
          `hsla(${blob.hue}, ${palette.sat}%, ${palette.light}%, ${palette.alpha})`,
        );
        gradient.addColorStop(
          0.6,
          `hsla(${blob.hue}, ${palette.sat}%, ${palette.light}%, ${palette.alpha * 0.4})`,
        );
        gradient.addColorStop(1, `hsla(${blob.hue}, ${palette.sat}%, ${palette.light}%, 0)`);

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, w, h);
      }

      ctx.globalCompositeOperation = "source-over";

      if (!prefersReducedMotion()) {
        animFrame = requestAnimationFrame(render);
      }
    };

    // Initialize
    resize();
    initBlobs();

    if (prefersReducedMotion()) {
      // Render one static frame
      animFrame = requestAnimationFrame(render);
    } else {
      animFrame = requestAnimationFrame(render);
    }

    // Listeners
    const onResize = () => {
      resize();
      initBlobs();
    };
    window.addEventListener("resize", onResize);

    // React to dark mode toggle
    const observer = new MutationObserver(() => {
      initBlobs();
    });
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => {
      cancelAnimationFrame(animFrame);
      window.removeEventListener("resize", onResize);
      observer.disconnect();
    };
  }, []);

  return <canvas ref={canvasRef} className="pointer-events-none fixed inset-0 z-0" />;
}
