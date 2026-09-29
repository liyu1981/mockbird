"use client";

/**
 * Canvas waveform for both the reference clip preview and the live generation
 * view. Redraws whenever the buffer or the playhead changes; drawing is cheap
 * (one vertical line per pixel column) and stays off the React render path.
 */

import { useEffect, useRef } from "react";

/** Safety net so a pathological layout can never allocate an unbounded buffer. */
const MAX_EDGE = 4096;

type Props = {
  /** Raw channel-major PCM (used for reference clips). */
  channels?: Float32Array[];
  /** Pre-computed peak envelope (used for live generations). */
  peaks?: Float32Array;
  className?: string;
  /** 0..1 playhead position. */
  progress?: number;
  /** Draw as a muted "placeholder" (used while nothing is generated yet). */
  empty?: boolean;
  channelsCount?: number;
};

export function WaveformCanvas({
  channels,
  peaks,
  className,
  progress = 0,
  empty = false,
  channelsCount = 2,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const dataRef = useRef({ channels, peaks, progress, empty, channelsCount });

  useEffect(() => {
    dataRef.current = { channels, peaks, progress, empty, channelsCount };
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      // Measure the canvas's own box, never its parent's, and never write layout
      // styles back. Measuring the parent and assigning the result to
      // `style.height` created a ResizeObserver feedback loop wherever the canvas
      // sits in an auto-height container: panel grows -> canvas grows -> panel
      // grows, doubling the backing store every tick until the tab crashed.
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const width = Math.min(MAX_EDGE, Math.max(1, Math.round(rect.width)));
      const height = Math.min(MAX_EDGE, Math.max(1, Math.round(rect.height)));
      const backingWidth = Math.round(width * ratio);
      const backingHeight = Math.round(height * ratio);
      if (canvas.width === backingWidth && canvas.height === backingHeight) return;

      canvas.width = backingWidth;
      canvas.height = backingHeight;
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      paint(context, dataRef.current, width, height);
    };

    draw();
    // Observe the canvas itself: it is the element whose box actually matters.
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [channels, peaks, progress, empty, channelsCount]);

  return <canvas ref={canvasRef} className={className ?? "waveform-canvas"} aria-hidden />;
}

type PaintInput = {
  channels?: Float32Array[];
  peaks?: Float32Array;
  progress: number;
  empty: boolean;
  channelsCount: number;
};

function paint(
  context: CanvasRenderingContext2D,
  data: PaintInput,
  width: number,
  height: number,
) {
  const { progress, empty, channelsCount } = data;
  context.clearRect(0, 0, width, height);
  const styles = getComputedStyle(document.documentElement);
  const muted = styles.getPropertyValue("--muted-foreground").trim() || "#8b8b8b";
  const accent = styles.getPropertyValue("--primary").trim() || "#4f46e5";

  const middle = height / 2;
  const playedTo = Math.floor(width * Math.min(1, Math.max(0, progress)));

  // Envelope path: cheap and already aggregated.
  if (data.peaks && data.peaks.length > 0) {
    context.lineWidth = 1;
    for (let x = 0; x < width; x += 1) {
      const index = Math.floor((x / width) * data.peaks.length);
      const peak = data.peaks[Math.min(index, data.peaks.length - 1)] ?? 0;
      const amplitude = Math.max(0.75, peak * middle * 0.94);
      context.strokeStyle = x <= playedTo ? accent : muted;
      context.globalAlpha = x <= playedTo ? 1 : 0.55;
      context.beginPath();
      context.moveTo(x + 0.5, middle - amplitude);
      context.lineTo(x + 0.5, middle + amplitude);
      context.stroke();
    }
    context.globalAlpha = 1;
    return;
  }

  const source = data.channels?.[0];
  if (empty || !source || source.length === 0) {
    context.strokeStyle = muted;
    context.globalAlpha = 0.35;
    context.beginPath();
    context.moveTo(0, middle + 0.5);
    context.lineTo(width, middle + 0.5);
    context.stroke();
    context.globalAlpha = 1;
    return;
  }

  // Raw PCM path: one vertical line per pixel column.
  const left: Float32Array = source;
  const right: Float32Array = data.channels?.[1] ?? source;
  context.lineWidth = 1;
  for (let x = 0; x < width; x += 1) {
    const start = Math.floor((x / width) * left.length);
    const end = Math.max(start + 1, Math.floor(((x + 1) / width) * left.length));
    let peak = 0;
    for (let index = start; index < end && index < left.length; index += 1) {
      const sum =
        channelsCount > 1 ? (left[index] + (right[index] ?? left[index])) * 0.5 : left[index];
      const value = Math.abs(sum);
      if (value > peak) peak = value;
    }
    const amplitude = Math.max(0.75, peak * middle * 0.94);
    context.strokeStyle = x <= playedTo ? accent : muted;
    context.globalAlpha = x <= playedTo ? 1 : 0.55;
    context.beginPath();
    context.moveTo(x + 0.5, middle - amplitude);
    context.lineTo(x + 0.5, middle + amplitude);
    context.stroke();
  }
  context.globalAlpha = 1;
}
