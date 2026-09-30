"use client";

import { useAtom, useAtomValue } from "jotai";
import { Download, Loader2, Pause, Play, RotateCcw, Sparkles, Square } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo } from "react";
import { toast } from "sonner";
import { useEngine } from "@/components/engine/EngineProvider";
import { Details, Panel } from "@/components/layout/Panel";
import { WaveformCanvas } from "@/components/studio/WaveformCanvas";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { downloadBlob } from "@/lib/tts/audio";
import { describeChunks, estimateRunSeconds } from "@/lib/tts/chunking";
import { engineAtom, enginePhaseAtom } from "@/state/modelAtoms";
import {
  chunkProgressAtom,
  generationAtom,
  historyAtom,
  realtimeFactorAtom,
  SAMPLE_TEXTS,
  synthParamsAtom,
  textAtom,
  textChunksAtom,
  textTokensAtom,
} from "@/state/studioAtoms";
import { allVoicesAtom, selectedVoiceAtom } from "@/state/voiceAtoms";

const SAMPLE_RATE = 48_000;

export default function GeneratePage() {
  const { generate, cancel, player, analyzeText, ready } = useEngine();
  const [text, setText] = useAtom(textAtom);
  const tokens = useAtomValue(textTokensAtom);
  const chunks = useAtomValue(textChunksAtom);
  const [params, setParams] = useAtom(synthParamsAtom);
  const [selected, setSelected] = useAtom(selectedVoiceAtom);
  const voices = useAtomValue(allVoicesAtom);
  const generation = useAtomValue(generationAtom);
  const progress = useAtomValue(chunkProgressAtom);
  const engine = useAtomValue(engineAtom);
  const phase = useAtomValue(enginePhaseAtom);
  const rtf = useAtomValue(realtimeFactorAtom);
  const [history, setHistory] = useAtom(historyAtom);

  const busy = generation.status === "starting" || generation.status === "running";
  const audioSeconds = generation.samples / SAMPLE_RATE;
  const chunkTotal = generation.chunks.length;
  const currentChunk =
    generation.chunkIndex >= 0 ? generation.chunks[generation.chunkIndex] : undefined;

  useEffect(() => {
    const handle = window.setTimeout(() => analyzeText(text), 600);
    return () => window.clearTimeout(handle);
  }, [analyzeText, params.enableNormalizeTtsText, params.voiceCloneMaxTextTokens, text]);

  // Pick something sensible the first time voices arrive.
  useEffect(() => {
    if (!selected && voices.length > 0) {
      setSelected(voices.find((voice) => voice.kind === "cloned") ?? voices[0]);
    }
  }, [selected, setSelected, voices]);

  useEffect(() => {
    if (generation.status !== "done" || !selected) return;
    setHistory((previous) =>
      [
        {
          id: `${Date.now()}`,
          createdAt: Date.now(),
          text,
          voiceLabel: selected.label,
          voiceKind: selected.kind,
          params,
          durationSec: generation.samples / SAMPLE_RATE,
          firstAudioMs: generation.stats?.firstAudioMs ?? null,
        },
        ...previous,
      ].slice(0, 12),
    );
    // Only once per completed generation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generation.status, generation.requestId]);

  const handleGenerate = useCallback(() => {
    if (!selected) {
      toast.error("Choose a voice first.");
      return;
    }
    generate({ text, voice: selected });
  }, [generate, selected, text]);

  const handleDownload = useCallback(() => {
    const blob = player.toWav();
    if (!blob) {
      toast.error("Generate something first.");
      return;
    }
    downloadBlob(
      blob,
      `mockbird-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.wav`,
    );
  }, [player]);

  const grouped = useMemo(() => {
    const groups = new Map<string, typeof voices>();
    for (const voice of voices) {
      const list = groups.get(voice.group) ?? [];
      list.push(voice);
      groups.set(voice.group, list);
    }
    return [...groups.entries()];
  }, [voices]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      {!ready && (
        <div className="shrink-0">
          <SetupCallout phase={phase} hasWeights={engine.files.length > 0} />
        </div>
      )}

      <Panel
        title="Say something"
        description="Type in any of the 20 supported languages, pick a voice, and press Speak."
        className="shrink-0"
      >
        <div className="space-y-4">
          <Textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            placeholder="Type here… try “欢迎使用 Mockbird” or “Hello from Mockbird”"
            rows={6}
            className="resize-y text-base"
          />

          <div className="flex flex-wrap items-center gap-1.5">
            {SAMPLE_TEXTS.map((sample) => (
              <Button
                key={sample.id}
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={() => setText(sample.text)}
              >
                {sample.label}
              </Button>
            ))}
          </div>

          {chunks.length > 0 && (
            <p className="text-xs text-muted-foreground">
              {tokens > 0 && <span className="mr-2">{tokens} tokens</span>}
              {describeChunks(chunks)}
              {chunks.length > 1 && (
                <span className="ml-2">· spoken one chunk after another</span>
              )}
            </p>
          )}

          {chunks.length > 1 && estimateRunSeconds(chunks) > 240 && (
            <p className="text-xs text-muted-foreground">
              Long passages are kept in memory until you save or regenerate (~380&nbsp;MB of PCM
              per 10 minutes).
            </p>
          )}

          <div className="grid gap-4 sm:grid-cols-2">
            <div className="space-y-1.5">
              <Label>Voice</Label>
              <Select
                value={selected?.id ?? ""}
                onValueChange={(value) => {
                  const voice = voices.find((item) => item.id === value);
                  if (voice) setSelected(voice);
                }}
              >
                <SelectTrigger className="w-full" aria-label="Voice">
                  <SelectValue placeholder="Choose a voice" />
                </SelectTrigger>
                <SelectContent className="max-h-72">
                  {grouped.map(([group, items]) => (
                    <SelectGroup key={group}>
                      <SelectLabel>{group}</SelectLabel>
                      {items.map((voice) => (
                        <SelectItem key={`${voice.kind}-${voice.id}`} value={voice.id}>
                          {voice.label}
                        </SelectItem>
                      ))}
                    </SelectGroup>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                Want your own voice?{" "}
                <Link href="/clone" className="underline underline-offset-2">
                  Clone it in the Clone tab
                </Link>
                .
              </p>
            </div>

            <div className="space-y-1.5">
              <Label>Length of each chunk</Label>
              <div className="flex h-9 items-center gap-3">
                <Slider
                  value={[params.maxNewFrames]}
                  min={75}
                  max={750}
                  step={25}
                  onValueChange={(value) =>
                    setParams({
                      ...params,
                      maxNewFrames: typeof value === "number" ? value : (value[0] ?? 375),
                    })
                  }
                />
                <span className="w-20 shrink-0 text-right text-xs text-muted-foreground">
                  ~{(params.maxNewFrames / 12.5).toFixed(0)}s
                </span>
              </div>
              <p className="text-xs text-muted-foreground">
                Long text is split on sentence boundaries, spoken chunk by chunk, and saved as
                one clip.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="lg"
              onClick={handleGenerate}
              disabled={busy || !ready || !selected}
              className="min-w-36"
            >
              {busy ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Sparkles className="size-4" />
              )}
              {busy ? "Working…" : "Speak"}
            </Button>

            {busy ? (
              <Button variant="outline" size="lg" onClick={cancel}>
                <Square className="size-4" /> Stop
              </Button>
            ) : (
              <>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() => (player.playing ? player.pause() : player.resume())}
                  disabled={player.totalFrames === 0}
                >
                  {player.playing ? <Pause className="size-4" /> : <Play className="size-4" />}
                  {player.playing ? "Pause" : "Replay"}
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={handleDownload}
                  disabled={player.totalFrames === 0}
                >
                  <Download className="size-4" /> Save audio
                </Button>
                <Button
                  variant="ghost"
                  size="lg"
                  onClick={handleGenerate}
                  disabled={!ready || !selected}
                  title="Say it again with a fresh random voice"
                >
                  <RotateCcw className="size-4" />
                </Button>
              </>
            )}

            <span className="ml-auto text-xs text-muted-foreground">
              {tokens > 0 && !chunks.length && <span className="mr-3">{tokens} tokens</span>}
              {audioSeconds > 0.05 && (
                <span className="mr-3">
                  {audioSeconds.toFixed(1)}s{rtf ? ` · ${rtf.toFixed(1)}× realtime` : ""}
                </span>
              )}
            </span>
          </div>

          {generation.error && (
            <p className="rounded-lg border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
              {generation.error}
            </p>
          )}

          {chunkTotal > 0 && (busy || generation.status === "stopping") && (
            <ChunkProgress
              done={generation.chunksDone}
              total={chunkTotal}
              progress={progress ?? 0}
              current={currentChunk}
            />
          )}

          {/* ---- the glass player ---- */}
          <div className="h-28 w-full overflow-hidden rounded-xl border border-border/60 bg-background/40">
            <WaveformCanvas
              peaks={player.peaks}
              className="h-full w-full"
              empty={player.totalFrames === 0}
              progress={player.totalFrames > 0 ? player.bufferedFrames / player.totalFrames : 0}
            />
          </div>

          {generation.stats && (
            <div className="flex flex-wrap gap-1.5">
              {generation.stats.firstAudioMs !== null && (
                <Badge variant="secondary">
                  started in {(generation.stats.firstAudioMs / 1000).toFixed(1)}s
                </Badge>
              )}
              {player.overflowed && <Badge variant="destructive">playback overflowed</Badge>}
            </div>
          )}
        </div>
      </Panel>

      <div className="shrink-0">
        <Details
          summary="More options (usually not needed)"
          description="Sampling, chunking and text normalisation. The defaults come from the model itself."
        >
          <ParamSlider
            label="Words per chunk"
            value={params.voiceCloneMaxTextTokens}
            min={25}
            max={150}
            step={5}
            onChange={(value) => setParams({ ...params, voiceCloneMaxTextTokens: value })}
          />
          <ParamSlider
            label="Variation"
            hint="Lower is more consistent, higher is more expressive"
            value={params.audioTemperature}
            min={0.1}
            max={1.5}
            step={0.05}
            onChange={(value) => setParams({ ...params, audioTemperature: value })}
          />
          <ParamSlider
            label="Repetition penalty"
            value={params.audioRepetitionPenalty}
            min={1}
            max={2}
            step={0.05}
            onChange={(value) => setParams({ ...params, audioRepetitionPenalty: value })}
          />
          <div className="flex items-center justify-between">
            <Label htmlFor="greedy">Always say it the same way</Label>
            <Switch
              id="greedy"
              checked={params.sampleMode === "greedy"}
              onCheckedChange={(checked) =>
                setParams({
                  ...params,
                  sampleMode: checked ? "greedy" : "fixed",
                  doSample: !checked,
                })
              }
            />
          </div>
          <div className="flex items-center justify-between">
            <Label htmlFor="normalize">Tidy up the text before speaking</Label>
            <Switch
              id="normalize"
              checked={params.enableNormalizeTtsText}
              onCheckedChange={(checked) =>
                setParams({ ...params, enableNormalizeTtsText: checked })
              }
            />
          </div>
        </Details>
      </div>

      {history.length > 0 && (
        <Panel
          grow
          title="Recent"
          description="Tap a line to load its text and voice again."
          actions={
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setHistory([])}
              className="text-xs"
            >
              Clear
            </Button>
          }
        >
          <div className="min-h-0 flex-1 overflow-y-auto overscroll-contain pr-1">
            <ul className="space-y-1.5">
              {history.map((entry) => (
                <li key={entry.id}>
                  <button
                    type="button"
                    onClick={() => {
                      setText(entry.text);
                      const voice = voices.find(
                        (item) =>
                          item.kind === entry.voiceKind && item.label === entry.voiceLabel,
                      );
                      if (voice) setSelected(voice);
                    }}
                    className="w-full rounded-lg px-3 py-2 text-left transition-colors hover:bg-foreground/5"
                  >
                    <span className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                      <span className="truncate">{entry.voiceLabel}</span>
                      <span className="shrink-0 font-mono">
                        {entry.durationSec.toFixed(1)}s
                      </span>
                    </span>
                    <span className="mt-0.5 line-clamp-2 block text-sm">{entry.text}</span>
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </Panel>
      )}
    </div>
  );
}

function ChunkProgress({
  done,
  total,
  progress,
  current,
}: {
  done: number;
  total: number;
  progress: number;
  current?: string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-3 text-xs text-muted-foreground">
        <span>
          Chunk {Math.min(done + 1, total)} of {total}
        </span>
        <span className="font-mono">{Math.round(progress * 100)}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-foreground/10">
        <div
          className="h-full rounded-full bg-primary transition-[width] duration-300"
          style={{ width: `${Math.max(2, Math.min(100, progress * 100))}%` }}
        />
      </div>
      {current && <p className="line-clamp-2 text-xs text-muted-foreground">{current}</p>}
    </div>
  );
}

function SetupCallout({ phase, hasWeights }: { phase: string; hasWeights: boolean }) {
  const busy = phase === "downloading" || phase === "loading";
  return (
    <div className="glass-control rounded-xl p-5">
      <div className="flex flex-wrap items-center gap-4">
        <div className="min-w-0 flex-1">
          <p className="font-medium">
            {busy
              ? phase === "downloading"
                ? "Downloading the model…"
                : "Starting the engine…"
              : "One-time setup"}
          </p>
          <p className="mt-1 text-sm text-muted-foreground">
            {busy
              ? "This happens once. It is stored in your browser, so next time you open Mockbird it starts instantly."
              : "Mockbird needs its speech model before it can make any sound. It is a one-off download that stays in this browser."}
          </p>
        </div>
        {/* `render` swaps in a Next <Link> (an <a>), so the Base UI button must
            drop its native-button semantics instead of warning about them. */}
        <Button size="lg" nativeButton={false} render={<Link href="/settings" />}>
          {busy ? <Loader2 className="size-4 animate-spin" /> : <Download className="size-4" />}
          {hasWeights ? "Finish setup" : "Get started"}
        </Button>
      </div>
    </div>
  );
}

function ParamSlider({
  label,
  hint,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <div>
          <Label>{label}</Label>
          {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
        </div>
        <span className="shrink-0 font-mono text-xs text-muted-foreground">{value}</span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(next) => onChange(typeof next === "number" ? next : (next[0] ?? min))}
      />
    </div>
  );
}
