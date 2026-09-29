"use client";

import { useAtom, useAtomValue } from "jotai";
import {
  Activity,
  Ban,
  Download,
  Loader2,
  Pause,
  Play,
  RotateCcw,
  Sliders,
  Square,
  Trash2,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo } from "react";
import { toast } from "sonner";
import { useEngine } from "@/components/engine/EngineProvider";
import { WaveformCanvas } from "@/components/studio/WaveformCanvas";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { downloadBlob } from "@/lib/tts/audio";
import { engineAtom, enginePhaseAtom } from "@/state/modelAtoms";
import {
  advancedOpenAtom,
  generationAtom,
  historyAtom,
  realtimeFactorAtom,
  SAMPLE_TEXTS,
  synthParamsAtom,
  textAtom,
  textTokensAtom,
} from "@/state/studioAtoms";
import { allVoicesAtom, selectedVoiceAtom, type VoiceSelection } from "@/state/voiceAtoms";

const SAMPLE_RATE = 48_000;

export default function StudioPage() {
  const { generate, cancel, player, countTokens, ready } = useEngine();
  const [text, setText] = useAtom(textAtom);
  const tokens = useAtomValue(textTokensAtom);
  const [params, setParams] = useAtom(synthParamsAtom);
  const [advanced, setAdvanced] = useAtom(advancedOpenAtom);
  const [selected, setSelected] = useAtom(selectedVoiceAtom);
  const voices = useAtomValue(allVoicesAtom);
  const generation = useAtomValue(generationAtom);
  const engine = useAtomValue(engineAtom);
  const phase = useAtomValue(enginePhaseAtom);
  const rtf = useAtomValue(realtimeFactorAtom);
  const [history, setHistory] = useAtom(historyAtom);

  const busy = generation.status === "starting" || generation.status === "running";
  const stopping = generation.status === "stopping";
  const audioSeconds = generation.samples / SAMPLE_RATE;

  // Token count for the chunk-budget indicator.
  useEffect(() => {
    const handle = setTimeout(() => countTokens(text), 500);
    return () => clearTimeout(handle);
  }, [countTokens, text]);

  // Default to the first available voice once the engine reports them.
  useEffect(() => {
    if (!selected && voices.length > 0) setSelected(voices[0]);
  }, [selected, setSelected, voices]);

  // Snapshot finished generations into the local history.
  useEffect(() => {
    if (generation.status !== "done" || !selected) return;
    setHistory((prev) => {
      const entry = {
        id: `${Date.now()}`,
        createdAt: Date.now(),
        text,
        voiceLabel: selected.label,
        voiceKind: selected.kind,
        params,
        durationSec: generation.samples / SAMPLE_RATE,
        firstAudioMs: generation.stats?.firstAudioMs ?? null,
      };
      return [entry, ...prev].slice(0, 20);
    });
    // Only once per completed generation.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [generation.status, generation.requestId]);

  const handleGenerate = useCallback(() => {
    if (!selected) {
      toast.error("Pick a voice first.");
      return;
    }
    generate({ text, voice: selected });
  }, [generate, selected, text]);

  const handleDownload = useCallback(() => {
    const blob = player.toWav();
    if (!blob) {
      toast.error("Nothing generated yet.");
      return;
    }
    downloadBlob(
      blob,
      `mockbird-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-")}.wav`,
    );
  }, [player]);

  const grouped = useMemo(() => {
    const groups = new Map<string, VoiceSelection[]>();
    for (const voice of voices) {
      const list = groups.get(voice.group) ?? [];
      list.push(voice);
      groups.set(voice.group, list);
    }
    return [...groups.entries()];
  }, [voices]);

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Write something to say</CardTitle>
            <CardDescription>
              Long text is split automatically into {params.voiceCloneMaxTextTokens}-token
              chunks with a natural pause between them — {tokens} tokens now.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <Textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Type in any of the 20 supported languages…"
              rows={7}
              className="resize-y text-base"
            />
            <div className="flex flex-wrap items-center gap-2">
              {SAMPLE_TEXTS.map((sample) => (
                <Button
                  key={sample.id}
                  variant="outline"
                  size="sm"
                  onClick={() => setText(sample.text)}
                >
                  {sample.label}
                </Button>
              ))}
              {text && (
                <Button variant="ghost" size="sm" onClick={() => setText("")}>
                  <Trash2 className="size-3.5" /> Clear
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Activity className="size-4 text-muted-foreground" /> Output
            </CardTitle>
            <CardDescription>
              Audio streams while it is being generated, then the whole clip can be saved as 48
              kHz stereo WAV.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div
              className="h-24 w-full overflow-hidden rounded-lg border bg-muted/30"
              data-samples={generation.samples}
              data-status={generation.status}
            >
              <WaveformCanvas
                peaks={player.peaks}
                className="h-full w-full"
                empty={player.totalFrames === 0}
                progress={
                  player.totalFrames > 0 ? player.bufferedFrames / player.totalFrames : 0
                }
              />
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Button
                onClick={handleGenerate}
                disabled={busy || stopping || !ready || !selected}
              >
                {busy ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Play className="size-4" />
                )}
                {busy ? "Generating…" : "Generate"}
              </Button>

              {busy ? (
                <Button variant="outline" onClick={cancel} disabled={stopping}>
                  <Square className="size-4" /> {stopping ? "Stopping…" : "Stop"}
                </Button>
              ) : (
                <>
                  <Button
                    variant="outline"
                    onClick={() => (player.playing ? player.pause() : player.resume())}
                    disabled={player.totalFrames === 0}
                  >
                    {player.playing ? (
                      <Pause className="size-4" />
                    ) : (
                      <Play className="size-4" />
                    )}
                    {player.playing ? "Pause" : "Play"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleDownload}
                    disabled={player.totalFrames === 0}
                  >
                    <Download className="size-4" /> WAV
                  </Button>
                  <Button
                    variant="ghost"
                    onClick={handleGenerate}
                    disabled={!ready || !selected}
                    title="Generate again with a new random seed"
                  >
                    <RotateCcw className="size-4" /> Again
                  </Button>
                </>
              )}

              <span className="ml-auto font-mono text-xs text-muted-foreground">
                {audioSeconds.toFixed(1)}s
                {generation.stats?.rtf ? ` · wall ${generation.stats.rtf.toFixed(1)}s` : ""}
                {rtf ? ` · ${rtf.toFixed(2)}× realtime` : ""}
              </span>
            </div>

            {generation.stats && (
              <div className="flex flex-wrap gap-1.5 text-[11px]">
                <Badge variant="secondary">
                  first audio {generation.stats.firstAudioMs ?? "—"} ms
                </Badge>
                <Badge variant="secondary">{generation.stats.textChunks} chunks</Badge>
                <Badge variant="secondary">
                  {generation.stats.generatedFrames || "—"} frames
                </Badge>
                {generation.stats.promptAudioFrames > 0 && (
                  <Badge variant="secondary">
                    {generation.stats.promptAudioFrames} prompt frames
                  </Badge>
                )}
                {player.overflowed && <Badge variant="destructive">buffer overflowed</Badge>}
              </div>
            )}

            {generation.error && (
              <p className="rounded-lg border border-destructive/40 bg-destructive/5 p-2 text-xs text-destructive">
                {generation.error}
              </p>
            )}
            {player.error && (
              <p className="rounded-lg border border-amber-500/40 bg-amber-500/5 p-2 text-xs">
                {player.error}
              </p>
            )}
            {!ready && <NotReadyPanel filesCached={engine.files.length} phase={phase} />}
          </CardContent>
        </Card>
      </div>

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Voice</CardTitle>
            <CardDescription>
              Built-in voices need no cloning encoder. Clone your own from the Voices tab.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Select
              value={selected?.id ?? ""}
              onValueChange={(value) => {
                const voice = voices.find((item) => item.id === value);
                if (voice) setSelected(voice);
              }}
            >
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a voice" />
              </SelectTrigger>
              <SelectContent>
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
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Length</CardTitle>
            <CardDescription>
              {params.maxNewFrames} frames ≈ {(params.maxNewFrames / 12.5).toFixed(0)}s per
              chunk
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <ParamSlider
              label="Max frames per chunk"
              value={params.maxNewFrames}
              min={75}
              max={750}
              step={25}
              suffix=" frames"
              onChange={(value) => setParams({ ...params, maxNewFrames: value })}
            />
            <ParamSlider
              label="Tokens per chunk"
              value={params.voiceCloneMaxTextTokens}
              min={25}
              max={150}
              step={5}
              suffix=" tokens"
              onChange={(value) => setParams({ ...params, voiceCloneMaxTextTokens: value })}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              <button
                type="button"
                onClick={() => setAdvanced(!advanced)}
                className="flex w-full items-center justify-between text-left"
              >
                <span className="flex items-center gap-2">
                  <Sliders className="size-4 text-muted-foreground" /> Advanced
                </span>
                <Badge variant="outline">{advanced ? "hide" : "show"}</Badge>
              </button>
            </CardTitle>
            <CardDescription>
              Sampling defaults come from the model&apos;s own manifest (temperature 0.8, top-p
              0.95, top-k 25, repetition penalty 1.2).
            </CardDescription>
          </CardHeader>
          {advanced && (
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="greedy">Greedy (deterministic)</Label>
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
              <ParamSlider
                label="Audio temperature"
                value={params.audioTemperature}
                min={0.1}
                max={1.5}
                step={0.05}
                onChange={(value) => setParams({ ...params, audioTemperature: value })}
              />
              <ParamSlider
                label="Audio top-p"
                value={params.audioTopP}
                min={0.1}
                max={1}
                step={0.01}
                onChange={(value) => setParams({ ...params, audioTopP: value })}
              />
              <ParamSlider
                label="Audio top-k"
                value={params.audioTopK}
                min={1}
                max={100}
                step={1}
                onChange={(value) => setParams({ ...params, audioTopK: value })}
              />
              <ParamSlider
                label="Repetition penalty"
                value={params.audioRepetitionPenalty}
                min={1}
                max={2}
                step={0.05}
                onChange={(value) => setParams({ ...params, audioRepetitionPenalty: value })}
              />
              <Separator />
              <div className="flex items-center justify-between">
                <Label htmlFor="normalize">Normalize text</Label>
                <Switch
                  id="normalize"
                  checked={params.enableNormalizeTtsText}
                  onCheckedChange={(checked) =>
                    setParams({ ...params, enableNormalizeTtsText: checked })
                  }
                />
              </div>
              <p className="text-[11px] text-muted-foreground">
                WeTextProcessing is a native dependency and unavailable in the browser, so
                normalization uses the model&apos;s JS fallback (slightly rougher on numbers and
                symbols).
              </p>
            </CardContent>
          )}
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">History</CardTitle>
            <CardDescription>Last 20 generations, stored in this browser.</CardDescription>
          </CardHeader>
          <CardContent>
            {history.length === 0 ? (
              <p className="text-sm text-muted-foreground">Nothing yet.</p>
            ) : (
              <ScrollArea className="h-56">
                <ul className="space-y-2 pr-3">
                  {history.map((entry) => (
                    <li key={entry.id} className="rounded-lg border p-2">
                      <div className="flex items-center justify-between gap-2 text-[11px] text-muted-foreground">
                        <span className="truncate">{entry.voiceLabel}</span>
                        <span className="shrink-0 font-mono">
                          {entry.durationSec.toFixed(1)}s
                        </span>
                      </div>
                      <p className="mt-1 line-clamp-2 text-xs">{entry.text}</p>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="mt-1 h-7"
                        onClick={() => {
                          setText(entry.text);
                          const voice = voices.find(
                            (item) =>
                              item.kind === entry.voiceKind && item.label === entry.voiceLabel,
                          );
                          if (voice) setSelected(voice);
                        }}
                      >
                        Reuse
                      </Button>
                    </li>
                  ))}
                </ul>
              </ScrollArea>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function NotReadyPanel({ filesCached, phase }: { filesCached: number; phase: string }) {
  const copy =
    phase === "downloading"
      ? "Weights are downloading… this page will unlock as soon as they land."
      : phase === "loading"
        ? "Weights are in place. The engine is warming up the ONNX sessions…"
        : phase === "unsupported"
          ? "This browser can't cache the model (no OPFS). Use Chrome or Edge for the full experience."
          : filesCached > 0
            ? "Some weights are cached. Finish the download to unlock the studio."
            : "The 763 MB model hasn't been downloaded yet. It's a one-time download, cached in this browser forever.";
  return (
    <div className="flex flex-col items-start gap-3 rounded-lg border border-primary/40 bg-primary/5 p-4 sm:flex-row sm:items-center sm:justify-between">
      <p className="flex items-start gap-2 text-sm text-foreground/90">
        {phase === "downloading" || phase === "loading" ? (
          <Loader2 className="mt-0.5 size-4 shrink-0 animate-spin text-primary" />
        ) : (
          <Ban className="mt-0.5 size-4 shrink-0 text-primary" />
        )}
        {copy}
      </p>
      {(phase === "idle" || phase === "error") && (
        <Button size="sm" render={<Link href="/model" />}>
          <Download className="size-4" /> Get the model
        </Button>
      )}
    </div>
  );
}

function ParamSlider({
  label,
  value,
  min,
  max,
  step,
  suffix = "",
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix?: string;
  onChange: (value: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs">
        <Label>{label}</Label>
        <span className="font-mono text-muted-foreground">
          {value}
          {suffix}
        </span>
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
