"use client";

import { useAtom, useAtomValue } from "jotai";
import {
  AudioWaveform,
  Download,
  Loader2,
  Pencil,
  Plus,
  Trash2,
  Upload,
  Users,
} from "lucide-react";
import { useCallback, useRef, useState } from "react";
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ReadingScriptCard } from "@/components/voice/ReadingScriptCard";
import { type RecordingResult, VoiceRecorder } from "@/components/voice/VoiceRecorder";
import { analyzePcm, decodeToModelPcm, downloadBlob, encodeWav } from "@/lib/tts/audio";
import { exportVoiceFile } from "@/lib/voice/io";
import { engineAtom } from "@/state/modelAtoms";
import {
  allVoicesAtom,
  builtinVoicesAtom,
  clonedVoicesAtom,
  cloningBusyAtom,
  cloningErrorAtom,
  draftAnalysisAtom,
  draftNameAtom,
  draftPcmAtom,
  draftSourceAtom,
} from "@/state/voiceAtoms";

export default function VoicesPage() {
  const { client, downloadModels } = useEngine();
  const engine = useAtomValue(engineAtom);
  const cloned = useAtomValue(clonedVoicesAtom);
  const builtin = useAtomValue(builtinVoicesAtom);
  const all = useAtomValue(allVoicesAtom);
  const [draftName, setDraftName] = useAtom(draftNameAtom);
  const [draftPcm, setDraftPcm] = useAtom(draftPcmAtom);
  const [draftSource, setDraftSource] = useAtom(draftSourceAtom);
  const analysis = useAtomValue(draftAnalysisAtom);
  const [busy, setBusy] = useAtom(cloningBusyAtom);
  const [cloningError, setCloningError] = useAtom(cloningErrorAtom);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; name: string } | null>(null);
  const [recording, setRecording] = useState(false);
  const [importing, setImporting] = useState(false);

  const acceptCapture = useCallback(
    (result: RecordingResult, source: "recording" | "upload") => {
      setDraftPcm({
        channels: result.channels,
        sampleRate: result.sampleRate,
        durationSec: result.durationSec,
      });
      setDraftSource(source);
      if (!draftName.trim()) {
        setDraftName(`Voice ${new Date().toLocaleDateString()}`);
      }
    },
    [draftName, setDraftName, setDraftPcm, setDraftSource],
  );

  const handleFile = useCallback(
    async (file: File) => {
      setCloningError(null);
      try {
        const pcm = await decodeToModelPcm(await file.arrayBuffer());
        acceptCapture({ ...pcm, analysis: analyzePcm(pcm.channels), blob: file }, "upload");
      } catch (error) {
        setCloningError(error instanceof Error ? error.message : String(error));
      }
    },
    [acceptCapture, setCloningError],
  );

  const createVoice = useCallback(async () => {
    if (!client || !draftPcm) return;
    setBusy(true);
    setCloningError(null);
    try {
      await client.createVoice(
        draftName.trim() || "Untitled voice",
        draftSource ?? "upload",
        draftPcm.channels,
        draftPcm.sampleRate,
      );
      setDraftPcm(null);
      setDraftSource(null);
      setDraftName("");
    } catch (error) {
      setCloningError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }, [
    client,
    draftName,
    draftPcm,
    draftSource,
    setBusy,
    setCloningError,
    setDraftName,
    setDraftPcm,
    setDraftSource,
  ]);

  const exportVoice = useCallback(async (id: string, name: string) => {
    try {
      const file = await exportVoiceFile(id);
      downloadBlob(
        new Blob([JSON.stringify(file)], { type: "application/json" }),
        `${name.replace(/\s+/g, "-").toLowerCase() || "voice"}.mockbirdvoice.json`,
      );
      toast.success("Voice exported.");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  }, []);

  const cloningMissing = !engine.cloningReady;

  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h1 className="text-2xl font-semibold tracking-tight">Voices</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Clone a voice from {IDEAL_RANGE} of clean, single-speaker audio, or start with one of
          the {builtin.length} built-in voices. Everything is encoded on-device; the recording
          is never uploaded.
        </p>
      </header>

      {cloningMissing && (
        <Card className="border-amber-500/40 bg-amber-500/5">
          <CardHeader>
            <CardTitle className="text-base">Voice cloning encoder not downloaded</CardTitle>
            <CardDescription>
              The optional 45 MB encoder turns your recording into audio codes. Grab just that
              part, or keep using the built-in voices below.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              size="sm"
              onClick={() => downloadModels(["codec-encode"])}
              disabled={engine.files.length === 0}
            >
              {engine.files.length === 0 ? (
                "Download the full model first"
              ) : (
                <>
                  <Download className="size-4" /> Get the cloning encoder (45 MB)
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
        <div className="space-y-4">
          <ReadingScriptCard recording={recording} />
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Clone a new voice</CardTitle>
            <CardDescription>
              Record or upload a sample. For the best likeness: one speaker, no music, no
              reverb, no background noise.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <VoiceRecorder
              onCaptured={(result) => acceptCapture(result, "recording")}
              onRecordingChange={setRecording}
              disabled={cloningMissing || busy}
            />

            <div className="flex items-center gap-2">
              <span className="h-px flex-1 bg-border" />
              <span className="text-[11px] text-muted-foreground">or</span>
              <span className="h-px flex-1 bg-border" />
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*,video/*"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) void handleFile(file);
                  event.target.value = "";
                }}
              />
              <Button
                variant="outline"
                onClick={() => fileInputRef.current?.click()}
                disabled={cloningMissing || busy}
              >
                <Upload className="size-4" /> Upload audio
              </Button>
              <Button variant="ghost" onClick={() => setImporting(true)}>
                <Plus className="size-4" /> Import .mockbirdvoice
              </Button>
            </div>

            {draftPcm && (
              <div className="space-y-3 rounded-lg border bg-accent/30 p-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium">
                    {draftPcm.durationSec.toFixed(1)}s · 48 kHz stereo
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setDraftPcm(null);
                      setDraftSource(null);
                    }}
                  >
                    Discard
                  </Button>
                </div>
                {analysis && (
                  <p className="text-[11px] text-muted-foreground">
                    peak {(analysis.peak * 100).toFixed(0)}% · RMS{" "}
                    {(analysis.rms * 100).toFixed(0)}%
                    {analysis.clipped > 0.005 ? " · some clipping detected" : ""}
                    {analysis.rms < 0.02 ? " · very quiet, try again closer to the mic" : ""}
                  </p>
                )}
                {/* A definite height on the *container* keeps the canvas from
                    ever being sized by its own content. */}
                <div className="h-16 w-full overflow-hidden rounded-md border bg-background/40">
                  <WaveformCanvas channels={draftPcm.channels} className="h-full w-full" />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="voice-name">Voice name</Label>
                  <Input
                    id="voice-name"
                    value={draftName}
                    onChange={(event) => setDraftName(event.target.value)}
                    placeholder="e.g. Narrator"
                  />
                </div>
                <div className="flex gap-2">
                  <Button onClick={createVoice} disabled={busy || cloningMissing}>
                    {busy ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <AudioWaveform className="size-4" />
                    )}
                    {busy ? "Encoding…" : "Create voice"}
                  </Button>
                  <Button
                    variant="outline"
                    onClick={() =>
                      downloadBlob(
                        encodeWav(draftPcm.channels, draftPcm.sampleRate),
                        "reference.wav",
                      )
                    }
                  >
                    <Download className="size-4" /> Save clip
                  </Button>
                </div>
                {cloningError && <p className="text-xs text-destructive">{cloningError}</p>}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Your voices</CardTitle>
              <CardDescription>
                Audio codes are stored in this browser. Export to move them.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {cloned.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  Nothing here yet — clone a voice to get started.
                </p>
              ) : (
                cloned.map((voice) => (
                  <div key={voice.id} className="flex items-center gap-3 rounded-lg border p-3">
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{voice.name}</p>
                      <p className="text-[11px] text-muted-foreground">
                        {voice.durationSec.toFixed(1)}s · {voice.frames} frames ·{" "}
                        {voice.numQuantizers} codebooks · {voice.source}
                      </p>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Rename"
                      onClick={() => setRenaming({ id: voice.id, name: voice.name })}
                    >
                      <Pencil className="size-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Export"
                      onClick={() => exportVoice(voice.id, voice.name)}
                    >
                      <Download className="size-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Delete"
                      onClick={() => {
                        if (confirm(`Delete “${voice.name}”?`)) {
                          client?.deleteVoice(voice.id);
                        }
                      }}
                    >
                      <Trash2 className="size-3.5 text-destructive" />
                    </Button>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Users className="size-4 text-muted-foreground" /> Built-in voices
              </CardTitle>
              <CardDescription>
                Ships with the 0.5 MB manifest — pre-computed codes, no cloning encoder needed.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-1.5">
                {all
                  .filter((voice) => voice.kind === "builtin")
                  .map((voice) => (
                    <Badge key={voice.id} variant="secondary" className="font-normal">
                      {voice.label}
                    </Badge>
                  ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <Dialog open={Boolean(renaming)} onOpenChange={(open) => !open && setRenaming(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Rename voice</DialogTitle>
            <DialogDescription>
              Only the label changes; the codes stay as they are.
            </DialogDescription>
          </DialogHeader>
          <Input
            value={renaming?.name ?? ""}
            onChange={(event) =>
              setRenaming((prev) => (prev ? { ...prev, name: event.target.value } : prev))
            }
          />
          <DialogFooter>
            <Button
              onClick={() => {
                if (renaming) client?.renameVoice(renaming.id, renaming.name.trim());
                setRenaming(null);
              }}
            >
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={importing} onOpenChange={setImporting}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Import a voice</DialogTitle>
            <DialogDescription>
              Paste the contents of a <code>.mockbirdvoice.json</code> file.
            </DialogDescription>
          </DialogHeader>
          <ImportBox
            onImport={(payload) => {
              client?.importVoice(payload);
              setImporting(false);
            }}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
}

const IDEAL_RANGE = "8–15 seconds";

function ImportBox({ onImport }: { onImport: (payload: string) => void }) {
  const [value, setValue] = useState("");
  return (
    <div className="space-y-3">
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        rows={8}
        placeholder='{"format":"mockbird-voice@1", …}'
        className="w-full rounded-lg border bg-background p-2 font-mono text-xs"
      />
      <Button onClick={() => value.trim() && onImport(value.trim())} disabled={!value.trim()}>
        Import
      </Button>
    </div>
  );
}
