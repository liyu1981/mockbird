"use client";

import { useAtom, useAtomValue } from "jotai";
import { Download, Loader2, Play, Plus, Trash2, Upload, WandSparkles } from "lucide-react";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";
import { useEngine } from "@/components/engine/EngineProvider";
import { Details, Panel } from "@/components/layout/Panel";
import { WaveformCanvas } from "@/components/studio/WaveformCanvas";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
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
  builtinVoicesAtom,
  clonedVoicesAtom,
  cloningBusyAtom,
  cloningErrorAtom,
  draftAnalysisAtom,
  draftNameAtom,
  draftPcmAtom,
  draftSourceAtom,
  selectedVoiceAtom,
} from "@/state/voiceAtoms";

export default function ClonePage() {
  const { client, downloadModels, generate, player } = useEngine();
  const engine = useAtomValue(engineAtom);
  const cloned = useAtomValue(clonedVoicesAtom);
  const builtin = useAtomValue(builtinVoicesAtom);
  const [, setSelectedVoice] = useAtom(selectedVoiceAtom);
  const [draftName, setDraftName] = useAtom(draftNameAtom);
  const [draftPcm, setDraftPcm] = useAtom(draftPcmAtom);
  const [draftSource, setDraftSource] = useAtom(draftSourceAtom);
  const analysis = useAtomValue(draftAnalysisAtom);
  const [busy, setBusy] = useAtom(cloningBusyAtom);
  const [cloningError, setCloningError] = useAtom(cloningErrorAtom);

  const [recording, setRecording] = useState(false);
  const [previewing, setPreviewing] = useState<string | null>(null);
  const [importing, setImporting] = useState(false);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const encoderMissing = !engine.cloningReady;

  const acceptCapture = useCallback(
    (result: RecordingResult, source: "recording" | "upload") => {
      setDraftPcm({
        channels: result.channels,
        sampleRate: result.sampleRate,
        durationSec: result.durationSec,
      });
      setDraftSource(source);
      if (!draftName.trim()) setDraftName(`Voice ${new Date().toLocaleDateString()}`);
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
    busy,
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
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  }, []);

  /** Say a sample line with one of the user's own voices. */
  const previewVoice = useCallback(
    (id: string, name: string) => {
      const voice = { kind: "cloned" as const, id, label: name, group: "Your voice" };
      setSelectedVoice(voice);
      generate({
        text: "你好，这是我的声音。 Hello, this is what my cloned voice sounds like.",
        voice,
      });
      player.resume();
      setPreviewing(id);
      window.setTimeout(() => setPreviewing(null), 8000);
    },
    [generate, player, setSelectedVoice],
  );

  return (
    <div className="space-y-4">
      <ReadingScriptCard recording={recording} />

      {encoderMissing && (
        <div className="glass-control rounded-xl p-5">
          <div className="flex flex-wrap items-center gap-4">
            <div className="min-w-0 flex-1">
              <p className="font-medium">One more piece to download</p>
              <p className="mt-1 text-sm text-muted-foreground">
                Cloning needs a small extra encoder (about 45 MB) that turns your recording into
                audio codes. The 18 built-in voices work without it.
              </p>
            </div>
            <Button
              size="lg"
              onClick={() => downloadModels(["codec-encode"])}
              disabled={engine.files.length === 0}
            >
              <Download className="size-4" /> Download encoder
            </Button>
          </div>
        </div>
      )}

      <Panel
        title="Record your voice"
        description="Read the paragraph above, then press record. Ten seconds of clear speech is plenty."
      >
        <div className="space-y-4">
          <VoiceRecorder
            onCaptured={(result) => acceptCapture(result, "recording")}
            onRecordingChange={setRecording}
            disabled={encoderMissing || busy}
          />

          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-muted-foreground">or</span>
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
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={encoderMissing || busy}
            >
              <Upload className="size-4" /> Upload a recording
            </Button>
          </div>

          {draftPcm && (
            <div className="space-y-3 rounded-xl border border-border/60 bg-background/40 p-4">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">
                  {draftPcm.durationSec.toFixed(1)} seconds captured
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

              <div className="h-16 w-full overflow-hidden rounded-lg border border-border/60 bg-background/40">
                <WaveformCanvas channels={draftPcm.channels} className="h-full w-full" />
              </div>

              {analysis && (
                <p className="text-xs text-muted-foreground">
                  {analysis.rms < 0.02
                    ? "This clip is very quiet — try moving closer to the microphone."
                    : "Level looks good."}{" "}
                  {analysis.clipped > 0.005 ? "A little clipping detected." : ""}
                </p>
              )}

              <div className="space-y-1.5">
                <Label htmlFor="voice-name">Give it a name</Label>
                <Input
                  id="voice-name"
                  value={draftName}
                  onChange={(event) => setDraftName(event.target.value)}
                  placeholder="e.g. Narrator"
                />
              </div>

              <div className="flex flex-wrap gap-2">
                <Button size="lg" onClick={createVoice} disabled={busy || encoderMissing}>
                  {busy ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <WandSparkles className="size-4" />
                  )}
                  {busy ? "Creating…" : "Create this voice"}
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  onClick={() =>
                    downloadBlob(
                      encodeWav(draftPcm.channels, draftPcm.sampleRate),
                      "recording.wav",
                    )
                  }
                >
                  <Download className="size-4" /> Save the raw clip
                </Button>
              </div>

              {cloningError && <p className="text-sm text-destructive">{cloningError}</p>}
            </div>
          )}
        </div>
      </Panel>

      <Panel
        title="Your voices"
        description="Stored in this browser. Use one in the Generate tab."
        actions={
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setAdvancedOpen((value) => !value)}
            className="text-xs"
          >
            {advancedOpen ? "Hide tools" : "Import / export"}
          </Button>
        }
      >
        {cloned.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Nothing here yet — record a sample above and your first voice will show up in this
            list.
          </p>
        ) : (
          <ul className="space-y-2">
            {cloned.map((voice) => (
              <li
                key={voice.id}
                className="flex flex-wrap items-center gap-3 rounded-lg border border-border/60 px-3 py-2.5"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">{voice.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {voice.durationSec.toFixed(1)}s · recorded{" "}
                    {new Date(voice.createdAt).toLocaleDateString()}
                  </p>
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => previewVoice(voice.id, voice.name)}
                  disabled={!engine.loaded}
                >
                  {previewing === voice.id ? (
                    <Loader2 className="size-3.5 animate-spin" />
                  ) : (
                    <Play className="size-3.5" />
                  )}
                  Hear it
                </Button>

                {advancedOpen && (
                  <>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => exportVoice(voice.id, voice.name)}
                    >
                      <Download className="size-3.5" /> Export
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-destructive"
                      onClick={() => {
                        if (confirm(`Delete “${voice.name}”?`)) {
                          client?.deleteVoice(voice.id);
                        }
                      }}
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </>
                )}
              </li>
            ))}
          </ul>
        )}

        {advancedOpen && (
          <p className="mt-3 text-xs text-muted-foreground">
            Voice files are plain JSON containing the audio codes. Import one to move a voice to
            another browser.
          </p>
        )}
      </Panel>

      <Details
        summary="Built-in voices"
        description={`${builtin.length} voices ship with the model — no recording needed.`}
      >
        <div className="flex flex-wrap gap-1.5">
          {builtin.map((voice) => (
            <Badge key={voice.voice} variant="secondary" className="font-normal">
              {voice.display_name}
            </Badge>
          ))}
        </div>
      </Details>

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

function ImportBox({ onImport }: { onImport: (payload: string) => void }) {
  const [value, setValue] = useState("");
  return (
    <div className="space-y-3">
      <textarea
        value={value}
        onChange={(event) => setValue(event.target.value)}
        rows={8}
        placeholder={'{"format":"mockbird-voice@1", …}'}
        className="w-full rounded-lg border bg-background p-2 font-mono text-xs"
      />
      <div className="flex items-center gap-2">
        <Button onClick={() => value.trim() && onImport(value.trim())} disabled={!value.trim()}>
          <Plus className="size-4" /> Import
        </Button>
        <Button variant="ghost" onClick={() => setValue("")}>
          Clear
        </Button>
      </div>
    </div>
  );
}
