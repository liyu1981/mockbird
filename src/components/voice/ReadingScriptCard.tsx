"use client";

/**
 * "Read this aloud" panel shown before recording a voice-cloning sample.
 *
 * Gives the speaker a fixed paragraph in the language their voice actually
 * speaks: consistent phonemes and prosody make the sample more faithful, and
 * the same text every time makes two recordings comparable.
 */

import { useAtomValue } from "jotai";
import { Check, Copy, Languages, Volume2, WandSparkles } from "lucide-react";
import { useCallback, useState } from "react";
import { toast } from "sonner";
import { useEngine } from "@/components/engine/EngineProvider";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  DEFAULT_SCRIPT_CODE,
  estimateReadingSeconds,
  getReadingScript,
  READING_SCRIPTS,
} from "@/lib/voice/readingScripts";
import { engineAtom } from "@/state/modelAtoms";
import { builtinVoicesAtom } from "@/state/voiceAtoms";

export function ReadingScriptCard({ recording = false }: { recording?: boolean }) {
  const { generate, player, ready } = useEngine();
  const voices = useAtomValue(builtinVoicesAtom);
  const engine = useAtomValue(engineAtom);
  const [code, setCode] = useState(DEFAULT_SCRIPT_CODE);
  const [copied, setCopied] = useState(false);

  const script = getReadingScript(code);
  const seconds = estimateReadingSeconds(script);
  const tooShort = seconds < 8;
  const tooLong = seconds > 25;

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(script.text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      toast.error("Clipboard access was blocked.");
    }
  }, [script.text]);

  const preview = useCallback(() => {
    // A built-in voice reads the target text so the speaker hears the intended
    // pronunciation before reading it themselves.
    const fallback = voices.find((voice) => /en/i.test(voice.group)) ?? voices[0];
    if (!voices.length) {
      toast.error("The engine is still loading — try again in a moment.");
      return;
    }
    const voice = script.code === "en" && voices[0] ? voices[0] : (fallback ?? voices[0]);
    if (!voice) return;
    generate({
      text: script.text,
      voice: {
        kind: "builtin",
        id: voice.voice,
        label: voice.display_name,
        group: voice.group,
      },
    });
    player.resume();
  }, [generate, player, script.text, script.code, voices]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Languages className="size-4 text-muted-foreground" /> What to read
        </CardTitle>
        <CardDescription>
          Pick the language your voice actually speaks, then read the paragraph aloud at your
          normal pace — roughly <strong className="text-foreground">{seconds} seconds</strong>.
          Consistency matters more than volume.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <Select value={code} onValueChange={(value) => setCode(value ?? DEFAULT_SCRIPT_CODE)}>
            <SelectTrigger className="w-44" aria-label="Script language">
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="max-h-80">
              {READING_SCRIPTS.map((item) => (
                <SelectItem key={item.code} value={item.code}>
                  {item.native} · {item.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button variant="outline" size="sm" onClick={copy}>
            {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
            {copied ? "Copied" : "Copy text"}
          </Button>

          <Button
            variant="outline"
            size="sm"
            onClick={preview}
            disabled={recording || !ready || !engine.loaded}
            title={
              recording
                ? "Stop recording before previewing"
                : ready
                  ? "Hear the paragraph read by a built-in voice"
                  : "The engine is still loading"
            }
          >
            <Volume2 className="size-3.5" /> Hear it first
          </Button>

          {tooShort && <Badge variant="secondary">a little short</Badge>}
          {tooLong && <Badge variant="secondary">a little long</Badge>}
        </div>

        <blockquote
          dir={script.dir}
          lang={script.code}
          className="rounded-lg border bg-muted/30 p-3 text-[15px] leading-relaxed text-pretty"
        >
          {script.text}
        </blockquote>

        <p className="text-xs text-muted-foreground">
          <WandSparkles className="mr-1 inline size-3" />
          Tips: one speaker only · no music, no fan, no reverb · keep 20–30 cm from the
          microphone · read the same paragraph each time you compare voices.
        </p>
      </CardContent>
    </Card>
  );
}
