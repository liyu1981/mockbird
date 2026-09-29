import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import "./globals.css";
import { EngineProvider } from "@/components/engine/EngineProvider";
import { AppShell } from "@/components/layout/AppShell";
import { ThemeProvider } from "@/components/ThemeProvider";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Mockbird · local voice cloning with MOSS-TTS-Nano",
    template: "%s · Mockbird",
  },
  description:
    "Clone a voice and generate speech in your browser. MOSS-TTS-Nano (0.1B, 48 kHz stereo, 20 languages) running fully on-device with ONNX Runtime Web.",
  applicationName: "Mockbird",
  appleWebApp: { capable: true, title: "Mockbird", statusBarStyle: "black-translucent" },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fafafa" },
    { media: "(prefers-color-scheme: dark)", color: "#09090b" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full bg-background text-foreground">
        <ThemeProvider>
          <TooltipProvider>
            <EngineProvider>
              <AppShell>{children}</AppShell>
              <Toaster position="bottom-right" richColors closeButton />
            </EngineProvider>
          </TooltipProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
