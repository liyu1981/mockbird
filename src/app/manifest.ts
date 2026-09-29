import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Mockbird · local voice cloning",
    short_name: "Mockbird",
    description:
      "Clone a voice and generate speech in your browser with MOSS-TTS-Nano, running fully on-device.",
    start_url: "/studio",
    display: "standalone",
    background_color: "#0a0a0a",
    theme_color: "#0a0a0a",
    orientation: "any",
    categories: ["productivity", "utilities"],
    icons: [{ src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" }],
  };
}
