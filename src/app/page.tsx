import { redirect } from "next/navigation";

/**
 * Entry point.
 *
 * The app is only useful once the ONNX weights are in the browser's OPFS cache,
 * so a first-time visitor is sent straight to the model downloader. Returning
 * users (engine already loaded) go to the studio. This is a server component and
 * cannot know the OPFS state, so we optimistically route to the studio: the
 * Studio page shows a "get the model" banner if the engine isn't ready yet, and
 * the sidebar exposes the downloader at all times. The static export keeps this
 * instant.
 */
export default function Home() {
  redirect("/studio");
}
