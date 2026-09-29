import { redirect } from "next/navigation";

/** Kept so old bookmarks and links keep working — Generate is the new home. */
export default function StudioRedirect() {
  redirect("/");
}
