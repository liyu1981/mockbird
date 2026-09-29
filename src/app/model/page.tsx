import { redirect } from "next/navigation";

/** Model management moved into the Settings tab. */
export default function ModelRedirect() {
  redirect("/settings");
}
