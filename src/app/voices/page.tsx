import { redirect } from "next/navigation";

/** The voice library now lives under the Clone tab. */
export default function VoicesRedirect() {
  redirect("/clone");
}
