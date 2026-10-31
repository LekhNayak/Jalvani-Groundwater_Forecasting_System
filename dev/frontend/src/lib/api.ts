import { useEffect, useState } from "react";
import type { ChatAction, District, DistrictLite, History, Meta, Overview, PriorityRow } from "./types";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json() as Promise<T>;
}

export const api = {
  meta: () => get<Meta>("/api/meta"),
  districts: () => get<{ districts: DistrictLite[] }>("/api/districts").then((x) => x.districts),
  district: (slug: string) => get<District>(`/api/districts/${slug}`),
  history: (slug: string) => get<History>(`/api/districts/${slug}/history`),
  overview: () => get<Overview>("/api/state/overview"),
  priorities: () => get<{ rows: PriorityRow[] }>("/api/state/priorities").then((x) => x.rows),
  geo: () => get<GeoJSON.FeatureCollection>("/api/geo/districts"),
  chat: (body: object) =>
    fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
      .then((r) => r.json() as Promise<{ reply: string; actions: ChatAction[]; mode: string }>),
  tts: (text: string, lang: string) =>
    fetch("/api/speech/tts", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, lang }) }),
};

/** Small fetch hook: {data, error, loading}. Refetches when `key` changes. */
export function useApi<T>(fn: () => Promise<T>, key: string) {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>(
    { data: null, error: null, loading: true });
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn().then((data) => alive && setState({ data, error: null, loading: false }))
        .catch((e) => alive && setState({ data: null, error: String(e), loading: false }));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return state;
}
