import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useApp } from "../context";
import { api } from "../lib/api";
import { Loader2, Square, Volume2 } from "lucide-react";
import { bcp47 } from "../lib/langs";
import { splitForSpeech } from "../lib/speech";

// A clip is made once per (language, text): the server keeps it for everyone, this keeps it for this tab.
const clips = new Map<string, Promise<string | null>>();
function clip(text: string, lang: string): Promise<string | null> {
  const key = `${lang}|${text}`;
  let p = clips.get(key);
  if (!p) {
    p = api.tts(text, lang).then(async (r) => (r.ok ? URL.createObjectURL(await r.blob()) : null)).catch(() => null);
    clips.set(key, p);
    void p.then((u) => { if (!u) clips.delete(key); });          // a failure is not remembered
  }
  return p;
}
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Read a card aloud. With the Gnani voice (server) it starts after the first short piece (about 1 s) and the rest is
 *  made while it plays; clips are cached and the first piece is made when the pointer reaches the button. Otherwise the
 *  device's own voice if it has one for the language; otherwise the button is not shown (dev/docs/SPEECH_PLAN.md). */
export default function ListenButton({ text }: { text: string }) {
  const { t, i18n } = useTranslation();
  const { meta } = useApp();
  const [state, setState] = useState<"idle" | "loading" | "playing">("idle");
  const [deviceVoice, setDeviceVoice] = useState(false);
  const audio = useRef<HTMLAudioElement | null>(null);
  const run = useRef(0);                                         // changes on every start and stop: cancels older runs
  const lang = bcp47(i18n.language);

  useEffect(() => {
    if (!("speechSynthesis" in window)) return;
    const check = () => setDeviceVoice(window.speechSynthesis.getVoices().some((v) => v.lang.toLowerCase().startsWith(lang.slice(0, 2))));
    check();
    window.speechSynthesis.addEventListener("voiceschanged", check);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", check);
  }, [lang]);
  useEffect(() => () => { run.current++; audio.current?.pause(); window.speechSynthesis?.cancel(); }, []);

  const server = !!meta?.features.speech && (meta.features.speech_languages ?? []).includes(i18n.language);
  if (!server && !deviceVoice) return null;

  const prefetch = () => { if (server && state === "idle") void clip(splitForSpeech(text)[0], i18n.language); };

  function stop() {
    run.current++;
    audio.current?.pause();
    window.speechSynthesis?.cancel();
    setState("idle");
  }

  function deviceSpeak(spoken: string, mine: number) {
    if (!("speechSynthesis" in window) || !deviceVoice) { setState("idle"); return; }
    const u = new SpeechSynthesisUtterance(spoken);
    u.lang = lang;
    u.onend = () => { if (run.current === mine) setState("idle"); };
    window.speechSynthesis.speak(u);
    setState("playing");
  }

  async function start() {
    const mine = ++run.current;
    setState("loading");
    if (!server) { deviceSpeak(text, mine); return; }
    const parts = splitForSpeech(text);
    const urls: Promise<string | null>[] = [];
    // one piece is made at a time (the voice service allows one request at a time), each ready before it is needed
    void (async () => { for (let i = 0; i < parts.length && run.current === mine; i++) { urls[i] = clip(parts[i], i18n.language); await urls[i]; } })();
    for (let i = 0; i < parts.length; i++) {
      while (!urls[i] && run.current === mine) await sleep(15);
      const url = await urls[i];
      if (run.current !== mine) return;
      if (!url) { deviceSpeak(parts.slice(i).join(" "), mine); return; }      // the voice service failed: use the device voice
      await new Promise<void>((resolve) => {
        const a = new Audio(url);
        audio.current = a;
        a.onended = () => resolve();
        a.onerror = () => resolve();
        a.play().then(() => { if (run.current === mine) setState("playing"); }).catch(() => resolve());
      });
      if (run.current !== mine) return;
    }
    setState("idle");
  }

  const busy = state !== "idle";
  return (
    <button onClick={() => (busy ? stop() : void start())} onPointerEnter={prefetch} onFocus={prefetch} onTouchStart={prefetch}
      data-tour="listen-button" aria-pressed={busy} aria-busy={state === "loading"}
      className="btn !min-h-[36px] !px-2.5 text-sm shrink-0">
      {state === "loading" ? <Loader2 size={15} aria-hidden="true" className="animate-spin" />
        : busy ? <Square size={14} aria-hidden="true" /> : <Volume2 size={15} aria-hidden="true" />}
      {busy ? t("common.stop") : t("common.listen")}
    </button>
  );
}
