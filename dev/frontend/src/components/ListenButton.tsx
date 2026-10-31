import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useApp } from "../context";
import { api } from "../lib/api";
import { Square, Volume2 } from "lucide-react";

const BCP47: Record<string, string> = { en: "en-IN", mr: "mr-IN", hi: "hi-IN" };

/** Read a card aloud. Uses the Gnani TTS backend when configured; otherwise the device's own voice if it has one
 *  for the chosen language; otherwise the button is not shown (dev/docs/SPEECH_PLAN.md). */
export default function ListenButton({ text }: { text: string }) {
  const { t, i18n } = useTranslation();
  const { meta } = useApp();
  const [speaking, setSpeaking] = useState(false);
  const [deviceVoice, setDeviceVoice] = useState(false);
  const audio = useRef<HTMLAudioElement | null>(null);
  const lang = BCP47[i18n.language] ?? "en-IN";

  useEffect(() => {
    if (!("speechSynthesis" in window)) return;
    const check = () => setDeviceVoice(window.speechSynthesis.getVoices().some((v) => v.lang.toLowerCase().startsWith(lang.slice(0, 2))));
    check();
    window.speechSynthesis.addEventListener("voiceschanged", check);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", check);
  }, [lang]);

  const server = !!meta?.features.speech;
  if (!server && !deviceVoice) return null;

  async function toggle() {
    if (speaking) {
      audio.current?.pause();
      window.speechSynthesis?.cancel();
      setSpeaking(false);
      return;
    }
    setSpeaking(true);
    if (server) {
      const r = await api.tts(text, i18n.language);
      if (r.ok) {
        audio.current = new Audio(URL.createObjectURL(await r.blob()));
        audio.current.onended = () => setSpeaking(false);
        audio.current.play();
        return;
      }
    }
    const u = new SpeechSynthesisUtterance(text);
    u.lang = lang;
    u.onend = () => setSpeaking(false);
    window.speechSynthesis.speak(u);
  }

  return (
    <button onClick={toggle} data-tour="listen-button" aria-pressed={speaking}
      className="btn !min-h-[36px] !px-2.5 text-sm shrink-0">
      {speaking ? <Square size={14} aria-hidden="true" /> : <Volume2 size={15} aria-hidden="true" />}{speaking ? t("common.stop") : t("common.listen")}
    </button>
  );
}
