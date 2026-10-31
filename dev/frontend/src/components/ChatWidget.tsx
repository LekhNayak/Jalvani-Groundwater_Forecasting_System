import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { useApp } from "../context";
import { api } from "../lib/api";
import { toWav16k } from "../lib/wav";
import type { ChatAction } from "../lib/types";

interface Msg { role: "user" | "assistant"; content: string; actions?: ChatAction[] }

/** The site guide: navigation help and explanations of what is on screen.
 *  Works now in offline guide mode; uses Groq once GROQ_API_KEY is set on the backend (dev/docs/CHATBOT.md). */
export default function ChatWidget() {
  const { t, i18n } = useTranslation();
  const { role, meta, highlight, startTour, visible } = useApp();
  const nav = useNavigate();
  const loc = useLocation();
  const slug = loc.pathname.match(/^\/(?:farmer|gov\/district)\/([a-z_]+)/)?.[1] ?? null;
  const [recording, setRecording] = useState(false);
  const recorder = useRef<MediaRecorder | null>(null);
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => { listRef.current?.scrollTo({ top: listRef.current.scrollHeight }); }, [msgs, open]);

  function run(actions: ChatAction[] | undefined) {
    for (const a of actions ?? []) {
      if (a.type === "highlight") { setOpen(false); highlight(a.element); }
      if (a.type === "start_tour") { setOpen(false); setTimeout(() => startTour(a.tour), 300); }
      // navigation is never automatic: the reply shows a "Go there" button (user stays in control)
    }
  }

  async function send(message: string) {
    if (!message.trim() || busy) return;
    const history = msgs.map(({ role: r, content }) => ({ role: r, content }));
    setMsgs((m) => [...m, { role: "user", content: message }]);
    setText("");
    setBusy(true);
    try {
      const r = await api.chat({ message, lang: i18n.language, role: role ?? "farmer", page: loc.pathname,
                                 district: slug, visible_context: visible, history });
      setMsgs((m) => [...m, { role: "assistant", content: r.reply, actions: r.actions }]);
      const navAction = r.actions.find((a) => a.type === "navigate");
      if (!navAction || navAction.route === loc.pathname) run(r.actions);
    } catch {
      setMsgs((m) => [...m, { role: "assistant", content: t("chat.error") }]);
    } finally {
      setBusy(false);
    }
  }

  /** push-to-talk: record (auto-stop at 25 s; Gnani accepts up to 30 s), convert to 16 kHz WAV, send to
   *  /api/speech/stt (Gnani); the text lands in the input box for the user to check, then Send */
  async function toggleMic() {
    if (recording) { recorder.current?.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const chunks: Blob[] = [];
      const rec = new MediaRecorder(stream);
      rec.ondataavailable = (e) => chunks.push(e.data);
      rec.onstop = async () => {
        stream.getTracks().forEach((tr) => tr.stop());
        clearTimeout(limit);
        setRecording(false);
        setBusy(true);
        try {
          const fd = new FormData();
          fd.append("audio", await toWav16k(new Blob(chunks, { type: rec.mimeType })), "speech.wav");
          fd.append("lang", i18n.language);
          const r = await fetch("/api/speech/stt", { method: "POST", body: fd });
          if (r.ok) { const { text: heard } = await r.json(); setText(heard); }
          else setMsgs((m) => [...m, { role: "assistant", content: t("chat.error") }]);
        } finally {
          setBusy(false);
        }
      };
      const limit = setTimeout(() => rec.state === "recording" && rec.stop(), 25000);
      recorder.current = rec;
      rec.start();
      setRecording(true);
    } catch {
      setRecording(false);
    }
  }

  const suggestions = role === "government"
    ? ["g.priorities", "g.map", "g.needs", "g.tour"] : ["s.forecast", "s.reliability", "s.tour", "s.district"];

  return (
    <>
      <button data-tour="chat-button" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-controls="chat-panel"
        aria-label={t("chat.open")}
        className="fixed bottom-4 right-4 z-50 rounded-full bg-water text-white shadow-lg px-4 sm:px-5 py-3 min-h-[52px] min-w-[52px] text-base font-semibold">
        <span aria-hidden="true">💬</span><span className="sr-only sm:not-sr-only sm:ml-1">{t("chat.open")}</span>
      </button>
      {open && (
        <section id="chat-panel" role="dialog" aria-label={t("chat.title")}
          className="fixed z-50 bottom-20 right-4 left-4 sm:left-auto sm:w-[380px] h-[70vh] max-h-[560px] card shadow-2xl flex flex-col">
          <header className="flex items-center justify-between px-4 py-3 border-b border-[var(--border)]">
            <div>
              <h2 className="font-semibold">{t("chat.title")}</h2>
              <p className="text-xs text-muted">{meta?.features.chat === "groq" ? t("chat.mode_groq") : t("chat.mode_guide")}</p>
            </div>
            <button onClick={() => setOpen(false)} className="px-3 py-2 min-h-[44px] rounded-lg hover:bg-plane" aria-label={t("common.close")}>✕</button>
          </header>
          <div ref={listRef} className="flex-1 overflow-y-auto px-4 py-3 space-y-3" aria-live="polite">
            <p className="text-ink2">{t("chat.hello")}</p>
            {msgs.map((m, i) => (
              <div key={i} className={m.role === "user" ? "text-right" : ""}>
                <div className={`inline-block max-w-[90%] rounded-2xl px-3 py-2 text-left ${m.role === "user" ? "bg-water text-white" : "bg-plane"}`}>
                  {m.content}
                </div>
                {m.actions?.filter((a) => a.type === "navigate").map((a, j) => (
                  <div key={j}>
                    <button className="mt-1 px-3 py-2 min-h-[44px] rounded-lg border border-[var(--border)] text-sm"
                      onClick={() => { if (a.type === "navigate") nav(a.route); run(m.actions?.filter((x) => x.type !== "navigate")); }}>
                      {t("chat.go")} →
                    </button>
                  </div>
                ))}
              </div>
            ))}
            {busy && <p className="text-muted">…</p>}
          </div>
          <div className="px-4 pb-2 flex flex-wrap gap-2">
            {suggestions.map((k) => (
              <button key={k} onClick={() => send(t(`chat.${k}`))} className="text-sm rounded-full border border-[var(--border)] px-3 py-1.5 min-h-[36px] hover:bg-plane">
                {t(`chat.${k}`)}
              </button>
            ))}
          </div>
          <form className="flex gap-2 p-3 border-t border-[var(--border)]" onSubmit={(e) => { e.preventDefault(); send(text); }}>
            <label htmlFor="chat-input" className="sr-only">{t("chat.placeholder")}</label>
            <input id="chat-input" value={text} onChange={(e) => setText(e.target.value)} placeholder={t("chat.placeholder")}
              className="flex-1 rounded-lg border border-[var(--border)] bg-surface px-3 py-2 min-h-[44px]" maxLength={500} />
            {meta?.features.speech && (
              <button type="button" onClick={toggleMic} aria-pressed={recording}
                className={`px-3 py-2 min-h-[44px] rounded-lg border border-[var(--border)] ${recording ? "bg-[var(--status-serious)]" : ""}`}
                aria-label={recording ? t("chat.listening") : t("chat.mic")} title={recording ? t("chat.listening") : t("chat.mic")}>🎤</button>
            )}
            <button type="submit" disabled={busy} className="px-4 py-2 min-h-[44px] rounded-lg bg-water text-white disabled:opacity-50">{t("chat.send")}</button>
          </form>
        </section>
      )}
    </>
  );
}
