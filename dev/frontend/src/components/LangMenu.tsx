import { useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { Check, ChevronDown, Languages, Monitor, Moon, Sun } from "lucide-react";
import { setLang } from "../i18n";
import { useApp } from "../context";
import { LANGS, type Lang } from "../lib/langs";
import type { ThemePref } from "../lib/theme";

/** Language dropdown (ARIA listbox): each language in its own script, with its English name for helpers.
 *  Keyboard: ↑ ↓ Home End to move, Enter or Space to choose, Esc to close. */
export default function LangMenu() {
  const { t, i18n } = useTranslation();
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const wrap = useRef<HTMLDivElement>(null);
  const btn = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const cur = LANGS.find((l) => l.code === i18n.language) ?? LANGS[0];

  useEffect(() => {
    if (!open) return;
    setActive(LANGS.findIndex((l) => l.code === cur.code));
    requestAnimationFrame(() => list.current?.focus());
    const away = (e: MouseEvent) => { if (!wrap.current?.contains(e.target as Node)) setOpen(false); };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open, cur.code]);
  useEffect(() => { list.current?.querySelector(`[data-i="${active}"]`)?.scrollIntoView({ block: "nearest" }); }, [active]);

  const choose = (code: Lang) => { setOpen(false); btn.current?.focus(); void setLang(code); };
  const onKey = (e: React.KeyboardEvent) => {
    const n = LANGS.length;
    const go: Record<string, () => void> = {
      ArrowDown: () => setActive((a) => (a + 1) % n), ArrowUp: () => setActive((a) => (a - 1 + n) % n),
      Home: () => setActive(0), End: () => setActive(n - 1),
      Enter: () => choose(LANGS[active].code), " ": () => choose(LANGS[active].code),
      Escape: () => { setOpen(false); btn.current?.focus(); },
    };
    if (e.key === "Tab") { setOpen(false); return; }
    if (go[e.key]) { e.preventDefault(); go[e.key](); }
  };

  return (
    <div ref={wrap} className="relative" data-tour="lang-switch">
      <button ref={btn} className="btn" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((o) => !o)}
        aria-label={`${t("common.language")}: ${cur.native}`}
        onKeyDown={(e) => { if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); setOpen(true); } }}>
        <Languages size={17} aria-hidden="true" className="text-muted" />
        <span lang={cur.code} className="font-semibold">{cur.native}</span>
        <ChevronDown size={16} aria-hidden="true" className={`text-muted transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <ul ref={list} role="listbox" tabIndex={-1} aria-label={t("common.language")} onKeyDown={onKey}
          aria-activedescendant={`lang-opt-${LANGS[active].code}`}
          className="absolute right-0 mt-2 w-[min(18rem,calc(100vw-2rem))] max-h-[70vh] overflow-auto tile !rounded-xl p-1.5 z-50 fade-in outline-none"
          style={{ boxShadow: "var(--shadow-pop)" }}>
          {LANGS.map((l, i) => (
            <li key={l.code} id={`lang-opt-${l.code}`} data-i={i} role="option" aria-selected={l.code === cur.code}
              onClick={() => choose(l.code)} onMouseMove={() => setActive(i)}
              className={`flex items-center gap-3 px-3 min-h-[44px] rounded-lg cursor-pointer ${i === active ? "bg-[var(--surface-2)]" : ""}`}>
              <span lang={l.code} className={`flex-1 ${l.code === cur.code ? "font-semibold" : ""}`}>{l.native}</span>
              <span className="text-xs text-muted">{l.english}</span>
              <Check size={16} aria-hidden="true" className={l.code === cur.code ? "text-water" : "invisible"} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** Start page: every language as a large button in its own script (recognition, not reading). */
export function LangGrid() {
  const { t, i18n } = useTranslation();
  return (
    <div role="group" aria-label={t("common.language")} data-tour="lang-switch" className="grid grid-cols-2 sm:grid-cols-4 gap-2">
      {LANGS.map((l) => {
        const on = i18n.language === l.code;
        return (
          <button key={l.code} aria-pressed={on} onClick={() => void setLang(l.code)}
            className={`text-left rounded-xl border px-3.5 py-2.5 min-h-[56px] transition-colors
              ${on ? "border-transparent bg-[var(--accent-soft)] ring-2 ring-[var(--accent)]" : "border-[var(--border)] hover:bg-[var(--surface-2)]"}`}>
            <span lang={l.code} className="block text-lg font-semibold leading-tight">{l.native}</span>
            <span className="block text-xs text-muted">{l.english}</span>
          </button>
        );
      })}
    </div>
  );
}

const THEMES: { id: ThemePref; Icon: typeof Sun }[] = [{ id: "light", Icon: Sun }, { id: "dark", Icon: Moon }, { id: "system", Icon: Monitor }];

/** Light / dark / follow the device, as three icon buttons with names for screen readers and tooltips. */
export function ThemeSwitch() {
  const { t } = useTranslation();
  const { theme, setTheme } = useApp();
  return (
    <div role="group" aria-label={t("common.theme")} data-tour="theme-switch" className="seg">
      {THEMES.map(({ id, Icon }) => (
        <button key={id} aria-pressed={theme === id} onClick={() => setTheme(id)} title={t(`common.theme_${id}`)}
          aria-label={t(`common.theme_${id}`)} className="!px-2.5">
          <Icon size={16} aria-hidden="true" />
        </button>
      ))}
    </div>
  );
}
