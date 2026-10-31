import { Link, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { setLang } from "../i18n";
import { useApp } from "../context";
import type { Lang } from "../lib/types";

const LANGS: { code: Lang; label: string }[] = [
  { code: "en", label: "English" }, { code: "mr", label: "मराठी" }, { code: "hi", label: "हिन्दी" },
];

export function LangSwitch({ big = false }: { big?: boolean }) {
  const { t, i18n } = useTranslation();
  return (
    <div role="group" aria-label={t("common.language")} data-tour="lang-switch" className="flex gap-1">
      {LANGS.map((l) => (
        <button key={l.code} lang={l.code} aria-pressed={i18n.language === l.code} onClick={() => setLang(l.code)}
          className={`${big ? "px-5 py-3 text-lg" : "px-3 py-2 text-sm"} min-h-[44px] rounded-lg border border-[var(--border)]
            ${i18n.language === l.code ? "bg-water text-white border-transparent" : "bg-surface text-ink hover:bg-plane"}`}>
          {l.label}
        </button>
      ))}
    </div>
  );
}

export default function Header() {
  const { t } = useTranslation();
  const { role, setRole } = useApp();
  const nav = useNavigate();
  const other = role === "government" ? "farmer" : "government";
  return (
    <header className="bg-surface border-b border-[var(--border)]">
      <div className="max-w-6xl mx-auto px-4 py-3 flex flex-wrap items-center gap-3 justify-between">
        <Link to="/" className="flex items-center gap-2 min-h-[44px]" aria-label={t("common.app_name")}>
          <svg width="28" height="28" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 2C8 8 5 11.5 5 15a7 7 0 0 0 14 0c0-3.5-3-7-7-13z" fill="var(--series-1)" />
          </svg>
          <span className="text-xl font-semibold">{t("common.app_name")}</span>
          <span className="hidden sm:inline text-ink2 text-sm">{t("common.tagline")}</span>
        </Link>
        <nav className="flex flex-wrap items-center gap-2">
          <LangSwitch />
          {role && (
            <button data-tour="role-switch" className="px-3 py-2 min-h-[44px] rounded-lg border border-[var(--border)] text-sm hover:bg-plane"
              onClick={() => { setRole(other); nav(other === "farmer" ? "/farmer" : "/gov"); }}>
              {t("common.switch_role")}: {t(other === "farmer" ? "common.role_farmer" : "common.role_government")}
            </button>
          )}
          <Link to="/help" data-tour="help-link" className="px-3 py-2 min-h-[44px] inline-flex items-center rounded-lg text-sm underline">
            {t("common.help")}
          </Link>
        </nav>
      </div>
    </header>
  );
}
