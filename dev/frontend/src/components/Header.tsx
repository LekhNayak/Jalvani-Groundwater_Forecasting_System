import { NavLink, useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Building2, CircleHelp, Sprout } from "lucide-react";
import LangMenu, { ThemeSwitch } from "./LangMenu";
import { useApp } from "../context";
import type { Role } from "../lib/types";

export function Logo() {
  return (
    <svg width="30" height="30" viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="9" fill="var(--series-1)" />
      <path d="M16 6.5c-3.6 5-6 8.1-6 11.2a6 6 0 0 0 12 0c0-3.1-2.4-6.2-6-11.2z" fill="#fff" />
      <path d="M12.6 19.4c1.4 1.6 3.6 2 5.6 1.2" stroke="var(--series-1)" strokeWidth="1.6" fill="none" strokeLinecap="round" />
    </svg>
  );
}

export default function Header() {
  const { t } = useTranslation();
  const { setRole } = useApp();
  const nav = useNavigate();
  const go = (r: Role) => { setRole(r); nav(r === "farmer" ? "/farmer" : "/gov"); };
  const tab = ({ isActive }: { isActive: boolean }) =>
    `inline-flex items-center gap-2 min-h-[44px] px-3 rounded-lg text-[0.95rem] transition-colors ${
      isActive ? "text-ink font-semibold bg-[var(--accent-soft)]" : "text-ink2 hover:text-ink hover:bg-[var(--surface-2)]"}`;
  return (
    <header className="bg-surface border-b border-[var(--border)] sticky top-0 z-40">
      <div className="w-full px-4 sm:px-6 lg:px-8 py-2.5 flex flex-wrap items-center gap-x-6 gap-y-2">
        <NavLink to="/" className="flex items-center gap-2.5 min-h-[44px] mr-2" aria-label={t("common.app_name")}>
          <Logo />
          <span className="leading-tight">
            <span className="block text-lg font-semibold tracking-tight">{t("common.app_name")}</span>
            <span className="hidden sm:block text-xs text-muted">{t("common.tagline")}</span>
          </span>
        </NavLink>
        <nav data-tour="role-switch" aria-label={t("common.switch_role")} className="flex items-center gap-1 order-3 lg:order-none w-full lg:w-auto">
          <NavLink to="/farmer" className={tab} onClick={(e) => { e.preventDefault(); go("farmer"); }}>
            <Sprout size={17} aria-hidden="true" />{t("common.role_farmer")}
          </NavLink>
          <NavLink to="/gov" className={tab} onClick={(e) => { e.preventDefault(); go("government"); }}>
            <Building2 size={17} aria-hidden="true" />{t("common.role_government")}
          </NavLink>
          <NavLink to="/help" data-tour="help-link" className={tab}>
            <CircleHelp size={17} aria-hidden="true" />{t("common.help")}
          </NavLink>
        </nav>
        <div className="ml-auto flex items-center gap-2"><ThemeSwitch /><LangMenu /></div>
      </div>
    </header>
  );
}
