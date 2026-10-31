import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ArrowRight, Building2, CalendarClock, LineChart, Network, Sprout, Waves, type LucideIcon } from "lucide-react";
import { Logo } from "../components/Header";
import { LangGrid } from "../components/LangMenu";
import { useApp } from "../context";
import { fmtDate } from "../lib/format";
import type { Lang, Role } from "../lib/types";

function Stat({ icon: Icon, value, label }: { icon: LucideIcon; value: string; label: string }) {
  return (
    <div className="tile p-5 fade-in">
      <div className="eyebrow flex items-center gap-2"><Icon size={15} aria-hidden="true" className="text-muted" />{label}</div>
      <div className="text-3xl font-semibold tracking-tight tabnum mt-2">{value}</div>
    </div>
  );
}

export default function StartPage() {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const { setRole, meta } = useApp();
  const nav = useNavigate();
  const choose = (r: Role) => { setRole(r); nav(r === "farmer" ? "/farmer" : "/gov"); };
  const roles: { r: Role; Icon: LucideIcon; title: string; desc: string }[] = [
    { r: "farmer", Icon: Sprout, title: t("common.role_farmer"), desc: t("start.farmer_desc") },
    { r: "government", Icon: Building2, title: t("common.role_government"), desc: t("start.government_desc") },
  ];
  return (
    <div className="grid grid-cols-1 md:grid-cols-6 xl:grid-cols-12 gap-4">
      <section className="tile p-7 sm:p-10 md:col-span-6 xl:col-span-7 flex flex-col gap-10 fade-in">
        <div>
          <div className="flex items-center gap-2.5 text-ink2"><Logo /><span className="font-semibold">{t("common.app_name")}</span>
            <span className="text-muted">· {t("common.tagline")}</span></div>
          <h1 className="text-4xl xl:text-5xl font-semibold tracking-tight mt-8 max-w-2xl leading-[1.1]">{t("start.welcome")}</h1>
          <p className="text-lg text-ink2 mt-4 max-w-xl">{t("start.intro")}</p>
        </div>
        <div className="mt-auto">
          <h2 className="eyebrow mb-3">{t("start.choose_language")}</h2>
          <LangGrid />
        </div>
      </section>

      <div className="md:col-span-6 xl:col-span-5 flex flex-col gap-4">
        <h2 className="sr-only">{t("start.choose_role")}</h2>
        {roles.map(({ r, Icon, title, desc }) => (
          <button key={r} onClick={() => choose(r)}
            className="tile group flex-1 p-6 sm:p-7 text-left flex flex-col gap-4 min-h-[180px] transition-[box-shadow,transform] hover:shadow-md hover:-translate-y-0.5 fade-in">
            <span className="inline-flex h-11 w-11 items-center justify-center rounded-xl bg-[var(--accent-soft)] text-water" aria-hidden="true"><Icon size={22} /></span>
            <span className="block">
              <span className="block text-xl font-semibold">{title}</span>
              <span className="block text-ink2 mt-1">{desc}</span>
            </span>
            <span className="mt-auto inline-flex items-center gap-1.5 text-sm font-semibold text-water">
              {t("start.open")}<ArrowRight size={16} aria-hidden="true" className="transition-transform group-hover:translate-x-0.5" />
            </span>
          </button>
        ))}
      </div>

      {meta && (
        <div className="col-span-full grid grid-cols-2 md:grid-cols-4 gap-4" aria-label={t("start.stats_title")}>
          <Stat icon={Network} value={String(meta.counts.districts)} label={t("gov.kpi.districts")} />
          <Stat icon={LineChart} value={String(meta.counts.with_forecast)} label={t("govd.status.forecast")} />
          <Stat icon={Waves} value={String(meta.counts.wells_used)} label={t("gov.kpi.wells")} />
          <Stat icon={CalendarClock} value={fmtDate(meta.freshness.groundwater_last_reading, lang)} label={t("start.latest_reading")} />
        </div>
      )}
    </div>
  );
}
