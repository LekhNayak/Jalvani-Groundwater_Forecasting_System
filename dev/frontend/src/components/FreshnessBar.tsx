import { useTranslation } from "react-i18next";
import { CalendarClock, CloudRain, Droplets, LineChart, Sprout, type LucideIcon } from "lucide-react";
import { useApp } from "../context";
import { fmtDate } from "../lib/format";
import type { Lang } from "../lib/types";

function Item({ icon: Icon, children }: { icon: LucideIcon; children: React.ReactNode }) {
  return <span className="inline-flex items-center gap-1.5"><Icon size={14} aria-hidden="true" className="text-muted" />{children}</span>;
}

/** Visibility of system status: how fresh every kind of data is, on every page. */
export default function FreshnessBar() {
  const { t, i18n } = useTranslation();
  const { meta, metaError } = useApp();
  const lang = i18n.language as Lang;
  if (metaError) return <div role="alert" className="bg-[var(--status-serious)]/20 text-ink text-sm px-4 py-2 text-center">{t("common.error")}</div>;
  if (!meta) return <div className="h-9 border-b border-[var(--border)]" />;
  const f = meta.freshness;
  return (
    <div data-tour="freshness-bar" className="border-b border-[var(--border)] bg-[var(--surface-2)] text-[0.8125rem] text-ink2">
      <div className="w-full px-4 sm:px-6 lg:px-8 py-2 flex flex-wrap items-center gap-x-5 gap-y-1">
        <span className="inline-flex items-center gap-1.5 font-semibold text-ink"><CalendarClock size={14} aria-hidden="true" />{t("freshness.title")}</span>
        <Item icon={Droplets}>{t("freshness.groundwater", { date: fmtDate(f.groundwater_last_reading, lang) })}
          <span className="text-muted">· {t("freshness.groundwater_note")}</span></Item>
        <Item icon={CloudRain}>{t("freshness.rain", { date: fmtDate(f.rain_as_of, lang) })}
          {f.rain_provisional_from && <span className="text-muted">· {t("freshness.rain_prov", { date: fmtDate(f.rain_provisional_from, lang) })}</span>}</Item>
        <Item icon={Sprout}>{t("freshness.soil", { date: fmtDate(f.soil_as_of, lang) })}</Item>
        <Item icon={LineChart}>{meta.model.operational_forecast ? t("freshness.forecast_issued", { date: fmtDate(meta.model.last_issue_date, lang) })
                                                                 : t("freshness.forecast_none")}</Item>
      </div>
    </div>
  );
}
