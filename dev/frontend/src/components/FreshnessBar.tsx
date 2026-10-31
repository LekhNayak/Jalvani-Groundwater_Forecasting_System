import { useTranslation } from "react-i18next";
import { useApp } from "../context";
import { fmtDate } from "../lib/format";
import type { Lang } from "../lib/types";

/** Visibility of system status: how fresh every kind of data is, on every page. */
export default function FreshnessBar() {
  const { t, i18n } = useTranslation();
  const { meta, metaError } = useApp();
  const lang = i18n.language as Lang;
  if (metaError) return <div role="alert" className="bg-[var(--status-serious)]/20 text-ink text-sm px-4 py-2 text-center">{t("common.error")}</div>;
  if (!meta) return null;
  const f = meta.freshness;
  return (
    <div data-tour="freshness-bar" className="bg-plane border-b border-[var(--border)] text-sm text-ink2">
      <div className="max-w-6xl mx-auto px-4 py-2 flex flex-wrap gap-x-5 gap-y-1">
        <span className="font-semibold text-ink">{t("freshness.title")}:</span>
        <span>{t("freshness.groundwater", { date: fmtDate(f.groundwater_last_reading, lang) })}
          <span className="text-muted"> ({t("freshness.groundwater_note")})</span></span>
        <span>{t("freshness.rain", { date: fmtDate(f.rain_as_of, lang) })}
          {f.rain_provisional_from && <span className="text-muted"> ({t("freshness.rain_prov", { date: fmtDate(f.rain_provisional_from, lang) })})</span>}</span>
        <span>{t("freshness.soil", { date: fmtDate(f.soil_as_of, lang) })}</span>
        <span>{meta.model.operational_forecast ? t("freshness.forecast_issued", { date: fmtDate(meta.model.last_issue_date, lang) })
                                               : t("freshness.forecast_none")}</span>
      </div>
    </div>
  );
}
