import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Card, CategoryBadge, ErrorBox, Loading, ReliabilityBadge } from "../components/Bits";
import { BacktestChart } from "../components/Charts";
import { useApp } from "../context";
import { districtName } from "../i18n";
import { api, useApi } from "../lib/api";
import { depth, fmtDate, fmtNum } from "../lib/format";
import { HORIZONS, type Horizon, type Lang } from "../lib/types";

interface Station { code: string; district: string; first: string; last: string; n_days: number }

export default function GovDistrictPage() {
  const { slug = "" } = useParams();
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const { setVisible } = useApp();
  const ds = useApi(() => api.district(slug), slug);
  const hist = useApi(() => api.history(slug), `h-${slug}`);
  const meta = useApi(api.meta, "meta-d");
  const st = useApi(() => fetch("/api/geo/stations").then((r) => r.json() as Promise<{ stations: Station[] }>), "stations");
  const [h, setH] = useState<Horizon>("t+30");

  useEffect(() => {
    const d = ds.data;
    if (d) setVisible({ page: "government district", district: d.district, status: d.status, groundwater: d.groundwater,
                        forecast: d.forecast, reliability: d.reliability, extraction: d.extraction, flags: d.flags.map((f) => f.key) });
  }, [ds.data, setVisible]);

  if (ds.loading) return <Loading />;
  if (!ds.data) return <ErrorBox />;
  const d = ds.data;
  const wells = (st.data?.stations ?? []).filter((s) => s.district === d.district);
  const period = meta.data?.model.backtest_period;

  return (
    <div className="space-y-4">
      <Link to="/gov" className="underline inline-flex min-h-[44px] items-center">← {t("govd.back")}</Link>
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-2xl font-semibold">{districtName(d.district, lang)}</h1>
        <span className="text-ink2">{t(`govd.status.${d.status}`)}</span>
        {d.monitoring.tier && <span className="text-sm text-muted">{t("govd.tier", { tier: d.monitoring.tier })} · {d.monitoring.wells} {t("gov.legend.wells")}
          {d.monitoring.completeness_pct !== null && ` · ${fmtNum(d.monitoring.completeness_pct, 0, lang)}%`}</span>}
      </div>

      <Card title={t("govd.flags_title")}>
        {d.flags.length ? <ul className="flex flex-wrap gap-2">{d.flags.map((f) => (
          <li key={f.key} className="rounded-full bg-plane border border-[var(--border)] px-3 py-1 text-sm">
            {t(`gov.flag.${f.key}`, Object.fromEntries(Object.entries(f.params).map(([k, v]) => [k, typeof v === "number" ? fmtNum(v, k.endsWith("_m") ? 1 : 0, lang) : v])))}
            <span className="text-muted"> · {t(`gov.action.${f.action}`)} (+{f.weight})</span></li>))}</ul>
          : <p className="text-ink2">{t("govd.no_flags")}</p>}
      </Card>

      {d.status === "forecast" && (
        <div className="grid gap-4 lg:grid-cols-3">
          <Card title={t("farmer.water_now_title")}>
            {d.groundwater && (<>
              <p className="text-3xl font-semibold tabnum">{fmtNum(d.groundwater.depth_m, 2, lang)} {t("common.m")}</p>
              <p className="text-ink2">{t("farmer.latest_reading", { date: fmtDate(d.groundwater.last_reading, lang) })}</p>
              {d.groundwater.vs_usual && <p className="mt-2">{t(`farmer.usual.${d.groundwater.vs_usual}`)} ({d.groundwater.vs_usual_m! > 0 ? "+" : ""}{fmtNum(d.groundwater.vs_usual_m, 2, lang)} {t("common.m")})</p>}
            </>)}
          </Card>
          <Card title={t("govd.latest_title")}>
            {d.forecast ? (
              <table className="w-full text-sm tabnum"><tbody>
                {HORIZONS.map((hz) => { const x = d.forecast!.horizons[hz]; return x && (
                  <tr key={hz} className="border-b border-[var(--gridline)]"><td className="py-1 pr-2">{t(`govd.h.${hz}`)}</td>
                    <td className="py-1 pr-2">{fmtNum(depth(x.q50), 2, lang)} {t("common.m")}</td>
                    <td className="py-1 text-ink2">{fmtNum(depth(x.q90), 1, lang)}–{fmtNum(depth(x.q10), 1, lang)} {t("common.m")}</td></tr>); })}
              </tbody></table>
            ) : <p className="text-ink2">{d.forecast_unavailable && t(`farmer.fc_reason.${d.forecast_unavailable.code}`, { share: d.forecast_unavailable.share_pct })}</p>}
          </Card>
          <Card title={t("govd.metrics_title")} tour="gov-metrics">
            {period && <p className="text-xs text-muted mb-2">{t("govd.metrics_period", { from: fmtDate(period[0], lang), to: fmtDate(period[1], lang) })}</p>}
            <table className="w-full text-sm tabnum">
              <thead className="text-ink2"><tr><th className="text-left"></th>{HORIZONS.map((hz) => <th key={hz} className="text-left">{t(`govd.h.${hz}`)}</th>)}</tr></thead>
              <tbody>
                {(["mase", "vs_sn", "coverage_pct"] as const).map((k) => (
                  <tr key={k} className="border-t border-[var(--gridline)]"><td className="py-1 pr-2 w-[40%]">{t(`govd.metrics.${k === "coverage_pct" ? "coverage" : k}`)}</td>
                    {HORIZONS.map((hz) => { const m = d.reliability?.[hz]; return <td key={hz} className="py-1 pr-2">{m ? (k === "coverage_pct" ? `${fmtNum(m[k], 0, lang)}%` : fmtNum(m[k], 2, lang)) : "–"}</td>; })}</tr>
                ))}
                <tr className="border-t border-[var(--gridline)]"><td className="py-1 pr-2">{t("gov.col.rel")}</td>
                  {HORIZONS.map((hz) => <td key={hz} className="py-1"><ReliabilityBadge label={d.reliability?.[hz]?.label} /></td>)}</tr>
              </tbody>
            </table>
            <p className="text-xs text-muted mt-2">{t("govd.metrics.note")}</p>
          </Card>
        </div>
      )}

      {d.status === "forecast" && hist.data && (
        <Card title={t("govd.history_title")} tour="gov-history-chart">
          <p className="text-sm text-ink2 mb-2">{t("govd.history_hint")}</p>
          <div role="radiogroup" aria-label={t("govd.horizon_label")} className="flex gap-2 mb-3">
            {HORIZONS.map((hz) => (
              <button key={hz} role="radio" aria-checked={h === hz} onClick={() => setH(hz)}
                className={`px-3 py-2 min-h-[44px] rounded-lg border text-sm ${h === hz ? "bg-water text-white border-transparent" : "border-[var(--border)]"}`}>
                {t(`govd.h.${hz}`)}</button>
            ))}
          </div>
          <BacktestChart history={hist.data} horizon={h} />
        </Card>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        {d.extraction && (
          <Card title={t("govd.extraction_title")} tour="gov-extraction">
            <table className="w-full text-sm tabnum mb-3">
              <thead className="text-ink2"><tr><th className="text-left">{t("govd.year")}</th><th className="text-left">{t("govd.stage")}</th><th className="text-left">{t("govd.category")}</th></tr></thead>
              <tbody>{d.extraction.history.map((x) => (
                <tr key={x.year} className="border-t border-[var(--gridline)]"><td className="py-1">{x.year}</td><td>{fmtNum(x.stage_pct, 1, lang)}%</td><td><CategoryBadge category={x.category} /></td></tr>))}</tbody>
            </table>
            {d.extraction.use_split_pct && (<>
              <p className="font-semibold text-sm mb-1">{t("govd.use_split", { year: d.extraction.latest_year })}</p>
              <ul className="text-sm space-y-1">{(["irrigation", "domestic", "industrial"] as const).map((k) => (
                <li key={k} className="flex items-center gap-2"><span className="w-28 shrink-0">{t(`govd.use.${k}`)}</span>
                  <span className="h-3 rounded-sm" style={{ width: `${Math.max(2, d.extraction!.use_split_pct![k])}%`, background: "var(--series-1)" }} aria-hidden="true" />
                  <span className="tabnum">{fmtNum(d.extraction!.use_split_pct![k], 0, lang)}%</span></li>))}</ul>
            </>)}
          </Card>
        )}
        <Card title={t("govd.wells_title")} tour="gov-wells">
          {wells.length ? (
            <div className="max-h-64 overflow-auto"><table className="w-full text-sm tabnum">
              <thead className="text-ink2"><tr><th className="text-left">{t("govd.well_code")}</th><th className="text-left">{t("govd.well_period")}</th></tr></thead>
              <tbody>{wells.map((w) => (
                <tr key={w.code} className="border-t border-[var(--gridline)]"><td className="py-1 pr-2">{w.code}</td>
                  <td className="py-1">{fmtDate(w.first, lang)} – {fmtDate(w.last, lang)} ({w.n_days})</td></tr>))}</tbody>
            </table></div>
          ) : <p className="text-ink2">{t("govd.wells_none")}</p>}
        </Card>
      </div>
    </div>
  );
}
