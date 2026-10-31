import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ArrowLeft, CloudRain, Droplets, Factory, Flag as FlagIcon, Info, LineChart, Target, Waves } from "lucide-react";
import { CategoryBadge, DirMark, ErrorBox, Loading, ReliabilityBadge, Tile } from "../components/Bits";
import { BacktestChart } from "../components/Charts";
import RangeStrip, { Sparkline, type StripRow } from "../components/RangeStrip";
import { useApp } from "../context";
import { districtName } from "../i18n";
import { api, useApi } from "../lib/api";
import { changeDir, fmtDate, fmtNum } from "../lib/format";
import { HORIZONS, type Horizon, type Lang } from "../lib/types";

interface Station { code: string; district: string; first: string; last: string; n_days: number }

const SKELETON = ["md:col-span-3 xl:col-span-3 h-72", "md:col-span-3 xl:col-span-5 h-72", "md:col-span-6 xl:col-span-4 h-72",
                  "md:col-span-6 xl:col-span-8 h-[28rem]", "md:col-span-6 xl:col-span-4 h-[28rem]"];

export default function GovDistrictPage() {
  const { slug = "" } = useParams();
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const { setVisible, meta } = useApp();
  const ds = useApi(() => api.district(slug), slug);
  const hist = useApi(() => api.history(slug), `h-${slug}`);
  const st = useApi(() => fetch("/api/geo/stations").then((r) => r.json() as Promise<{ stations: Station[] }>), "stations");
  const [h, setH] = useState<Horizon>("t+30");

  useEffect(() => {
    const d = ds.data;
    if (d) setVisible({ page: "government district", district: d.district, status: d.status, groundwater: d.groundwater,
                        forecast: d.forecast, reliability: d.reliability, extraction: d.extraction, flags: d.flags.map((f) => f.key) });
  }, [ds.data, setVisible]);

  if (ds.loading) return <Loading tiles={SKELETON} />;
  if (!ds.data) return <ErrorBox />;
  const d = ds.data;
  const g = d.groundwater;
  const wells = (st.data?.stations ?? []).filter((s) => s.district === d.district);
  const period = meta?.model.backtest_period;
  const fc = d.forecast;
  const strip: StripRow[] = fc ? HORIZONS.filter((hz) => fc.horizons[hz]).map((hz) => ({
    key: hz, label: t(`govd.h.${hz}`), sub: fmtDate(fc.horizons[hz]!.target_date, lang), f: fc.horizons[hz]! })) : [];
  const spark = (hist.data?.observed_weekly ?? []).slice(-26).map(([, v]) => v);
  const r = d.rain, so = d.soil;
  const rain = r && (
    <Tile title={t("farmer.rain_title")} icon={CloudRain}>
      {r.season.pct_of_usual !== null && (
        <div className="flex items-baseline gap-2 flex-wrap">
          <span className="text-3xl font-semibold tracking-tight tabnum">{fmtNum(r.season.pct_of_usual, 0, lang)}%</span>
          <span className="text-sm text-ink2">{t("farmer.of_usual_rain", { date: fmtDate(r.season.from, lang) })}</span>
        </div>
      )}
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2.5 mt-3 text-sm">
        <div><dt className="text-muted">{t("farmer.rain_since", { date: fmtDate(r.season.from, lang) })}</dt>
          <dd className="font-semibold tabnum">{fmtNum(r.season.mm, 0, lang)} mm</dd></div>
        <div><dt className="text-muted">{t("farmer.last_30d")}</dt>
          <dd className="font-semibold tabnum">{fmtNum(r.last_30d.mm, 0, lang)} mm <span className="font-normal text-ink2">· {t("farmer.pct_of_usual", { pct: fmtNum(r.last_30d.pct_of_usual, 0, lang) })}</span></dd></div>
        {so?.sm_l1.vs_usual && <div className="col-span-2"><dt className="text-muted">{t("farmer.soil_title", { date: fmtDate(so.as_of, lang) })}</dt>
          <dd>{t(`farmer.soil_label.${so.sm_l1.vs_usual}`)} <span className="text-ink2 tabnum">({fmtNum(so.sm_l1.pct, 0, lang)}% / {fmtNum(so.sm_l1.usual_pct, 0, lang)}%)</span></dd></div>}
      </dl>
      {r.provisional_from && <p className="text-xs text-muted mt-3">{t("gov.need.provisional", { date: fmtDate(r.provisional_from, lang) })}</p>}
    </Tile>
  );
  const fmtP = (p: Record<string, unknown>) => Object.fromEntries(Object.entries(p).map(([k, v]) => [k, typeof v === "number" ? fmtNum(v, k.endsWith("_m") ? 1 : 0, lang) : v]));

  const flags = (
    <Tile title={t("govd.flags_title")} icon={FlagIcon}>
      {d.flags.length ? (
        <ul className="space-y-2.5">
          {d.flags.map((f) => (
            <li key={f.key} className="flex items-start justify-between gap-3 text-[0.95rem]">
              <span>{t(`gov.flag.${f.key}`, fmtP(f.params))}<span className="block text-xs text-muted">{t(`gov.action.${f.action}`)}</span></span>
              <span className="chip tabnum shrink-0">+{f.weight}</span>
            </li>
          ))}
        </ul>
      ) : <p className="text-ink2">{t("govd.no_flags")}</p>}
    </Tile>
  );

  return (
    <div className="space-y-5">
      <Link to="/gov" className="inline-flex items-center gap-1.5 min-h-[44px] text-ink2 hover:text-ink">
        <ArrowLeft size={16} aria-hidden="true" />{t("govd.back")}</Link>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <h1 className="text-[1.75rem] leading-tight font-semibold tracking-tight">{districtName(d.district, lang)}</h1>
        <span className="chip">{t(`govd.status.${d.status}`)}</span>
        {d.monitoring.tier && <span className="text-sm text-muted tabnum">{t("govd.tier", { tier: d.monitoring.tier })} · {d.monitoring.wells} {t("gov.legend.wells")}
          {d.monitoring.completeness_pct !== null && ` · ${fmtNum(d.monitoring.completeness_pct, 0, lang)}%`}</span>}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-6 xl:grid-cols-12 gap-4">
        {d.status === "forecast" && g ? (<>
          <Tile title={t("farmer.water_now_title")} icon={Droplets} className="md:col-span-3 xl:col-span-3" bodyClass="flex flex-col">
            <div className="flex items-baseline gap-2">
              <span className="text-5xl font-semibold tracking-tight tabnum">{fmtNum(g.depth_m, 2, lang)}</span>
              <span className="text-ink2">{t("farmer.chart_axis")}</span>
            </div>
            <p className="text-sm text-muted mt-1">{t("farmer.week_avg", { date: fmtDate(g.last_reading, lang) })}</p>
            {fc && Math.abs(fc.level_now_m - g.level_m) >= 0.2 && (
              <p className="text-sm text-ink2 mt-1">{t("farmer.last_day", { date: fmtDate(fc.issue_date, lang), v: fmtNum(-fc.level_now_m, 2, lang) })}</p>
            )}
            {spark.length > 1 && (
              <figure className="mt-4 flex-1 flex flex-col min-h-[80px] max-h-[200px]">
                <Sparkline points={spark} fill />
                <figcaption className="text-xs text-muted mt-1">{t("farmer.spark_caption")}</figcaption>
              </figure>
            )}
            <ul className="mt-4 space-y-2.5 text-[0.95rem]">
              {g.vs_usual && <li className="flex items-center gap-2"><DirMark dir={g.vs_usual === "deeper" ? "fall" : g.vs_usual === "shallower" ? "rise" : "steady"} />
                <span>{t(`farmer.usual.${g.vs_usual}`)} <span className="tabnum text-ink2">({g.vs_usual_m! > 0 ? "+" : ""}{fmtNum(g.vs_usual_m, 2, lang)} {t("common.m")})</span></span></li>}
              {changeDir(g.change_30d_m) && <li className="flex items-center gap-2"><DirMark dir={changeDir(g.change_30d_m)} />
                <span>{t(`farmer.trend30.${changeDir(g.change_30d_m)}`, { v: fmtNum(Math.abs(g.change_30d_m ?? 0), 2, lang) })}</span></li>}
              {changeDir(g.change_1y_m, 0.2) && <li className="flex items-center gap-2"><DirMark dir={changeDir(g.change_1y_m, 0.2)} />
                <span>{t(`farmer.trend1y.${changeDir(g.change_1y_m, 0.2)}`, { v: fmtNum(Math.abs(g.change_1y_m ?? 0), 2, lang) })}</span></li>}
            </ul>
            {g.usual_years.length > 0 && <p className="text-xs text-muted pt-4">{t("farmer.usual_note", { years: g.usual_years.join(", ") })}</p>}
          </Tile>

          <Tile title={t("govd.latest_title")} icon={LineChart} className="md:col-span-3 xl:col-span-5">
            {fc ? (<>
              <p className="text-sm text-ink2 mb-4">{t("farmer.forecast_start", { date: fmtDate(fc.issue_date, lang), v: fmtNum(-fc.level_now_m, 2, lang) })}</p>
              <RangeStrip rows={strip} now={fc.level_now_m} />
            </>) : <p className="text-ink2">{d.forecast_unavailable && t(`farmer.fc_reason.${d.forecast_unavailable.code}`, { share: d.forecast_unavailable.share_pct })}</p>}
          </Tile>

          <Tile title={t("govd.metrics_title")} icon={Target} tour="gov-metrics" className="md:col-span-6 xl:col-span-4" bodyClass="flex flex-col">
            {period && <p className="text-xs text-muted mb-3">{t("govd.metrics_period", { from: fmtDate(period[0], lang), to: fmtDate(period[1], lang) })}</p>}
            <table className="w-full text-sm tabnum">
              <thead className="text-ink2"><tr><th className="text-left font-normal" />{HORIZONS.map((hz) => <th key={hz} className="text-right font-semibold pb-1.5">{t(`govd.h.${hz}`)}</th>)}</tr></thead>
              <tbody>
                {(["mase", "vs_sn", "coverage_pct"] as const).map((k) => (
                  <tr key={k} className="border-t border-[var(--gridline)]"><td className="py-2 pr-2 text-ink2">{t(`govd.metrics.${k === "coverage_pct" ? "coverage" : k}`)}</td>
                    {HORIZONS.map((hz) => { const m = d.reliability?.[hz]; return <td key={hz} className="py-2 text-right font-semibold">{m ? (k === "coverage_pct" ? `${fmtNum(m[k], 0, lang)}%` : fmtNum(m[k], 2, lang)) : "–"}</td>; })}</tr>
                ))}
                <tr className="border-t border-[var(--gridline)]"><td className="py-2 pr-2 text-ink2">{t("gov.col.rel")}</td>
                  {HORIZONS.map((hz) => <td key={hz} className="py-2 text-right"><ReliabilityBadge label={d.reliability?.[hz]?.label} /></td>)}</tr>
              </tbody>
            </table>
            <p className="text-xs text-muted mt-auto pt-3">{t("govd.metrics.note")}</p>
          </Tile>

          <Tile title={t("govd.history_title")} icon={LineChart} tour="gov-history-chart" className="md:col-span-6 xl:col-span-8"
            actions={
              <div role="radiogroup" aria-label={t("govd.horizon_label")} className="seg">
                {HORIZONS.map((hz) => <button key={hz} role="radio" aria-checked={h === hz} onClick={() => setH(hz)}>{t(`govd.h.${hz}`)}</button>)}
              </div>}>
            <p className="text-sm text-muted mb-3">{t("govd.history_hint")}</p>
            {hist.data ? <BacktestChart history={hist.data} horizon={h} /> : <div className="skeleton h-[340px]" />}
          </Tile>
          <div className="md:col-span-6 xl:col-span-4 grid md:grid-cols-2 xl:grid-cols-1 gap-4 content-start">{flags}{rain}</div>
        </>) : (<>
          <div role="note" className="tile p-5 sm:p-6 md:col-span-6 xl:col-span-8 flex gap-3 fade-in">
            <Info size={18} aria-hidden="true" className="mt-0.5 shrink-0 text-ink2" />
            <div>
              <p className="font-semibold">{t(d.status === "no_data" ? "farmer.no_groundwater_title" : "govd.status.insufficient_data")}</p>
              <p className="text-ink2">{d.status === "no_data" ? t("farmer.no_groundwater_text", { district: districtName(d.district, lang, false) })
                                                              : t("farmer.insufficient_text", { district: districtName(d.district, lang, false) })}</p>
            </div>
          </div>
          <div className="md:col-span-6 xl:col-span-4 grid md:grid-cols-2 xl:grid-cols-1 gap-4 content-start">{flags}{rain}</div>
        </>)}

        {d.extraction && (
          <Tile title={t("govd.extraction_title")} icon={Factory} tour="gov-extraction" className="md:col-span-6 xl:col-span-5">
            <table className="w-full text-sm tabnum mb-4">
              <thead className="text-ink2"><tr><th className="text-left font-semibold pb-1.5">{t("govd.year")}</th>
                <th className="text-right font-semibold pb-1.5">{t("govd.stage")}</th><th className="text-left font-semibold pb-1.5 pl-4">{t("govd.category")}</th></tr></thead>
              <tbody>{d.extraction.history.map((x) => (
                <tr key={x.year} className="border-t border-[var(--gridline)]"><td className="py-2">{x.year}</td>
                  <td className="py-2 text-right font-semibold">{fmtNum(x.stage_pct, 1, lang)}%</td><td className="py-2 pl-4"><CategoryBadge category={x.category} /></td></tr>))}</tbody>
            </table>
            {d.extraction.use_split_pct && (<>
              <p className="text-sm font-semibold mb-2">{t("govd.use_split", { year: d.extraction.latest_year })}</p>
              <ul className="text-sm space-y-2">{(["irrigation", "domestic", "industrial"] as const).map((k) => (
                <li key={k} className="grid grid-cols-[7rem_1fr_3rem] items-center gap-3"><span>{t(`govd.use.${k}`)}</span>
                  <span className="h-2 rounded-full bg-[var(--surface-2)] overflow-hidden" aria-hidden="true">
                    <span className="block h-full rounded-full" style={{ width: `${Math.max(1, d.extraction!.use_split_pct![k])}%`, background: "var(--series-1)" }} /></span>
                  <span className="tabnum text-right">{fmtNum(d.extraction!.use_split_pct![k], 0, lang)}%</span></li>))}</ul>
            </>)}
          </Tile>
        )}
        <Tile title={t("govd.wells_title")} icon={Waves} tour="gov-wells" className={`md:col-span-6 ${d.extraction ? "xl:col-span-7" : "xl:col-span-12"}`}>
          {wells.length ? (
            <div className="max-h-72 overflow-auto -mx-1 px-1"><table className="data-table w-full text-sm tabnum">
              <thead className="text-ink2"><tr><th className="text-left font-semibold py-1.5">{t("govd.well_code")}</th>
                <th className="text-left font-semibold py-1.5">{t("govd.well_period")}</th></tr></thead>
              <tbody>{wells.map((w) => (
                <tr key={w.code} className="border-t border-[var(--gridline)]"><td className="py-2 pr-3 font-semibold">{w.code}</td>
                  <td className="py-2 text-ink2">{fmtDate(w.first, lang)} – {fmtDate(w.last, lang)} <span className="text-muted">({w.n_days})</span></td></tr>))}</tbody>
            </table></div>
          ) : <p className="text-ink2">{t("govd.wells_none")}</p>}
        </Tile>
      </div>
    </div>
  );
}
