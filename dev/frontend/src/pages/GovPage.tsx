import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { CategoryBadge, ErrorBox, Loading, ReliabilityBadge } from "../components/Bits";
import MapView, { MapLegend, type MapLayer } from "../components/MapView";
import { useApp } from "../context";
import { districtName } from "../i18n";
import { api, useApi } from "../lib/api";
import { fmtDate, fmtNum } from "../lib/format";
import type { Flag, Lang, PriorityRow } from "../lib/types";

const LAYERS: MapLayer[] = ["vs_usual", "forecast", "extraction", "wells", "reliability"];
type SortKey = "score" | "district" | "stage_pct" | "depth_m" | "vs_usual_m" | "forecast_change_t30_m" | "wells";

function Kpi({ value, label, sub }: { value: string; label: string; sub?: string }) {
  return (
    <div className="card p-4">
      <div className="text-3xl font-semibold">{value}</div>
      <div className="text-ink2">{label}</div>
      {sub && <div className="text-sm text-muted">{sub}</div>}
    </div>
  );
}

function FlagChip({ f }: { f: Flag }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const p = Object.fromEntries(Object.entries(f.params).map(([k, v]) => [k, typeof v === "number" ? fmtNum(v, k.endsWith("_m") ? 1 : 0, lang) : v]));
  return <span className="inline-block rounded-full bg-plane border border-[var(--border)] px-2 py-0.5 text-xs mr-1 mb-1">{t(`gov.flag.${f.key}`, p)}</span>;
}

export default function GovPage() {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const { setRole, role, setVisible } = useApp();
  const ov = useApi(api.overview, "overview");
  const list = useApi(api.districts, "districts");
  const pr = useApi(api.priorities, "priorities");
  const geo = useApi(api.geo, "geo");
  const [layer, setLayer] = useState<MapLayer>("vs_usual");
  const [action, setAction] = useState<string>("all");
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "score", desc: true });

  useEffect(() => { if (role !== "government") setRole("government"); }, [role, setRole]);
  useEffect(() => {
    if (!ov.data || !pr.data) return;
    setVisible({ page: "government overview", counts: ov.data.counts, extraction_categories: ov.data.extraction_categories,
                 flag_counts: ov.data.flag_counts, map_layer: layer,
                 top_priorities: pr.data.slice(0, 6).map((r) => ({ district: r.district, score: r.score, reasons: r.flags.map((f) => f.key) })) });
  }, [ov.data, pr.data, layer, setVisible]);

  const rows = useMemo(() => {
    const r = (pr.data ?? []).filter((x) => action === "all" || x.actions.includes(action));
    const k = sort.key;
    return [...r].sort((a, b) => {
      const av = k === "district" ? districtName(a.district, lang, false) : (a[k] as number | null);
      const bv = k === "district" ? districtName(b.district, lang, false) : (b[k] as number | null);
      if (av === null || av === undefined) return 1;
      if (bv === null || bv === undefined) return -1;
      const c = typeof av === "string" ? av.localeCompare(bv as string, lang) : (av as number) - (bv as number);
      return sort.desc ? -c : c;
    });
  }, [pr.data, action, sort, lang]);

  if (ov.loading || list.loading || pr.loading) return <Loading />;
  if (!ov.data || !list.data || !pr.data) return <ErrorBox />;
  const o = ov.data;
  const semi = (o.extraction_categories["Semi-Critical"] ?? 0) + (o.extraction_categories["Critical"] ?? 0) + (o.extraction_categories["Over-Exploited"] ?? 0);
  const names = (flag: string) => pr.data!.filter((r) => r.flags.some((f) => f.key === flag)).map((r) => districtName(r.district, lang, false));
  const noData = pr.data.filter((r) => r.status !== "forecast").map((r) => districtName(r.district, lang, false));

  const th = (key: SortKey, label: string) => (
    <th scope="col" className="py-2 pr-3 text-left font-semibold whitespace-nowrap"
      aria-sort={sort.key === key ? (sort.desc ? "descending" : "ascending") : "none"}>
      <button className="underline-offset-2 hover:underline min-h-[36px]" onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : true }))}>
        {label}{sort.key === key ? (sort.desc ? " ▼" : " ▲") : ""}
      </button>
    </th>
  );

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold">{t("gov.title")}</h1>
      <div data-tour="kpi-row" className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <Kpi value={String(o.counts.districts)} label={t("gov.kpi.districts")} sub={`${o.counts.with_forecast} ${t("gov.kpi.with_forecast")}`} />
        <Kpi value={String(o.counts.wells_used)} label={t("gov.kpi.wells")} />
        <Kpi value={String(semi)} label={t("gov.kpi.semi_or_worse")} sub={`CGWB ${o.freshness.extraction_year}`} />
        <Kpi value={o.model.operational_forecast ? String(o.flag_counts.forecast_decline ?? 0) : "–"} label={t("gov.kpi.falling")}
          sub={o.model.operational_forecast ? fmtDate(o.model.last_issue_date, lang) : t("freshness.forecast_none")} />
        <Kpi value={String(o.counts.no_data + o.counts.insufficient_data)} label={t("gov.kpi.no_data")} />
      </div>

      <section className="card p-4 sm:p-5" aria-labelledby="map-h">
        <h2 id="map-h" className="text-lg font-semibold mb-2">{t("gov.map_title")}</h2>
        <fieldset data-tour="map-layers" className="mb-3">
          <legend className="text-sm text-ink2 mb-1">{t("gov.map_layers")}</legend>
          <div className="flex flex-wrap gap-2">
            {LAYERS.map((l) => (
              <label key={l} className={`cursor-pointer rounded-lg border px-3 py-2 min-h-[44px] inline-flex items-center text-sm
                ${layer === l ? "bg-water text-white border-transparent" : "border-[var(--border)] hover:bg-plane"}`}>
                <input type="radio" name="layer" value={l} checked={layer === l} onChange={() => setLayer(l)} className="sr-only" />
                {t(`gov.layer.${l}`)}
              </label>
            ))}
          </div>
        </fieldset>
        {layer === "forecast" && !o.model.operational_forecast && <p className="text-sm text-ink2 mb-2">{t("gov.map_no_forecast")}</p>}
        <div className="mb-2"><MapLegend layer={layer} /></div>
        {geo.data ? <MapView geo={geo.data} districts={list.data} layer={layer} /> : <Loading />}
        <p className="text-sm text-muted mt-2">{t("gov.map_hint")}</p>
      </section>

      <section data-tour="priority-table" className="card p-4 sm:p-5" aria-labelledby="pt-h">
        <div className="flex flex-wrap items-end justify-between gap-3 mb-3">
          <div>
            <h2 id="pt-h" className="text-lg font-semibold">{t("gov.table_title")}</h2>
            <p className="text-sm text-ink2">{t("gov.table_hint")}</p>
          </div>
          <div className="flex flex-wrap gap-2 items-center">
            <label className="text-sm" htmlFor="action-filter">{t("gov.col.action")}</label>
            <select id="action-filter" value={action} onChange={(e) => setAction(e.target.value)}
              className="rounded-lg border border-[var(--border)] bg-surface px-2 py-2 min-h-[44px]">
              <option value="all">{t("gov.filter_all")}</option>
              {["improve_data", "manage_demand", "monitor"].map((a) => <option key={a} value={a}>{t(`gov.action.${a}`)}</option>)}
            </select>
            <a data-tour="download-csv" href="/api/export/priorities.csv" download
              className="px-3 py-2 min-h-[44px] inline-flex items-center rounded-lg border border-[var(--border)] hover:bg-plane text-sm">
              ⬇ {t("gov.download")}</a>
          </div>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm tabnum">
            <thead className="text-ink2 border-b border-[var(--gridline)]">
              <tr>
                {th("district", t("gov.col.district"))}{th("score", t("gov.col.score"))}
                <th scope="col" className="py-2 pr-3 text-left font-semibold">{t("gov.col.reasons")}</th>
                {th("stage_pct", t("gov.col.stage"))}{th("depth_m", t("gov.col.depth"))}{th("vs_usual_m", t("gov.col.vs_usual"))}
                {th("forecast_change_t30_m", t("gov.col.fc30"))}{th("wells", t("gov.col.wells"))}
                <th scope="col" className="py-2 pr-3 text-left font-semibold">{t("gov.col.rel")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r: PriorityRow) => (
                <tr key={r.slug} className="border-b border-[var(--gridline)] align-top">
                  <td className="py-2 pr-3"><Link className="underline" to={`/gov/district/${r.slug}`}>{districtName(r.district, lang, false)}</Link></td>
                  <td className="py-2 pr-3 font-semibold">{r.score}</td>
                  <td className="py-2 pr-3 min-w-[220px]">{r.flags.map((f) => <FlagChip key={f.key} f={f} />)}</td>
                  <td className="py-2 pr-3 whitespace-nowrap">{r.stage_pct !== null ? `${fmtNum(r.stage_pct, 0, lang)}%` : "–"} <CategoryBadge category={r.category} /></td>
                  <td className="py-2 pr-3">{fmtNum(r.depth_m, 1, lang)}</td>
                  <td className="py-2 pr-3">{r.vs_usual_m !== null ? `${r.vs_usual_m > 0 ? "+" : ""}${fmtNum(r.vs_usual_m, 1, lang)}` : "–"}</td>
                  <td className="py-2 pr-3">{r.forecast_change_t30_m !== null ? `${r.forecast_change_t30_m > 0 ? "+" : ""}${fmtNum(r.forecast_change_t30_m, 1, lang)}` : "–"}</td>
                  <td className="py-2 pr-3">{r.wells ?? 0}</td>
                  <td className="py-2 pr-3"><ReliabilityBadge label={r.reliability_t30} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-muted mt-2">{t("gov.method_note")}</p>
      </section>

      <section data-tour="needs-panel" className="card p-4 sm:p-5" aria-labelledby="needs-h">
        <h2 id="needs-h" className="text-lg font-semibold mb-3">{t("gov.needs_title")}</h2>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-semibold mb-2">✓ {t("gov.have_title")}</h3>
            <ul className="list-disc pl-5 space-y-1 text-ink2">
              <li>{t("gov.have.wells", { n: o.counts.wells_used })}</li>
              <li>{t("gov.have.forecast", { n: o.counts.with_forecast, from: fmtDate(o.model.backtest_period[0], lang), to: fmtDate(o.model.backtest_period[1], lang) })}</li>
              <li>{t("gov.have.rain", { date: fmtDate(o.freshness.rain_as_of, lang) })}</li>
              <li>{t("gov.have.extraction", { year: o.freshness.extraction_year })}</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold mb-2">→ {t("gov.need_title")}</h3>
            <ul className="list-disc pl-5 space-y-1 text-ink2">
              <li>{t("gov.need.no_data", { n: noData.length, list: noData.join(", ") })}</li>
              {names("single_well").length > 0 && <li>{t("gov.need.single", { list: names("single_well").join(", ") })}</li>}
              <li>{t("gov.need.fresh", { date: fmtDate(o.freshness.groundwater_last_reading, lang) })}</li>
              {o.freshness.rain_provisional_from && <li>{t("gov.need.provisional", { date: fmtDate(o.freshness.rain_provisional_from, lang) })}</li>}
              {names("reliability_limited").length > 0 && <li>{t("gov.need.limited", { list: names("reliability_limited").join(", ") })}</li>}
              {!o.model.operational_forecast && <li>{t("gov.need.operational")}</li>}
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
