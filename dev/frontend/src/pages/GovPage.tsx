import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { ArrowDown, ArrowRight, ArrowUp, ArrowUpDown, CircleCheck, CircleDashed, ClipboardList, Download, Gauge, Layers,
         ListOrdered, MapPin, Network, TrendingDown, Waves, type LucideIcon } from "lucide-react";
import { CategoryBadge, ErrorBox, Loading, ReliabilityBadge, Tile } from "../components/Bits";
import MapView, { MapLegend, type MapLayer } from "../components/MapView";
import { useApp } from "../context";
import { districtName } from "../i18n";
import { api, useApi } from "../lib/api";
import { fmtDate, fmtNum } from "../lib/format";
import type { Flag, Lang, PriorityRow } from "../lib/types";

const LAYERS: MapLayer[] = ["vs_usual", "forecast", "extraction", "wells", "reliability"];
type SortKey = "score" | "district" | "stage_pct" | "depth_m" | "vs_usual_m" | "forecast_change_t30_m" | "wells";
const CATS = ["Safe", "Semi-Critical", "Critical", "Over-Exploited"] as const;
const CAT_COLOUR: Record<string, string> = { Safe: "var(--status-good)", "Semi-Critical": "var(--status-warning)",
  Critical: "var(--status-serious)", "Over-Exploited": "var(--status-critical)" };
const SKELETON = ["col-span-2 md:col-span-3 xl:col-span-3 h-36", "col-span-2 md:col-span-3 xl:col-span-3 h-36",
  "md:col-span-2 xl:col-span-2 h-36", "md:col-span-2 xl:col-span-2 h-36", "col-span-2 md:col-span-2 xl:col-span-2 h-36",
  "col-span-2 md:col-span-6 xl:col-span-8 h-[40rem]", "col-span-2 md:col-span-6 xl:col-span-4 h-[40rem]"];

const signed = (v: number | null | undefined, lang: Lang, digits = 1) =>
  v === null || v === undefined ? "–" : `${v > 0 ? "+" : ""}${fmtNum(v, digits, lang)}`;

function Kpi({ icon: Icon, value, label, sub, className = "", children }:
  { icon: LucideIcon; value: string; label: string; sub?: string; className?: string; children?: React.ReactNode }) {
  return (
    <div className={`tile p-5 flex flex-col fade-in ${className}`}>
      <div className="flex items-center gap-2 eyebrow"><Icon size={15} aria-hidden="true" className="text-muted" />{label}</div>
      <div className="text-4xl font-semibold tracking-tight tabnum mt-2">{value}</div>
      {sub && <div className="text-sm text-muted mt-0.5">{sub}</div>}
      {children && <div className="mt-auto pt-3">{children}</div>}
    </div>
  );
}

/** thin stacked bar with a 2px gap between parts; the counts are always written beside it */
function StackBar({ parts }: { parts: { key: string; n: number; colour: string; label: string }[] }) {
  const total = parts.reduce((a, p) => a + p.n, 0) || 1;
  const shown = parts.filter((p) => p.n > 0);
  return (
    <div>
      <div className="flex h-2 rounded-full overflow-hidden gap-[2px]" aria-hidden="true">
        {shown.map((p) => <div key={p.key} style={{ width: `${(p.n / total) * 100}%`, background: p.colour }} />)}
      </div>
      <ul className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-ink2">
        {shown.map((p) => (
          <li key={p.key} className="flex items-center gap-1.5">
            <span className="inline-block w-2 h-2 rounded-sm" style={{ background: p.colour }} aria-hidden="true" />
            <span className="tabnum font-semibold text-ink">{p.n}</span>{p.label}
          </li>
        ))}
      </ul>
    </div>
  );
}

function FlagChip({ f }: { f: Flag }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const p = Object.fromEntries(Object.entries(f.params).map(([k, v]) => [k, typeof v === "number" ? fmtNum(v, k.endsWith("_m") ? 1 : 0, lang) : v]));
  return <span className="chip !text-xs mr-1 mb-1">{t(`gov.flag.${f.key}`, p)}</span>;
}

function Glance({ r }: { r: PriorityRow | undefined }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  return (
    <Tile title={t("gov.glance_title")} icon={MapPin} tour="glance-panel" bodyClass="flex flex-col">
      {!r ? <p className="text-ink2">{t("gov.glance_hint")}</p> : (
        <div key={r.slug} className="fade-in flex flex-col flex-1">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-2xl font-semibold tracking-tight">{districtName(r.district, lang)}</h3>
            <span className="text-sm text-muted">{t(`govd.status.${r.status}`)}</span>
          </div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 mt-4 text-[0.95rem]">
            {r.status !== "forecast" && r.depth_m === null ? (
              <div className="col-span-2 rounded-xl bg-[var(--surface-2)] px-3 py-2 text-sm text-ink2">
                <dt className="sr-only">{t("gov.col.depth")}</dt><dd>{t("farmer.no_groundwater_title")}</dd></div>
            ) : (<>
            <div><dt className="text-xs text-muted">{t("gov.col.depth")}</dt><dd className="font-semibold tabnum">{fmtNum(r.depth_m, 1, lang)}</dd></div>
            <div><dt className="text-xs text-muted">{t("gov.col.vs_usual")}</dt><dd className="font-semibold tabnum">{signed(r.vs_usual_m, lang)}</dd></div>
            <div><dt className="text-xs text-muted">{t("gov.col.fc30")}</dt><dd className="font-semibold tabnum">{signed(r.forecast_change_t30_m, lang)}</dd></div>
            <div><dt className="text-xs text-muted">{t("gov.col.wells")}</dt><dd className="font-semibold tabnum">{r.wells ?? 0}</dd></div>
            </>)}
            <div className="col-span-2"><dt className="text-xs text-muted">{t("gov.col.stage")}</dt>
              <dd className="flex flex-wrap items-center gap-2"><span className="font-semibold tabnum">{r.stage_pct !== null ? `${fmtNum(r.stage_pct, 0, lang)}%` : "–"}</span>
                <CategoryBadge category={r.category} /></dd></div>
            {r.reliability_t30 && <div className="col-span-2"><dt className="text-xs text-muted">{t("gov.col.rel")}</dt>
              <dd className="mt-0.5"><ReliabilityBadge label={r.reliability_t30} /></dd></div>}
          </dl>
          {r.flags.length > 0 && (
            <div className="mt-4">
              <p className="text-xs text-muted mb-1.5">{t("gov.col.reasons")} · {t("gov.col.score")} {r.score}</p>
              <div>{r.flags.map((f) => <FlagChip key={f.key} f={f} />)}</div>
            </div>
          )}
          <div className="mt-auto pt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-muted">{t("gov.glance_hint")}</p>
            <Link to={`/gov/district/${r.slug}`} className="btn btn-primary">{t("gov.open_district")}<ArrowRight size={16} aria-hidden="true" /></Link>
          </div>
        </div>
      )}
    </Tile>
  );
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
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => { if (role !== "government") setRole("government"); }, [role, setRole]);
  const selected = picked ?? pr.data?.[0]?.slug ?? null;
  const sel = pr.data?.find((r) => r.slug === selected);
  useEffect(() => {
    if (!ov.data || !pr.data) return;
    setVisible({ page: "government overview", counts: ov.data.counts, extraction_categories: ov.data.extraction_categories,
                 flag_counts: ov.data.flag_counts, map_layer: layer, selected_district: sel?.district,
                 top_priorities: pr.data.slice(0, 6).map((r) => ({ district: r.district, score: r.score, reasons: r.flags.map((f) => f.key) })) });
  }, [ov.data, pr.data, layer, sel, setVisible]);

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

  if (ov.loading || list.loading || pr.loading) return <Loading tiles={SKELETON} />;
  if (!ov.data || !list.data || !pr.data) return <ErrorBox />;
  const o = ov.data;
  const cats = o.extraction_categories;
  const semi = (cats["Semi-Critical"] ?? 0) + (cats["Critical"] ?? 0) + (cats["Over-Exploited"] ?? 0);
  const names = (flag: string) => pr.data!.filter((r) => r.flags.some((f) => f.key === flag)).map((r) => districtName(r.district, lang, false));
  const noData = pr.data.filter((r) => r.status !== "forecast").map((r) => districtName(r.district, lang, false));

  const th = (key: SortKey, label: string, num = false) => {
    const on = sort.key === key;
    const Icon = on ? (sort.desc ? ArrowDown : ArrowUp) : ArrowUpDown;
    return (
      <th scope="col" className={`py-2.5 px-3 font-semibold whitespace-nowrap ${num ? "text-right" : "text-left"}`}
        aria-sort={on ? (sort.desc ? "descending" : "ascending") : "none"}>
        <button className={`inline-flex items-center gap-1 min-h-[36px] hover:text-ink ${num ? "flex-row-reverse" : ""}`}
          onClick={() => setSort((s) => ({ key, desc: s.key === key ? !s.desc : true }))}>
          {label}<Icon size={13} aria-hidden="true" className={on ? "text-ink" : "text-muted"} />
        </button>
      </th>
    );
  };

  return (
    <div className="space-y-5">
      <div>
        <h1 className="text-[1.75rem] leading-tight font-semibold tracking-tight">{t("gov.title")}</h1>
        <p className="text-sm text-muted mt-0.5">{t("gov.subtitle")}</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-6 xl:grid-cols-12 gap-4">
        <div data-tour="kpi-row" className="col-span-full grid grid-cols-2 md:grid-cols-6 xl:grid-cols-12 gap-4">
          <Kpi icon={Network} className="col-span-2 md:col-span-3 xl:col-span-3" value={String(o.counts.districts)} label={t("gov.kpi.districts")}>
            <StackBar parts={[
              { key: "f", n: o.counts.with_forecast, colour: "var(--series-1)", label: t("govd.status.forecast") },
              { key: "i", n: o.counts.insufficient_data, colour: "var(--band-2)", label: t("govd.status.insufficient_data") },
              { key: "n", n: o.counts.no_data, colour: "var(--nodata)", label: t("govd.status.no_data") }]} />
          </Kpi>
          <Kpi icon={Gauge} className="col-span-2 md:col-span-3 xl:col-span-3" value={String(semi)} label={t("gov.kpi.semi_or_worse")}
            sub={`CGWB ${o.freshness.extraction_year}`}>
            <StackBar parts={CATS.map((c) => ({ key: c, n: cats[c] ?? 0, colour: CAT_COLOUR[c], label: t(`cat.${c}`) }))} />
          </Kpi>
          <Kpi icon={Waves} className="md:col-span-2 xl:col-span-2" value={String(o.counts.wells_used)} label={t("gov.kpi.wells")} />
          <Kpi icon={TrendingDown} className="md:col-span-2 xl:col-span-2"
            value={o.model.operational_forecast ? String(o.flag_counts.forecast_decline ?? 0) : "–"} label={t("gov.kpi.falling")}
            sub={o.model.operational_forecast ? t("freshness.forecast_issued", { date: fmtDate(o.model.last_issue_date, lang) }) : t("freshness.forecast_none")} />
          <Kpi icon={CircleDashed} className="col-span-2 md:col-span-2 xl:col-span-2" value={String(o.counts.no_data + o.counts.insufficient_data)}
            label={t("gov.kpi.no_data")} />
        </div>

        <section className="tile p-5 sm:p-6 col-span-2 md:col-span-6 xl:col-span-8 flex flex-col min-w-0 fade-in" aria-labelledby="map-h">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
            <h2 id="map-h" className="eyebrow flex items-center gap-2"><Layers size={16} aria-hidden="true" className="text-muted" />{t("gov.map_title")}</h2>
            <fieldset data-tour="map-layers" className="min-w-0 max-w-full overflow-x-auto">
              <legend className="sr-only">{t("gov.map_layers")}</legend>
              <div className="seg">
                {LAYERS.map((l) => (
                  <label key={l} className={`whitespace-nowrap ${layer === l ? "on" : ""}`}>
                    <input type="radio" name="layer" value={l} checked={layer === l} onChange={() => setLayer(l)} className="sr-only" />
                    {t(`gov.layer.${l}`)}
                  </label>
                ))}
              </div>
            </fieldset>
          </div>
          {layer === "forecast" && !o.model.operational_forecast && <p className="text-sm text-ink2 mb-2">{t("gov.map_no_forecast")}</p>}
          {geo.data ? <MapView geo={geo.data} districts={list.data} layer={layer} selected={selected} onSelect={setPicked}
                        className="h-[380px] sm:h-[500px] xl:h-[640px]" />
                    : <div className="skeleton h-[380px] sm:h-[500px] xl:h-[640px]" />}
          <div className="pt-4 flex flex-wrap items-center justify-between gap-3">
            <MapLegend layer={layer} />
            <p className="text-xs text-muted">{t("gov.map_hint")}</p>
          </div>
        </section>

        <div className="col-span-2 md:col-span-6 xl:col-span-4 grid md:grid-cols-2 xl:grid-cols-1 gap-4 content-start">
          <Glance r={sel} />
          <Tile title={t("gov.needs_title")} icon={ClipboardList} tour="needs-panel">
            <h3 className="text-sm font-semibold flex items-center gap-1.5 mb-2"><CircleCheck size={15} aria-hidden="true" style={{ color: "var(--status-good)" }} />{t("gov.have_title")}</h3>
            <ul className="space-y-1.5 text-sm text-ink2 mb-4">
              <li>{t("gov.have.wells", { n: o.counts.wells_used })}</li>
              <li>{t("gov.have.forecast", { n: o.counts.with_forecast, from: fmtDate(o.model.backtest_period[0], lang), to: fmtDate(o.model.backtest_period[1], lang) })}</li>
              <li>{t("gov.have.rain", { date: fmtDate(o.freshness.rain_as_of, lang) })}</li>
              <li>{t("gov.have.extraction", { year: o.freshness.extraction_year })}</li>
            </ul>
            <h3 className="text-sm font-semibold flex items-center gap-1.5 mb-2"><CircleDashed size={15} aria-hidden="true" className="text-fall" />{t("gov.need_title")}</h3>
            <ul className="space-y-1.5 text-sm text-ink2">
              <li>{t("gov.need.no_data", { n: noData.length, list: noData.join(", ") })}</li>
              {names("single_well").length > 0 && <li>{t("gov.need.single", { list: names("single_well").join(", ") })}</li>}
              <li>{t("gov.need.fresh", { date: fmtDate(o.freshness.groundwater_last_reading, lang) })}</li>
              {o.freshness.rain_provisional_from && <li>{t("gov.need.provisional", { date: fmtDate(o.freshness.rain_provisional_from, lang) })}</li>}
              {names("reliability_limited").length > 0 && <li>{t("gov.need.limited", { list: names("reliability_limited").join(", ") })}</li>}
              {!o.model.operational_forecast && <li>{t("gov.need.operational")}</li>}
            </ul>
          </Tile>
        </div>

        <section data-tour="priority-table" className="tile p-5 sm:p-6 col-span-full min-w-0 fade-in" aria-labelledby="pt-h">
          <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
            <div>
              <h2 id="pt-h" className="eyebrow flex items-center gap-2"><ListOrdered size={16} aria-hidden="true" className="text-muted" />{t("gov.table_title")}</h2>
              <p className="text-sm text-muted mt-1">{t("gov.table_hint")} · {t("gov.rows", { n: rows.length })}</p>
            </div>
            <div className="flex flex-wrap gap-2 items-center">
              <label className="text-sm text-ink2" htmlFor="action-filter">{t("gov.col.action")}</label>
              <select id="action-filter" value={action} onChange={(e) => setAction(e.target.value)}
                className="rounded-[10px] border border-[var(--border)] bg-surface px-3 min-h-[44px] text-[0.9375rem]">
                <option value="all">{t("gov.filter_all")}</option>
                {["improve_data", "manage_demand", "monitor"].map((a) => <option key={a} value={a}>{t(`gov.action.${a}`)}</option>)}
              </select>
              <a data-tour="download-csv" href="/api/export/priorities.csv" download className="btn text-sm">
                <Download size={16} aria-hidden="true" />{t("gov.download")}</a>
            </div>
          </div>
          <div className="overflow-x-auto -mx-5 sm:-mx-6 px-5 sm:px-6">
            <table className="data-table w-full text-sm tabnum">
              <thead className="text-ink2">
                <tr className="border-b border-[var(--gridline)]">
                  {th("district", t("gov.col.district"))}{th("score", t("gov.col.score"), true)}
                  <th scope="col" className="py-2.5 px-3 text-left font-semibold">{t("gov.col.reasons")}</th>
                  {th("stage_pct", t("gov.col.stage"), true)}
                  <th scope="col" className="py-2.5 px-3 text-left font-semibold"><span className="sr-only">{t("govd.category")}</span></th>
                  {th("depth_m", t("gov.col.depth"), true)}{th("vs_usual_m", t("gov.col.vs_usual"), true)}
                  {th("forecast_change_t30_m", t("gov.col.fc30"), true)}{th("wells", t("gov.col.wells"), true)}
                  <th scope="col" className="py-2.5 px-3 text-left font-semibold">{t("gov.col.rel")}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r: PriorityRow) => (
                  <tr key={r.slug} className={`border-b border-[var(--gridline)] align-top transition-colors hover:bg-[var(--surface-2)]
                    ${r.slug === selected ? "bg-[var(--accent-soft)]" : ""}`}>
                    <td className="py-2.5 px-3 whitespace-nowrap"><Link className="font-semibold hover:underline" to={`/gov/district/${r.slug}`}>{districtName(r.district, lang, false)}</Link></td>
                    <td className="py-2.5 px-3 text-right font-semibold">{r.score}</td>
                    <td className="py-2 px-3 min-w-[260px]">{r.flags.map((f) => <FlagChip key={f.key} f={f} />)}</td>
                    <td className="py-2.5 px-3 text-right">{r.stage_pct !== null ? `${fmtNum(r.stage_pct, 0, lang)}%` : "–"}</td>
                    <td className="py-2 px-3"><CategoryBadge category={r.category} /></td>
                    <td className="py-2.5 px-3 text-right">{fmtNum(r.depth_m, 1, lang)}</td>
                    <td className="py-2.5 px-3 text-right">{signed(r.vs_usual_m, lang)}</td>
                    <td className="py-2.5 px-3 text-right">{signed(r.forecast_change_t30_m, lang)}</td>
                    <td className="py-2.5 px-3 text-right">{r.wells ?? 0}</td>
                    <td className="py-2 px-3"><ReliabilityBadge label={r.reliability_t30} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-muted mt-3">{t("gov.method_note")}</p>
        </section>
      </div>
    </div>
  );
}
