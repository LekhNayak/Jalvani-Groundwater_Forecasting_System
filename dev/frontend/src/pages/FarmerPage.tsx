import { lazy, Suspense, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { AlertTriangle, CalendarClock, CloudRain, Droplet, Droplets, Factory, Info, Lightbulb, LineChart, MapPin,
         ChevronDown, type LucideIcon } from "lucide-react";
import { CategoryBadge, DirMark, ErrorBox, Loading, ReliabilityBadge, Tile } from "../components/Bits";
import DistrictPicker from "../components/DistrictPicker";
import ListenButton from "../components/ListenButton";
import RangeStrip, { Sparkline, type StripRow } from "../components/RangeStrip";
import { useApp } from "../context";
import { districtName } from "../i18n";
import { api, useApi } from "../lib/api";
import { changeDir, depth, fmtDate, fmtNum } from "../lib/format";
import { HORIZONS, type District, type History, type Lang } from "../lib/types";

const FarmerDepthChart = lazy(() => import("../components/Charts").then((m) => ({ default: m.FarmerDepthChart })));

const DAYS: Record<string, number> = { "t+7": 7, "t+14": 14, "t+30": 30 };
const ADVICE_ICON: Record<string, LucideIcon> = { warn: AlertTriangle, good: Droplet, info: Info };
/** whole days from an ISO date to today (negative = in the future) */
const daysSince = (iso: string) => Math.floor((Date.now() - new Date(`${iso}T00:00:00`).getTime()) / 86_400_000);
const dirOf = (w: string | null | undefined) => (w === "falling" ? "fall" : w === "rising" ? "rise" : "steady");
const wide = () => typeof window !== "undefined" && window.matchMedia?.("(min-width: 1280px)").matches;

/** skeleton in the shape of the farmer grid */
const SKELETON = ["md:col-span-6 xl:col-span-4 h-[22rem]", "md:col-span-6 xl:col-span-8 h-[22rem]", "md:col-span-3 xl:col-span-5 h-60",
                  "md:col-span-3 xl:col-span-3 h-60", "md:col-span-6 xl:col-span-4 h-60"];

function Note({ icon: Icon, children, tone = "info" }: { icon: LucideIcon; children: React.ReactNode; tone?: "info" | "warn" }) {
  return (
    <div role="note" className="col-span-full tile px-5 py-3.5 flex items-start gap-3 fade-in"
      style={tone === "warn" ? { borderLeft: "4px solid var(--status-warning)" } : undefined}>
      <Icon size={18} aria-hidden="true" className="mt-0.5 shrink-0 text-ink2" />
      <div className="text-[0.95rem]">{children}</div>
    </div>
  );
}

function WaterNow({ d, hist, lang }: { d: District; hist: History | null; lang: Lang }) {
  const { t } = useTranslation();
  const g = d.groundwater!;
  const d30 = changeDir(g.change_30d_m), d1y = changeDir(g.change_1y_m, 0.2);
  const t30 = d30 && t(`farmer.trend30.${d30}`, { v: fmtNum(Math.abs(g.change_30d_m ?? 0), 1, lang) });
  const t1y = d1y && t(`farmer.trend1y.${d1y}`, { v: fmtNum(Math.abs(g.change_1y_m ?? 0), 1, lang) });
  const spoken = [t("common.m_below_ground", { v: fmtNum(g.depth_m, 1, lang) }), t30, t1y, g.vs_usual && t(`farmer.usual.${g.vs_usual}`)]
    .filter(Boolean).join(". ");
  const weeks = (hist?.observed_weekly ?? []).slice(-26);
  const spark = weeks.map(([, v]) => v);
  const sparkFrom = weeks[0]?.[0], sparkTo = weeks[weeks.length - 1]?.[0];
  return (
    <Tile title={t("farmer.water_now_title")} icon={Droplets} tour="card-water-now" className="md:col-span-6 xl:col-span-4"
      actions={<ListenButton text={spoken} />} bodyClass="flex flex-col">
      <div className="flex items-baseline gap-2">
        <span className="text-5xl font-semibold tracking-tight tabnum">{fmtNum(g.depth_m, 1, lang)}</span>
        <span className="text-ink2">{t("farmer.chart_axis")}</span>
      </div>
      <p className="text-sm text-muted mt-1">{t("farmer.week_avg", { date: fmtDate(g.last_reading, lang) })}</p>
      {d.forecast && Math.abs(d.forecast.level_now_m - g.level_m) >= 0.2 && (
        <p className="text-sm text-ink2 mt-1">{t("farmer.last_day", { date: fmtDate(d.forecast.issue_date, lang), v: fmtNum(-d.forecast.level_now_m, 1, lang) })}</p>
      )}

      {spark.length > 1 && (
        <figure className="mt-4 flex-1 flex flex-col min-h-[96px] max-h-[220px]">
          <Sparkline points={spark} fill />
          <div className="flex justify-between text-xs text-muted mt-1 tabnum" aria-hidden="true">
            <span>{fmtDate(sparkFrom, lang)}</span><span>{fmtDate(sparkTo, lang)}</span>
          </div>
          <figcaption className="text-xs text-muted mt-0.5">{t("farmer.spark_caption")} · {t("farmer.chart_hint")}</figcaption>
        </figure>
      )}

      <ul className="mt-4 space-y-2 text-[0.95rem]">
        {t30 && <li className="flex items-center gap-2"><DirMark dir={d30} />{t30}</li>}
        {t1y && <li className="flex items-center gap-2"><DirMark dir={d1y} />{t1y}</li>}
      </ul>

      {g.vs_usual && (
        <div className="pt-4">
          <p className="inline-flex items-center gap-2 rounded-xl px-3 py-2 font-semibold text-[0.95rem]"
            style={{ background: g.vs_usual === "deeper" ? "color-mix(in srgb, var(--status-serious) 22%, transparent)" : "var(--surface-2)" }}>
            <DirMark dir={g.vs_usual === "deeper" ? "fall" : g.vs_usual === "shallower" ? "rise" : "steady"} />{t(`farmer.usual.${g.vs_usual}`)}
          </p>
          {g.usual_years.length > 0 && <p className="text-xs text-muted mt-1.5">{t("farmer.usual_note", { years: g.usual_years.join(", ") })}</p>}
        </div>
      )}
    </Tile>
  );
}

function Reliability({ d }: { d: District }) {
  const { t } = useTranslation();
  const rel = d.reliability?.["t+30"]?.label;
  if (!rel) return null;
  return (
    <div data-tour="reliability-badge" className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
      <span className="text-sm text-ink2">{t("farmer.reliability_title")}</span>
      <ReliabilityBadge label={rel} />
      <span className="text-sm text-muted">{t(`farmer.rel_explain.${rel}`)}</span>
    </div>
  );
}

function ForecastTile({ d, lang }: { d: District; lang: Lang }) {
  const { t } = useTranslation();
  const span = "md:col-span-6 xl:col-span-8";
  if (!d.forecast) {
    const r = d.forecast_unavailable;
    return (
      <Tile title={t("farmer.forecast_title")} icon={LineChart} tour="card-forecast" className={span} bodyClass="flex flex-col">
        <p className="text-lg font-semibold">{t("farmer.forecast_unavailable")}</p>
        {r && <p className="text-ink2 mt-1 max-w-2xl">{t(`farmer.fc_reason.${r.code}`, { share: r.share_pct })}</p>}
        <div className="mt-auto pt-6"><Reliability d={d} /></div>
      </Tile>
    );
  }
  const f = d.forecast;
  const rows: StripRow[] = HORIZONS.filter((h) => f.horizons[h]).map((h) => {
    const x = f.horizons[h]!;
    const dir = d.direction[h] ?? "uncertain";
    return { key: h, label: t("farmer.forecast_days", { n: DAYS[h] }), sub: fmtDate(x.target_date, lang), f: x,
             tag: <span className="inline-flex items-center gap-1.5 text-sm text-ink2"><DirMark dir={dirOf(dir)} size={15} />{t(`farmer.fc_dir.${dir}`)}</span> };
  });
  const spoken = HORIZONS.filter((h) => f.horizons[h]).map((h) => {
    const x = f.horizons[h]!;
    const parts = [`${t("farmer.forecast_days", { n: DAYS[h] })}: ${t("farmer.expected", { v: fmtNum(depth(x.q50), 1, lang) })}`];
    if (x.likely_lo !== null && x.likely_lo !== undefined)
      parts.push(t("farmer.most_likely", { lo: fmtNum(depth(x.likely_hi), 1, lang), hi: fmtNum(depth(x.likely_lo), 1, lang) }));
    parts.push(t("farmer.could_be", { lo: fmtNum(depth(x.q90), 1, lang), hi: fmtNum(depth(x.q10), 1, lang) }));
    parts.push(t(`farmer.fc_dir.${d.direction[h] ?? "uncertain"}`));
    return parts.join(", ");
  });
  const past = f.horizons["t+30"] && daysSince(f.horizons["t+30"].target_date) > 0;
  return (
    <Tile title={t("farmer.forecast_title")} icon={LineChart} tour="card-forecast" className={span}
      actions={<ListenButton text={spoken.join(". ")} />} bodyClass="flex flex-col">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-5">
        <p className="text-sm text-ink2">{t("farmer.forecast_start", { date: fmtDate(f.issue_date, lang), v: fmtNum(-f.level_now_m, 1, lang) })}</p>
        {past && (
          <p role="note" className="inline-flex items-center gap-1.5 text-sm rounded-lg px-2.5 py-1 bg-[var(--surface-2)] text-ink2">
            <CalendarClock size={14} aria-hidden="true" />{t("farmer.forecast_past")}
          </p>
        )}
      </div>
      <RangeStrip rows={rows} now={f.level_now_m} />
      <div className="mt-auto pt-5 border-t border-[var(--gridline)] mt-5 space-y-3">
        <p className="text-sm text-muted">{t("farmer.range_note")}</p>
        <Reliability d={d} />
      </div>
    </Tile>
  );
}

function RainSoil({ d, lang, span }: { d: District; lang: Lang; span: string }) {
  const { t } = useTranslation();
  const r = d.rain, s = d.soil;
  if (!r) return null;
  const season = r.season;
  const pct = season.pct_of_usual;
  const soilLabel = s?.sm_l1.vs_usual ? t(`farmer.soil_label.${s.sm_l1.vs_usual}`) : "";
  const spoken = [t("farmer.rain_season", { date: fmtDate(season.from, lang), mm: fmtNum(season.mm, 0, lang) }),
    pct !== null ? t("farmer.rain_vs_usual", { pct: fmtNum(pct, 0, lang) }) : "", soilLabel].filter(Boolean).join(". ");
  return (
    <Tile title={t("farmer.rain_title")} icon={CloudRain} tour="card-rain-soil" className={span}
      actions={<ListenButton text={spoken} />}>
      {pct !== null ? (
        <>
          <div className="flex items-baseline gap-2">
            <span className="text-4xl font-semibold tracking-tight tabnum">{fmtNum(pct, 0, lang)}%</span>
            <span className="text-ink2 text-[0.95rem]">{t("farmer.of_usual_rain", { date: fmtDate(season.from, lang) })}</span>
          </div>
          <div className="relative mt-3 h-2.5 rounded-full bg-[var(--surface-2)] border border-[var(--border)]" aria-hidden="true">
            <div className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.min(100, (pct / 150) * 100)}%`, background: "var(--series-1)" }} />
            <div className="absolute -inset-y-1 w-0.5 rounded" style={{ left: `${(100 / 150) * 100}%`, background: "var(--text-secondary)" }} />
          </div>
          <div className="relative h-4 text-xs text-muted mt-1" aria-hidden="true">
            <span className="absolute left-0">0%</span>
            <span className="absolute -translate-x-1/2" style={{ left: `${(100 / 150) * 100}%` }}>100%</span>
            <span className="absolute right-0">150%</span>
          </div>
        </>
      ) : <p className="text-lg">{t("farmer.rain_season", { date: fmtDate(season.from, lang), mm: fmtNum(season.mm, 0, lang) })}</p>}

      <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 mt-4 text-[0.95rem]">
        <div>
          <dt className="text-sm text-muted">{t("farmer.rain_since", { date: fmtDate(season.from, lang) })}</dt>
          <dd className="font-semibold tabnum">{fmtNum(season.mm, 0, lang)} mm
            {season.provisional_days > 0 && <span className="block font-normal text-xs text-muted">{t("common.days_provisional", { n: season.provisional_days })}</span>}</dd>
        </div>
        <div>
          <dt className="text-sm text-muted">{t("farmer.last_30d")}</dt>
          <dd className="font-semibold tabnum">{fmtNum(r.last_30d.mm, 0, lang)} mm
            <span className="font-normal text-ink2"> · {t("farmer.pct_of_usual", { pct: fmtNum(r.last_30d.pct_of_usual, 0, lang) })}</span>
            {r.last_30d.provisional_days > 0 && <span className="block font-normal text-xs text-muted">{t("common.provisional")}</span>}</dd>
        </div>
        {s && s.sm_l1.vs_usual && (
          <div className="sm:col-span-2">
            <dt className="text-sm text-muted">{t("farmer.soil_title", { date: fmtDate(s.as_of, lang) })}</dt>
            <dd><span className="font-semibold">{soilLabel}</span>
              <span className="text-ink2"> · {t("farmer.soil_value", { v: fmtNum(s.sm_l1.pct, 0, lang), u: fmtNum(s.sm_l1.usual_pct, 0, lang) })}</span></dd>
          </div>
        )}
      </dl>
      {pct !== null && <p className="text-xs text-muted mt-3">{t("farmer.rain_usual_note", { years: season.usual_years.join(", ") })}</p>}
    </Tile>
  );
}

function GroundwaterUse({ d, lang, span }: { d: District; lang: Lang; span: string }) {
  const { t } = useTranslation();
  const x = d.extraction;
  if (!x) return null;
  const split = x.use_split_pct;
  const parts = split ? (["irrigation", "domestic", "industrial"] as const).filter((k) => split[k] > 0) : [];
  const shade: Record<string, string> = { irrigation: "var(--series-1)", domestic: "var(--band-2)", industrial: "var(--axis)" };
  return (
    <Tile title={t("farmer.extraction_title", { year: x.latest_year })} icon={Factory} className={span}>
      <div className="flex items-baseline gap-2 flex-wrap">
        <span className="text-4xl font-semibold tracking-tight tabnum">{fmtNum(x.stage_pct, 0, lang)}%</span>
        <CategoryBadge category={x.category} />
      </div>
      <p className="text-ink2 text-[0.95rem] mt-1">{t("farmer.use_of")}</p>
      {x.history.length > 1 && (
        <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-sm text-ink2 tabnum" aria-label={t("govd.stage")}>
          {x.history.map((hy) => <li key={hy.year}><span className="text-muted">{hy.year}</span> {fmtNum(hy.stage_pct, 0, lang)}%</li>)}
        </ul>
      )}
      {split && parts.length > 0 && (
        <div className="mt-4">
          <div className="flex h-2.5 rounded-full overflow-hidden gap-[2px]" aria-hidden="true">
            {parts.map((k) => <div key={k} style={{ width: `${split[k]}%`, background: shade[k] }} />)}
          </div>
          <ul className="mt-2 space-y-1 text-sm">
            {parts.map((k) => (
              <li key={k} className="flex items-center gap-2">
                <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: shade[k] }} aria-hidden="true" />
                <span className="flex-1">{t(`govd.use.${k}`)}</span><span className="tabnum text-ink2">{fmtNum(split[k], 0, lang)}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </Tile>
  );
}

function Advice({ d, lang, span }: { d: District; lang: Lang; span: string }) {
  const { t } = useTranslation();
  if (!d.advice.length) return null;
  const fmt = (params: Record<string, number | string>) => Object.fromEntries(Object.entries(params).map(([k, v]) => [k,
    typeof v === "number" ? fmtNum(v, k.endsWith("_m") ? 1 : 0, lang) : /^\d{4}-\d{2}-\d{2}$/.test(String(v)) ? fmtDate(String(v), lang) : v]));
  return (
    <Tile title={t("farmer.advice_title")} icon={Lightbulb} tour="advice-list" className={span}>
      <ul className="space-y-3">
        {d.advice.map((a) => {
          const Icon = ADVICE_ICON[a.level] ?? Info;
          return (
            <li key={a.key} className="flex gap-3 text-[0.95rem]">
              <span aria-hidden="true" className="mt-0.5 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-[var(--surface-2)] text-ink2">
                <Icon size={15} /></span>
              <span>{t(a.key, fmt(a.params))}</span>
            </li>
          );
        })}
      </ul>
    </Tile>
  );
}

function DistrictView({ slug }: { slug: string }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const nav = useNavigate();
  const { setVisible, saveDistrict } = useApp();
  const ds = useApi(() => api.district(slug), slug);
  const hist = useApi(() => api.history(slug), `h-${slug}`);
  const [details, setDetails] = useState(wide);

  useEffect(() => { saveDistrict(slug); }, [slug, saveDistrict]);
  useEffect(() => {
    const d = ds.data;
    if (!d) return;
    setVisible({ page: "farmer", district: d.district, status: d.status, groundwater: d.groundwater, forecast: d.forecast,
                 forecast_unavailable: d.forecast_unavailable, reliability_t30: d.reliability?.["t+30"]?.label,
                 rain_season: d.rain?.season, soil: d.soil?.sm_l1, extraction_stage_pct: d.extraction?.stage_pct,
                 advice: d.advice.map((a) => a.key) });
  }, [ds.data, setVisible]);

  if (ds.loading) return <Loading tiles={SKELETON} />;
  if (ds.error || !ds.data) return <ErrorBox onRetry={() => nav(0)} />;
  const d = ds.data;
  const g = d.groundwater;
  const stale = g ? daysSince(g.last_reading) : 0;
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-[1.75rem] leading-tight font-semibold tracking-tight">{districtName(d.district, lang)}</h1>
          <p className="text-sm text-muted mt-0.5">{t("common.district_level_only")}</p>
        </div>
        <button data-tour="district-picker" onClick={() => nav("/farmer?pick=1")} className="btn">
          <MapPin size={16} aria-hidden="true" />{t("farmer.change_district")}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-6 xl:grid-cols-12 gap-4">
        {g && stale > 30 && (
          <Note icon={CalendarClock} tone="warn">{t("farmer.stale_note", { date: fmtDate(g.last_reading, lang), days: stale })}</Note>
        )}
        {d.status !== "forecast" && !g && (
          <Note icon={Info}>
            <p className="font-semibold">{t("farmer.no_groundwater_title")}</p>
            <p className="text-ink2">{d.status === "no_data" ? t("farmer.no_groundwater_text", { district: districtName(d.district, lang, false) })
                                                            : t("farmer.insufficient_text", { district: districtName(d.district, lang, false) })}</p>
          </Note>
        )}

        {g && <WaterNow d={d} hist={hist.data} lang={lang} />}
        {g && <ForecastTile d={d} lang={lang} />}
        <RainSoil d={d} lang={lang} span="md:col-span-3 xl:col-span-5" />
        <GroundwaterUse d={d} lang={lang} span="md:col-span-3 xl:col-span-3" />
        <Advice d={d} lang={lang} span="md:col-span-6 xl:col-span-4" />

        {g && (
          <section data-tour="details-chart" className="tile p-5 sm:p-6 col-span-full fade-in" aria-labelledby="chart-h">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 id="chart-h" className="eyebrow flex items-center gap-2"><LineChart size={16} aria-hidden="true" className="text-muted" />{t("farmer.details_title")}</h2>
                <p className="text-sm text-muted mt-1">{t("farmer.chart_hint")} {d.forecast && t("farmer.chart_note")}</p>
              </div>
              <button className="btn text-sm" aria-expanded={details} aria-controls="chart-body" onClick={() => setDetails((x) => !x)}>
                {details ? t("common.details_hide") : t("common.details_show")}
                <ChevronDown size={16} aria-hidden="true" className={`transition-transform ${details ? "rotate-180" : ""}`} />
              </button>
            </div>
            {details && (
              <div id="chart-body" className="mt-4">
                {hist.data ? <Suspense fallback={<div className="skeleton h-[280px]" />}><FarmerDepthChart history={hist.data} forecast={d.forecast} /></Suspense>
                           : <div className="skeleton h-[280px]" />}
              </div>
            )}
          </section>
        )}
      </div>
    </div>
  );
}

export default function FarmerPage() {
  const { slug } = useParams();
  const nav = useNavigate();
  const { savedDistrict, setRole, role } = useApp();
  const list = useApi(api.districts, "districts");
  const picking = new URLSearchParams(window.location.search).has("pick");

  useEffect(() => { if (role !== "farmer") setRole("farmer"); }, [role, setRole]);
  useEffect(() => {
    if (!slug && savedDistrict && !picking) nav(`/farmer/${savedDistrict}`, { replace: true });
  }, [slug, savedDistrict, picking, nav]);

  if (slug) return <DistrictView slug={slug} />;
  if (list.loading) return <Loading tiles={["col-span-full h-[28rem]"]} />;
  if (list.error || !list.data) return <ErrorBox onRetry={() => nav(0)} />;
  return <DistrictPicker districts={list.data} onSelect={(s) => nav(`/farmer/${s}`)} />;
}
