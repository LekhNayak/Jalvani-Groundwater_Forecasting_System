import { lazy, Suspense, useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Card, DirMark, ErrorBox, Loading, ReliabilityBadge } from "../components/Bits";
import DistrictPicker from "../components/DistrictPicker";
import ListenButton from "../components/ListenButton";
import { useApp } from "../context";
import { districtName } from "../i18n";
import { api, useApi } from "../lib/api";
import { changeDir, depth, fmtDate, fmtNum } from "../lib/format";
import { HORIZONS, type District, type Lang } from "../lib/types";

const FarmerDepthChart = lazy(() => import("../components/Charts").then((m) => ({ default: m.FarmerDepthChart })));

const DAYS: Record<string, number> = { "t+7": 7, "t+14": 14, "t+30": 30 };
const ADVICE_ICON: Record<string, string> = { warn: "⚠️", good: "💧", info: "ℹ️" };

function WaterNow({ d, lang }: { d: District; lang: Lang }) {
  const { t } = useTranslation();
  const g = d.groundwater!;
  const d30 = changeDir(g.change_30d_m), d1y = changeDir(g.change_1y_m, 0.2);
  const lines = [
    t("common.m_below_ground", { v: fmtNum(g.depth_m, 1, lang) }),
    d30 && t(`farmer.trend30.${d30}`, { v: fmtNum(Math.abs(g.change_30d_m ?? 0), 1, lang) }),
    d1y && t(`farmer.trend1y.${d1y}`, { v: fmtNum(Math.abs(g.change_1y_m ?? 0), 1, lang) }),
    g.vs_usual && t(`farmer.usual.${g.vs_usual}`),
  ].filter(Boolean) as string[];
  return (
    <Card title={t("farmer.water_now_title")} tour="card-water-now" actions={<ListenButton text={lines.join(". ")} />}>
      <p className="text-4xl font-semibold tabnum">{fmtNum(g.depth_m, 1, lang)} <span className="text-xl font-normal">{t("common.m")}</span></p>
      <p className="text-ink2 mb-3">{t("farmer.latest_reading", { date: fmtDate(g.last_reading, lang) })}</p>
      <ul className="space-y-1">
        {d30 && <li><DirMark dir={d30} />{lines[1]}</li>}
        {d1y && <li><DirMark dir={d1y} />{t(`farmer.trend1y.${d1y}`, { v: fmtNum(Math.abs(g.change_1y_m ?? 0), 1, lang) })}</li>}
      </ul>
      {g.vs_usual && (
        <p className={`mt-3 inline-block rounded-lg px-3 py-1.5 font-semibold ${g.vs_usual === "deeper" ? "bg-[var(--status-serious)]/25" : "bg-plane"}`}>
          <DirMark dir={g.vs_usual === "deeper" ? "fall" : g.vs_usual === "shallower" ? "rise" : "steady"} />{t(`farmer.usual.${g.vs_usual}`)}
        </p>
      )}
      {g.usual_years.length > 0 && <p className="text-xs text-muted mt-2">{t("farmer.usual_note", { years: g.usual_years.join(", ") })}</p>}
    </Card>
  );
}

function ForecastCard({ d, lang }: { d: District; lang: Lang }) {
  const { t } = useTranslation();
  const rel = d.reliability?.["t+30"]?.label;
  if (!d.forecast) {
    const r = d.forecast_unavailable;
    return (
      <Card title={t("farmer.forecast_title")} tour="card-forecast">
        <p className="font-semibold">{t("farmer.forecast_unavailable")}</p>
        {r && <p className="text-ink2 mt-1">{t(`farmer.fc_reason.${r.code}`, { share: r.share_pct })}</p>}
        {rel && <div className="mt-3" data-tour="reliability-badge"><span className="text-sm text-ink2 mr-2">{t("farmer.reliability_title")}:</span>
          <ReliabilityBadge label={rel} /><p className="text-sm text-ink2 mt-1">{t(`farmer.rel_explain.${rel}`)}</p></div>}
      </Card>
    );
  }
  const f = d.forecast;
  const spoken = HORIZONS.filter((h) => f.horizons[h]).map((h) =>
    `${t("farmer.forecast_days", { n: DAYS[h] })}: ${t("farmer.expected", { v: fmtNum(depth(f.horizons[h]!.q50), 1, lang) })}, ` +
    t(`farmer.fc_dir.${d.direction[h] ?? "uncertain"}`));
  return (
    <Card title={t("farmer.forecast_title")} tour="card-forecast" actions={<ListenButton text={spoken.join(". ")} />}>
      <p className="text-sm text-ink2 mb-2">{t("farmer.forecast_from", { date: fmtDate(f.issue_date, lang) })}</p>
      <ul className="divide-y divide-[var(--gridline)]">
        {HORIZONS.map((h) => {
          const x = f.horizons[h];
          if (!x) return null;
          const dir = d.direction[h] ?? "uncertain";
          const lo = depth(x.q90), hi = depth(x.q10);
          return (
            <li key={h} className="py-2">
              <div className="flex justify-between gap-2"><span className="font-semibold">{t("farmer.forecast_days", { n: DAYS[h] })}</span>
                <span className="text-sm text-muted">{fmtDate(x.target_date, lang)}</span></div>
              <div><DirMark dir={dir === "falling" ? "fall" : dir === "rising" ? "rise" : "steady"} />
                {t("farmer.expected", { v: fmtNum(depth(x.q50), 1, lang) })} · {t(`farmer.fc_dir.${dir}`)}</div>
              <div className="text-sm text-ink2">{t("farmer.range", { lo: fmtNum(lo, 1, lang), hi: fmtNum(hi, 1, lang) })}</div>
            </li>
          );
        })}
      </ul>
      {rel && <div className="mt-3" data-tour="reliability-badge"><span className="text-sm text-ink2 mr-2">{t("farmer.reliability_title")}:</span>
        <ReliabilityBadge label={rel} /><p className="text-sm text-ink2 mt-1">{t(`farmer.rel_explain.${rel}`)}</p></div>}
    </Card>
  );
}

function RainSoil({ d, lang }: { d: District; lang: Lang }) {
  const { t } = useTranslation();
  const r = d.rain, s = d.soil;
  if (!r) return null;
  const season = r.season;
  const text = [
    t("farmer.rain_season", { date: fmtDate(season.from, lang), mm: fmtNum(season.mm, 0, lang) }),
    season.pct_of_usual !== null ? t("farmer.rain_vs_usual", { pct: fmtNum(season.pct_of_usual, 0, lang) }) : "",
    s?.sm_l1.vs_usual ? t(`farmer.soil_label.${s.sm_l1.vs_usual}`) : "",
  ].filter(Boolean);
  return (
    <Card title={t("farmer.rain_title")} tour="card-rain-soil" actions={<ListenButton text={text.join(". ")} />}>
      <p className="text-lg">{text[0]}</p>
      {season.pct_of_usual !== null && (
        <div className="mt-2">
          <div className="flex justify-between text-sm"><span>{text[1]}</span>
            {season.provisional_days > 0 && <span className="text-muted">{t("common.days_provisional", { n: season.provisional_days })}</span>}</div>
          <div className="h-3 rounded-full bg-plane border border-[var(--border)] mt-1 overflow-hidden" aria-hidden="true">
            <div className="h-full" style={{ width: `${Math.min(100, (season.pct_of_usual / 150) * 100)}%`, background: "var(--series-1)" }} />
          </div>
          <p className="text-xs text-muted mt-1">{t("farmer.rain_usual_note", { years: season.usual_years.join(", ") })}</p>
        </div>
      )}
      <p className="text-sm text-ink2 mt-2">
        {t("farmer.rain_30d", { mm: fmtNum(r.last_30d.mm, 0, lang), pct: fmtNum(r.last_30d.pct_of_usual, 0, lang) })}
        {r.last_30d.provisional_days > 0 && <span className="text-muted"> · {t("common.provisional")}</span>}
      </p>
      {s && s.sm_l1.vs_usual && (
        <p className="mt-3"><span className="font-semibold">{t(`farmer.soil_label.${s.sm_l1.vs_usual}`)}</span>
          <span className="block text-sm text-ink2">{t("farmer.soil_value", { v: fmtNum(s.sm_l1.pct, 0, lang), u: fmtNum(s.sm_l1.usual_pct, 0, lang) })} · {t("common.as_of", { date: fmtDate(s.as_of, lang) })}</span></p>
      )}
    </Card>
  );
}

function DistrictView({ slug }: { slug: string }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const nav = useNavigate();
  const { setVisible, saveDistrict } = useApp();
  const ds = useApi(() => api.district(slug), slug);
  const hist = useApi(() => api.history(slug), `h-${slug}`);
  const [details, setDetails] = useState(false);

  useEffect(() => { saveDistrict(slug); }, [slug, saveDistrict]);
  useEffect(() => {
    const d = ds.data;
    if (!d) return;
    setVisible({ page: "farmer", district: d.district, status: d.status, groundwater: d.groundwater, forecast: d.forecast,
                 forecast_unavailable: d.forecast_unavailable, reliability_t30: d.reliability?.["t+30"]?.label,
                 rain_season: d.rain?.season, soil: d.soil?.sm_l1, advice: d.advice.map((a) => a.key) });
  }, [ds.data, setVisible]);

  if (ds.loading) return <Loading />;
  if (ds.error || !ds.data) return <ErrorBox onRetry={() => nav(0)} />;
  const d = ds.data;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">{districtName(d.district, lang)}</h1>
          <p className="text-sm text-muted">{t("common.district_level_only")}</p>
        </div>
        <button data-tour="district-picker" onClick={() => nav("/farmer?pick=1")}
          className="px-4 py-2 min-h-[44px] rounded-lg border border-[var(--border)] hover:bg-plane">{t("farmer.change_district")}</button>
      </div>

      {d.status !== "forecast" && (
        <div className="card p-4" role="note">
          <p className="font-semibold">{t("farmer.no_groundwater_title")}</p>
          <p className="text-ink2">{d.status === "no_data" ? t("farmer.no_groundwater_text", { district: districtName(d.district, lang, false) })
                                                          : t("farmer.insufficient_text", { district: districtName(d.district, lang, false) })}</p>
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-3">
        {d.groundwater && <WaterNow d={d} lang={lang} />}
        {d.status === "forecast" && <ForecastCard d={d} lang={lang} />}
        <RainSoil d={d} lang={lang} />
      </div>

      {d.advice.length > 0 && (
        <section data-tour="advice-list" className="card p-4 sm:p-5" aria-labelledby="advice-h">
          <h2 id="advice-h" className="text-lg font-semibold mb-2">{t("farmer.advice_title")}</h2>
          <ul className="space-y-2">
            {d.advice.map((a) => (
              <li key={a.key} className="flex gap-2"><span aria-hidden="true">{ADVICE_ICON[a.level]}</span>
                <span>{t(a.key, Object.fromEntries(Object.entries(a.params).map(([k, v]) => [k, typeof v === "number" ? fmtNum(v, k.endsWith("_m") ? 1 : 0, lang) : v])))}</span></li>
            ))}
          </ul>
        </section>
      )}

      {d.groundwater && (
        <section className="card p-4 sm:p-5" data-tour="details-chart">
          <button className="w-full flex justify-between items-center min-h-[44px] font-semibold" aria-expanded={details} onClick={() => setDetails((x) => !x)}>
            <span>{t("farmer.details_title")}</span><span className="text-sm underline">{details ? t("common.details_hide") : t("common.details_show")}</span>
          </button>
          {details && hist.data && (<><p className="text-sm text-muted mb-2">{t("farmer.chart_hint")}</p><Suspense fallback={<Loading />}><FarmerDepthChart history={hist.data} /></Suspense></>)}
        </section>
      )}
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
  if (list.loading) return <Loading />;
  if (list.error || !list.data) return <ErrorBox onRetry={() => nav(0)} />;
  return <DistrictPicker districts={list.data} onSelect={(s) => nav(`/farmer/${s}`)} />;
}
