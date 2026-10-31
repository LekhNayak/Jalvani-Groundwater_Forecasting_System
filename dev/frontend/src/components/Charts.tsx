import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { fmtDate, fmtNum } from "../lib/format";
import type { History, Horizon, Lang } from "../lib/types";

const ts = (iso: string) => new Date(`${iso}T00:00:00`).getTime();
const iso = (t: number) => new Date(t).toISOString().slice(0, 10);

function monthTick(t: number, lang: Lang) {
  return new Intl.DateTimeFormat(`${lang}-IN-u-nu-latn`, { month: "short", year: "2-digit" }).format(new Date(t));
}

interface Row { t: number; obs?: number | null; q50?: number | null; band?: [number, number] | null }

function Legend({ items }: { items: { label: string; swatch: "line" | "dash" | "band" }[] }) {
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-ink2 mb-2" aria-hidden="true">
      {items.map((it) => (
        <li key={it.label} className="flex items-center gap-1.5">
          {it.swatch === "band"
            ? <span className="inline-block w-4 h-3 rounded-sm" style={{ background: "var(--band-1)" }} />
            : <span className="inline-block w-5 h-0 border-t-2" style={{ borderColor: it.swatch === "line" ? "var(--text-primary)" : "var(--series-1)",
                borderStyle: it.swatch === "dash" ? "dashed" : "solid" }} />}
          {it.label}
        </li>
      ))}
    </ul>
  );
}

function ChartTooltip({ active, payload, label, lang, names }:
  { active?: boolean; payload?: { dataKey: string; value: number | [number, number] }[]; label?: number; lang: Lang;
    names: Record<string, string> }) {
  const { t } = useTranslation();
  if (!active || !payload?.length || label === undefined) return null;
  return (
    <div className="card px-3 py-2 text-sm shadow">
      <div className="font-semibold mb-1">{fmtDate(iso(label), lang)}</div>
      {payload.filter((p) => p.value !== null && p.value !== undefined).map((p) => (
        <div key={p.dataKey} className="tabnum">
          {names[p.dataKey]}: {Array.isArray(p.value) ? `${fmtNum(p.value[0], 1, lang)} – ${fmtNum(p.value[1], 1, lang)}` : fmtNum(p.value, 1, lang)} {t("common.m")}
        </div>
      ))}
    </div>
  );
}

function DataTable({ rows, lang, names }: { rows: Row[]; lang: Lang; names: Record<string, string> }) {
  const { t } = useTranslation();
  return (
    <div className="max-h-72 overflow-auto">
      <table className="w-full text-sm tabnum">
        <thead><tr className="text-left text-ink2"><th className="py-1 pr-3">{t("common.date")}</th>
          {Object.keys(names).map((k) => <th key={k} className="py-1 pr-3">{names[k]}</th>)}</tr></thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.t} className="border-t border-[var(--gridline)]">
              <td className="py-1 pr-3 whitespace-nowrap">{fmtDate(iso(r.t), lang)}</td>
              {Object.keys(names).map((k) => {
                const v = r[k as keyof Row] as number | [number, number] | null | undefined;
                return <td key={k} className="py-1 pr-3">{Array.isArray(v) ? `${fmtNum(v[0], 1, lang)} – ${fmtNum(v[1], 1, lang)}` : fmtNum(v ?? null, 1, lang)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function DepthPlot({ rows, names, legend, height = 260 }:
  { rows: Row[]; names: Record<string, string>; legend: { label: string; swatch: "line" | "dash" | "band" }[]; height?: number }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const [table, setTable] = useState(false);
  return (
    <div>
      <div className="flex items-center justify-between gap-2">
        <Legend items={legend} />
        <button className="text-sm underline min-h-[40px]" onClick={() => setTable((x) => !x)}>
          {table ? t("common.show_chart") : t("common.show_table")}
        </button>
      </div>
      {table ? <DataTable rows={rows} lang={lang} names={names} /> : (
        <div style={{ height }} role="img" aria-label={legend.map((l) => l.label).join(", ")}>
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 4 }}>
              <CartesianGrid stroke="var(--gridline)" vertical={false} />
              <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={(v) => monthTick(v, lang)}
                stroke="var(--axis)" tick={{ fill: "var(--text-muted)", fontSize: 12 }} minTickGap={28} />
              <YAxis reversed width={44} stroke="var(--axis)" tick={{ fill: "var(--text-muted)", fontSize: 12 }}
                domain={["auto", "auto"]} tickFormatter={(v) => fmtNum(v, 0, lang)}
                label={{ value: t("farmer.chart_axis"), angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 12 }} />
              <Tooltip content={<ChartTooltip lang={lang} names={names} />} cursor={{ stroke: "var(--axis)", strokeWidth: 1 }} />
              {"band" in names && <Area dataKey="band" stroke="none" fill="var(--band-1)" fillOpacity={0.85} isAnimationActive={false} connectNulls={false} />}
              {"q50" in names && <Line dataKey="q50" stroke="var(--series-1)" strokeWidth={2} strokeDasharray="6 4" dot={false} isAnimationActive={false} connectNulls={false} />}
              {"obs" in names && <Line dataKey="obs" stroke="var(--text-primary)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />}
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

/** Farmer view: the last ~18 months of measured depth, plus the latest forecast and its range (if produced). */
export function FarmerDepthChart({ history }: { history: History }) {
  const { t } = useTranslation();
  const rows = useMemo(() => {
    const obs = history.observed_weekly.slice(-78).map(([d, v]) => ({ t: ts(d), obs: -v } as Row));
    const path = history.latest_path?.rows.map(([d, lo, mid, hi]) => ({
      t: ts(d as string), q50: -(mid as number), band: [-(hi as number), -(lo as number)] as [number, number] } as Row)) ?? [];
    return [...obs, ...path].sort((a, b) => a.t - b.t);
  }, [history]);
  const names: Record<string, string> = { obs: t("farmer.chart_observed") };
  const legend: { label: string; swatch: "line" | "dash" | "band" }[] = [{ label: t("farmer.chart_observed"), swatch: "line" }];
  if (history.latest_path) {
    names.q50 = t("farmer.chart_forecast");
    names.band = t("farmer.chart_range");
    legend.push({ label: t("farmer.chart_forecast"), swatch: "dash" }, { label: t("farmer.chart_range"), swatch: "band" });
  }
  return <DepthPlot rows={rows} names={names} legend={legend} />;
}

/** Government view: measured depth vs the past (back-tested) forecasts at one horizon, by target date. */
export function BacktestChart({ history, horizon }: { history: History; horizon: Horizon }) {
  const { t } = useTranslation();
  const rows = useMemo(() => {
    const b = history.backtest[horizon];
    if (!b) return [];
    return b.rows.map(([, target, actual, lo, mid, hi]) => ({
      t: ts(target as string), obs: actual === null ? null : -(actual as number), q50: -(mid as number),
      band: [-(hi as number), -(lo as number)] as [number, number] } as Row));
  }, [history, horizon]);
  const names = { band: t("govd.chart_past_range"), q50: t("govd.chart_past_fc"), obs: t("govd.chart_actual") };
  return <DepthPlot rows={rows} names={names} height={300}
    legend={[{ label: t("govd.chart_actual"), swatch: "line" }, { label: t("govd.chart_past_fc"), swatch: "dash" },
             { label: t("govd.chart_past_range"), swatch: "band" }]} />;
}
