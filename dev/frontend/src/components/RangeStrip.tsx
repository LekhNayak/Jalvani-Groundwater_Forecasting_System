import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { depth, fmtNum } from "../lib/format";
import type { HorizonForecast, Lang } from "../lib/types";

export interface StripRow { key: string; label: string; sub?: string; tag?: ReactNode; f: HorizonForecast }

/** Forecast ranges on one shared depth scale (left = shallower, right = deeper):
 *  light band = "could be" (8 in 10), darker band = "most likely" (about 1 in 2), dark tick = expected level,
 *  dashed line = level at the last reading. The same numbers are always printed as text beside it. */
export default function RangeStrip({ rows, now }: { rows: StripRow[]; now: number }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const nowD = depth(now)!;
  const vals = rows.flatMap((r) => [depth(r.f.q10)!, depth(r.f.q90)!]).concat(nowD);
  const span = Math.max(...vals) - Math.min(...vals);
  const lo = Math.min(...vals) - Math.max(0.15, span * 0.08);
  const hi = Math.max(...vals) + Math.max(0.15, span * 0.08);
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  const w = (a: number, b: number) => `${(Math.abs(b - a) / (hi - lo)) * 100}%`;

  return (
    <div>
      <ul className="space-y-5">
        {rows.map((r) => {
          const mid = depth(r.f.q50)!, near = depth(r.f.q90)!, far = depth(r.f.q10)!;
          const lk = r.f.likely_lo !== undefined && r.f.likely_lo !== null;
          const lNear = lk ? depth(r.f.likely_hi)! : mid, lFar = lk ? depth(r.f.likely_lo)! : mid;
          return (
            <li key={r.key}>
              <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 mb-2">
                <span className="flex flex-wrap items-center gap-x-3 gap-y-1">
                  <span className="font-semibold">{r.label}</span>{r.tag}
                </span>
                {r.sub && <span className="text-sm text-muted tabnum">{r.sub}</span>}
              </div>
              <div className="relative h-7" role="img"
                aria-label={`${r.label}: ${t("farmer.expected", { v: fmtNum(mid, 1, lang) })}; ` +
                  (lk ? `${t("farmer.most_likely", { lo: fmtNum(lNear, 1, lang), hi: fmtNum(lFar, 1, lang) })}; ` : "") +
                  t("farmer.could_be", { lo: fmtNum(near, 1, lang), hi: fmtNum(far, 1, lang) })}>
                <div className="absolute inset-y-[11px] left-0 right-0 rounded-full" style={{ background: "var(--gridline)" }} />
                <div className="absolute inset-y-[7px] rounded-full" style={{ left: x(near), width: w(near, far), background: "var(--band-1)" }} />
                {lk && <div className="absolute inset-y-[4px] rounded-full" style={{ left: x(lNear), width: w(lNear, lFar), background: "var(--band-2)" }} />}
                <div className="absolute inset-y-0 w-[3px] -ml-[1.5px] rounded-full" style={{ left: x(mid), background: "var(--rise)" }} />
                <div className="absolute -inset-y-1 w-0 border-l-2 border-dashed" style={{ left: x(nowD), borderColor: "var(--text-secondary)" }} />
              </div>
              <p className="mt-1.5 text-[0.95rem]">
                <span className="font-semibold tabnum">{t("farmer.expected", { v: fmtNum(mid, 1, lang) })}</span>
                {lk && <span className="text-ink2"> · {t("farmer.most_likely", { lo: fmtNum(lNear, 1, lang), hi: fmtNum(lFar, 1, lang) })}</span>}
                <span className="text-muted text-sm"> · {t("farmer.could_be", { lo: fmtNum(near, 1, lang), hi: fmtNum(far, 1, lang) })}</span>
              </p>
            </li>
          );
        })}
      </ul>
      <div className="mt-3 text-xs text-muted" aria-hidden="true">
        <div className="flex justify-between"><span>← {t("farmer.shallower")}</span><span>{t("farmer.deeper")} →</span></div>
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5">
          <span className="flex items-center gap-1.5"><span className="inline-block w-4 h-2.5 rounded-full" style={{ background: "var(--band-2)" }} />{t("farmer.legend_likely")}</span>
          <span className="flex items-center gap-1.5"><span className="inline-block w-4 h-2.5 rounded-full" style={{ background: "var(--band-1)" }} />{t("farmer.legend_could")}</span>
          <span className="flex items-center gap-1.5"><span className="inline-block w-0.5 h-3" style={{ background: "var(--rise)" }} />{t("farmer.legend_expected")}</span>
          <span className="flex items-center gap-1.5"><span className="inline-block h-3 border-l-2 border-dashed" style={{ borderColor: "var(--text-secondary)" }} />{t("farmer.legend_now", { v: fmtNum(nowD, 1, lang) })}</span>
        </div>
      </div>
    </div>
  );
}

/** Small trend line of recent weekly levels (the numbers are in the tile). Up = closer to the surface. With `fill` it
 *  grows to the height its flex parent gives it. */
export function Sparkline({ points, height = 56, fill = false }: { points: number[]; height?: number; fill?: boolean }) {
  if (points.length < 2) return null;
  const min = Math.min(...points), max = Math.max(...points);
  const pad = (max - min) * 0.1 || 0.1;
  const W = 240, H = 100;
  const xy = points.map((v, i) => [(i / (points.length - 1)) * W, ((max + pad - v) / (max - min + 2 * pad)) * H]);
  const d = xy.map(([a, b], i) => `${i ? "L" : "M"}${a.toFixed(1)},${b.toFixed(1)}`).join(" ");
  const [lx, ly] = xy[xy.length - 1];
  return (
    <div className={`relative w-full ${fill ? "flex-1 min-h-[64px]" : ""}`} style={fill ? undefined : { height }} aria-hidden="true">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="absolute inset-0 w-full h-full">
        <path d={`${d} L${W},${H} L0,${H} Z`} fill="var(--accent-soft)" />
        <path d={d} fill="none" stroke="var(--series-1)" strokeWidth={2} vectorEffect="non-scaling-stroke" />
      </svg>
      {/* end dot drawn in HTML so it stays round when the svg stretches */}
      <span className="absolute w-2 h-2 -ml-1 -mt-1 rounded-full" style={{ left: `${(lx / W) * 100}%`, top: `${(ly / H) * 100}%`, background: "var(--series-1)" }} />
    </div>
  );
}
