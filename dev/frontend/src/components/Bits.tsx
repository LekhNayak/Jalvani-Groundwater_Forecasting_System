import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { AlertTriangle, Check, CircleAlert, Minus, TrendingDown, TrendingUp, X, type LucideIcon } from "lucide-react";
import type { Reliability } from "../lib/types";

/** Status colours always come with an icon and a word (never colour alone). */
const STATUS_STYLE: Record<string, { bg: string; Icon: LucideIcon }> = {
  good: { bg: "var(--status-good)", Icon: Check },
  fair: { bg: "var(--status-warning)", Icon: Minus },
  limited: { bg: "var(--status-serious)", Icon: CircleAlert },
  Safe: { bg: "var(--status-good)", Icon: Check },
  "Semi-Critical": { bg: "var(--status-warning)", Icon: Minus },
  Critical: { bg: "var(--status-serious)", Icon: AlertTriangle },
  "Over-Exploited": { bg: "var(--status-critical)", Icon: X },
};

export function StatusPill({ kind, label }: { kind: string; label: string }) {
  const s = STATUS_STYLE[kind];
  const Icon = s?.Icon ?? Minus;
  return (
    <span className="chip font-semibold whitespace-nowrap">
      <span aria-hidden="true" className="inline-flex h-4 w-4 items-center justify-center rounded-full text-[#111110]"
        style={{ background: s?.bg ?? "var(--nodata)" }}><Icon size={11} strokeWidth={3} /></span>
      {label}
    </span>
  );
}

export function ReliabilityBadge({ label }: { label: Reliability | null | undefined }) {
  const { t } = useTranslation();
  if (!label) return null;
  return <StatusPill kind={label} label={t(`farmer.rel.${label}`)} />;
}

export function CategoryBadge({ category }: { category: string | null | undefined }) {
  const { t } = useTranslation();
  if (!category) return null;
  return <StatusPill kind={category} label={t(`cat.${category}`)} />;
}

/** A bento tile: an icon + small title (eyebrow), optional actions, content. Grid spans come from className. */
export function Tile({ title, icon: Icon, children, tour, actions, className = "", bodyClass = "" }:
  { title?: ReactNode; icon?: LucideIcon; children: ReactNode; tour?: string; actions?: ReactNode; className?: string;
    bodyClass?: string }) {
  return (
    <section data-tour={tour} className={`tile p-5 sm:p-6 flex flex-col min-w-0 fade-in ${className}`}
      aria-label={typeof title === "string" ? title : undefined}>
      {(title || actions) && (
        <header className="flex flex-wrap items-center justify-between gap-x-3 gap-y-2 mb-4">
          <h2 className="eyebrow flex items-center gap-2">
            {Icon && <Icon size={16} strokeWidth={2} aria-hidden="true" className="text-muted" />}{title}
          </h2>
          {actions}
        </header>
      )}
      <div className={`flex-1 min-w-0 ${bodyClass}`}>{children}</div>
    </section>
  );
}

/** Loading placeholder in the shape of the page's bento grid (no layout jump when data arrives). */
export function Loading({ tiles = ["md:col-span-6 xl:col-span-4 h-56", "md:col-span-6 xl:col-span-8 h-56", "md:col-span-3 xl:col-span-5 h-40", "md:col-span-3 xl:col-span-7 h-40"] }:
  { tiles?: string[] }) {
  const { t } = useTranslation();
  return (
    <div role="status" aria-live="polite" className="grid grid-cols-1 md:grid-cols-6 xl:grid-cols-12 gap-4">
      <span className="sr-only">{t("common.loading")}</span>
      {tiles.map((c, i) => <div key={i} className={`skeleton ${c}`} aria-hidden="true" />)}
    </div>
  );
}

export function ErrorBox({ onRetry }: { onRetry?: () => void }) {
  const { t } = useTranslation();
  return (
    <div role="alert" className="tile p-6 flex flex-col items-start gap-3">
      <p className="flex items-center gap-2"><AlertTriangle size={18} aria-hidden="true" />{t("common.error")}</p>
      {onRetry && <button className="btn btn-primary" onClick={onRetry}>{t("common.retry")}</button>}
    </div>
  );
}

/** Direction icon: water rising (blue) / falling (red-orange) / steady. Always next to words. */
export function DirMark({ dir, size = 16 }: { dir: "rise" | "fall" | "steady" | null; size?: number }) {
  if (!dir) return null;
  const [Icon, color] = { rise: [TrendingUp, "var(--rise)"], fall: [TrendingDown, "var(--fall)"],
                          steady: [Minus, "var(--text-muted)"] }[dir] as [LucideIcon, string];
  return <Icon size={size} aria-hidden="true" style={{ color }} className="inline-block shrink-0" />;
}
