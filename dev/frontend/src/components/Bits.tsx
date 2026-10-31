import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { Reliability } from "../lib/types";

/** Status colours always come with an icon and a word (never colour alone). */
const STATUS_STYLE: Record<string, { bg: string; icon: string }> = {
  good: { bg: "var(--status-good)", icon: "✓" },
  fair: { bg: "var(--status-warning)", icon: "≈" },
  limited: { bg: "var(--status-serious)", icon: "!" },
  Safe: { bg: "var(--status-good)", icon: "✓" },
  "Semi-Critical": { bg: "var(--status-warning)", icon: "!" },
  Critical: { bg: "var(--status-serious)", icon: "!!" },
  "Over-Exploited": { bg: "var(--status-critical)", icon: "✕" },
};

export function StatusPill({ kind, label }: { kind: string; label: string }) {
  const s = STATUS_STYLE[kind] ?? { bg: "var(--nodata)", icon: "–" };
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-sm font-semibold border border-[var(--border)]">
      <span aria-hidden="true" className="inline-flex h-5 w-5 items-center justify-center rounded-full text-xs text-[#0b0b0b]"
        style={{ background: s.bg }}>{s.icon}</span>
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

export function Card({ title, children, tour, actions, className = "" }:
  { title: ReactNode; children: ReactNode; tour?: string; actions?: ReactNode; className?: string }) {
  return (
    <section data-tour={tour} className={`card p-4 sm:p-5 ${className}`} aria-label={typeof title === "string" ? title : undefined}>
      <div className="flex items-start justify-between gap-2 mb-2">
        <h2 className="text-lg font-semibold">{title}</h2>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Loading() {
  const { t } = useTranslation();
  return <p role="status" aria-live="polite" className="text-ink2 py-8 text-center">{t("common.loading")}</p>;
}

export function ErrorBox({ onRetry }: { onRetry?: () => void }) {
  const { t } = useTranslation();
  return (
    <div role="alert" className="card p-4 text-center">
      <p>{t("common.error")}</p>
      {onRetry && <button className="mt-2 px-4 py-2 rounded-lg bg-water text-white min-h-[44px]" onClick={onRetry}>{t("common.retry")}</button>}
    </div>
  );
}

/** ▲ / ▼ / ● direction mark with a word: water rising (blue) / falling (red-orange) / steady. */
export function DirMark({ dir }: { dir: "rise" | "fall" | "steady" | null }) {
  if (!dir) return null;
  const m = { rise: ["▲", "var(--rise)"], fall: ["▼", "var(--fall)"], steady: ["●", "var(--text-muted)"] }[dir];
  return <span aria-hidden="true" style={{ color: m[1] }} className="mr-1">{m[0]}</span>;
}
