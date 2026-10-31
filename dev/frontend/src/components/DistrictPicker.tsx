import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronRight, MapPin, Search } from "lucide-react";
import { districtName, districtSearchTerms } from "../i18n";
import type { DistrictLite, Lang } from "../lib/types";

/** Search-as-you-type list (no free text is ever submitted: the user always picks from the list). */
export default function DistrictPicker({ districts, onSelect }: { districts: DistrictLite[]; onSelect: (slug: string) => void }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const [q, setQ] = useState("");
  const list = useMemo(() => {
    const s = q.trim().toLowerCase();
    const items = districts.map((d) => ({ ...d, label: districtName(d.district, lang) }))
      .sort((a, b) => a.label.localeCompare(b.label, lang));
    return s ? items.filter((d) => districtSearchTerms(d.district).includes(s)) : items;
  }, [districts, lang, q]);

  return (
    <section data-tour="district-picker" className="tile p-5 sm:p-7 fade-in" aria-labelledby="pick-h">
      <div className="flex flex-wrap items-end justify-between gap-4 mb-5">
        <div>
          <h1 id="pick-h" className="text-2xl font-semibold tracking-tight">{t("farmer.pick_title")}</h1>
          <p className="text-ink2 mt-1">{t("farmer.pick_hint")}</p>
        </div>
        <div className="relative w-full md:w-[420px]">
          <Search size={18} aria-hidden="true" className="absolute left-3.5 top-1/2 -translate-y-1/2 text-muted" />
          <label htmlFor="district-search" className="sr-only">{t("farmer.pick_placeholder")}</label>
          <input id="district-search" type="search" value={q} onChange={(e) => setQ(e.target.value)} autoComplete="off" autoFocus
            placeholder={t("farmer.pick_placeholder")}
            onKeyDown={(e) => { if (e.key === "Enter" && list[0]) onSelect(list[0].slug); }}
            className="w-full rounded-xl border border-[var(--border)] bg-[var(--surface-2)] pl-11 pr-3 py-3 text-lg min-h-[52px]" />
        </div>
      </div>
      {list.length === 0 && <p className="text-ink2 py-6">{t("farmer.no_results")}</p>}
      <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-6 gap-2" aria-label={t("farmer.pick_title")}>
        {list.map((d) => (
          <li key={d.slug} className="flex">
            <button onClick={() => onSelect(d.slug)}
              className="group w-full h-full text-left rounded-xl border border-[var(--border)] px-3.5 py-3 min-h-[56px] bg-surface
                         hover:bg-[var(--accent-soft)] hover:border-transparent transition-colors flex items-center gap-2.5">
              <MapPin size={16} aria-hidden="true" className={d.status === "forecast" ? "text-water" : "text-muted"} />
              <span className="flex-1 min-w-0">
                <span className="block leading-snug">{d.label}</span>
                {d.status !== "forecast" && <span className="block text-xs text-muted">{t("gov.legend.no_data")}</span>}
              </span>
              <ChevronRight size={16} aria-hidden="true" className="text-muted opacity-0 group-hover:opacity-100 transition-opacity" />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
