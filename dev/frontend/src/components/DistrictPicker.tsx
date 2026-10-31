import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
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
    <div data-tour="district-picker" className="card p-4 sm:p-5">
      <h2 className="text-lg font-semibold mb-1">{t("farmer.pick_title")}</h2>
      <p className="text-sm text-ink2 mb-3">{t("farmer.pick_hint")}</p>
      <label htmlFor="district-search" className="sr-only">{t("farmer.pick_placeholder")}</label>
      <input id="district-search" type="search" value={q} onChange={(e) => setQ(e.target.value)} autoComplete="off"
        placeholder={t("farmer.pick_placeholder")}
        onKeyDown={(e) => { if (e.key === "Enter" && list[0]) onSelect(list[0].slug); }}
        className="w-full rounded-lg border border-[var(--border)] bg-surface px-3 py-3 text-lg min-h-[48px] mb-3" />
      {list.length === 0 && <p className="text-ink2">{t("farmer.no_results")}</p>}
      <ul className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-[50vh] overflow-y-auto" aria-label={t("farmer.pick_title")}>
        {list.map((d) => (
          <li key={d.slug}>
            <button onClick={() => onSelect(d.slug)}
              className="w-full text-left rounded-lg border border-[var(--border)] px-3 py-3 min-h-[48px] hover:bg-plane flex justify-between gap-2">
              <span>{d.label}</span>
              {d.status !== "forecast" && <span className="text-xs text-muted self-center">{t("gov.legend.no_data")}</span>}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
