import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { GeoJSON, MapContainer } from "react-leaflet";
import type { Layer, PathOptions } from "leaflet";
import type { TFunction } from "i18next";
import { districtName } from "../i18n";
import { fmtNum } from "../lib/format";
import type { DistrictLite, Lang } from "../lib/types";

export type MapLayer = "vs_usual" | "forecast" | "extraction" | "wells" | "reliability";

// Diverging (blue = shallower / rising, red = deeper / falling, grey midpoint); status for categories;
// one-hue blue for counts. Every colour is repeated as words in the legend, the tooltip and the table.
const DIV = ["#b83232", "#e66767", "#f0efec", "#6da7ec", "#1c5cab"];
const STATUS: Record<string, string> = { good: "#0ca30c", fair: "#fab219", limited: "#ec835a",
  Safe: "#0ca30c", "Semi-Critical": "#fab219", Critical: "#ec835a", "Over-Exploited": "#d03b3b" };
const WELLS = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"];
const NODATA = "#d9d8d3";

function binDiverging(v: number | null, inner: number, outer: number) {
  if (v === null || v === undefined) return NODATA;
  if (v <= -outer) return DIV[0];
  if (v <= -inner) return DIV[1];
  if (v < inner) return DIV[2];
  if (v < outer) return DIV[3];
  return DIV[4];
}

export function layerColour(d: DistrictLite, layer: MapLayer): string {
  const L = d.layers;
  switch (layer) {
    case "vs_usual": return binDiverging(L.vs_usual_m, 0.5, 2);
    case "forecast": return binDiverging(L.forecast_change_t30_m, 0.25, 1);
    case "extraction": return L.category ? STATUS[L.category] ?? NODATA : NODATA;
    case "wells": {
      const w = L.wells ?? 0;
      return w <= 0 ? NODATA : w === 1 ? WELLS[0] : w <= 4 ? WELLS[1] : w <= 9 ? WELLS[2] : WELLS[3];
    }
    case "reliability": return L.reliability_t30 ? STATUS[L.reliability_t30] : NODATA;
  }
}

export function MapLegend({ layer }: { layer: MapLayer }) {
  const { t } = useTranslation();
  const items: [string, string][] = layer === "vs_usual"
    ? [[DIV[0], `${t("gov.legend.deeper")} ≥ 2 m`], [DIV[1], `${t("gov.legend.deeper")} 0.5–2 m`], [DIV[2], t("gov.legend.about")],
       [DIV[3], `${t("gov.legend.shallower")} 0.5–2 m`], [DIV[4], `${t("gov.legend.shallower")} ≥ 2 m`]]
    : layer === "forecast"
    ? [[DIV[0], `${t("gov.legend.fall")} ≥ 1 m`], [DIV[1], `${t("gov.legend.fall")} 0.25–1 m`], [DIV[2], t("gov.legend.steady")],
       [DIV[3], `${t("gov.legend.rise")} 0.25–1 m`], [DIV[4], `${t("gov.legend.rise")} ≥ 1 m`]]
    : layer === "extraction"
    ? (["Safe", "Semi-Critical", "Critical", "Over-Exploited"] as const).map((c) => [STATUS[c], t(`cat.${c}`)] as [string, string])
    : layer === "wells"
    ? [[WELLS[0], `1 ${t("gov.legend.wells")}`], [WELLS[1], `2–4`], [WELLS[2], `5–9`], [WELLS[3], `10+`]]
    : (["good", "fair", "limited"] as const).map((c) => [STATUS[c], t(`farmer.rel.${c}`)] as [string, string]);
  return (
    <ul className="flex flex-wrap gap-x-4 gap-y-1 text-sm text-ink2" aria-label={t(`gov.layer.${layer}`)}>
      {[...items, [NODATA, t("gov.legend.no_data")] as [string, string]].map(([c, label]) => (
        <li key={label} className="flex items-center gap-1.5">
          <span className="inline-block w-4 h-4 rounded-sm border border-[var(--border)]" style={{ background: c }} />{label}
        </li>
      ))}
    </ul>
  );
}

function valueText(d: DistrictLite, layer: MapLayer, t: TFunction, lang: Lang): string {
  const L = d.layers;
  const m = (v: number | null) => (v === null ? t("gov.legend.no_data") : `${v > 0 ? "+" : ""}${fmtNum(v, 1, lang)} m`);
  switch (layer) {
    case "vs_usual": return m(L.vs_usual_m);
    case "forecast": return m(L.forecast_change_t30_m);
    case "extraction": return L.stage_pct === null ? t("gov.legend.no_data") : `${fmtNum(L.stage_pct, 0, lang)}% · ${t(`cat.${L.category}`)}`;
    case "wells": return `${L.wells ?? 0} ${t("gov.legend.wells")}`;
    case "reliability": return L.reliability_t30 ? t(`farmer.rel.${L.reliability_t30}`) : t("gov.legend.no_data");
  }
}

export default function MapView({ geo, districts, layer }: { geo: GeoJSON.FeatureCollection; districts: DistrictLite[]; layer: MapLayer }) {
  const { t, i18n } = useTranslation();
  const lang = i18n.language as Lang;
  const nav = useNavigate();
  const byName = useMemo(() => Object.fromEntries(districts.map((d) => [d.district, d])), [districts]);
  const style = (f?: GeoJSON.Feature): PathOptions => {
    const d = f ? byName[f.properties?.district] : undefined;
    return { fillColor: d ? layerColour(d, layer) : NODATA, fillOpacity: 0.92, color: "#ffffff", weight: 1.2 };
  };
  const onEach = (f: GeoJSON.Feature, l: Layer) => {
    const d = byName[f.properties?.district];
    if (!d) return;
    l.bindTooltip(`<strong>${districtName(d.district, lang, false)}</strong><br/>${valueText(d, layer, t, lang)}`, { sticky: true });
    l.on({ click: () => nav(`/gov/district/${d.slug}`),
           mouseover: (e) => e.target.setStyle({ weight: 3, color: "#0b0b0b" }),
           mouseout: (e) => e.target.setStyle({ weight: 1.2, color: "#ffffff" }) });
  };
  return (
    <div data-tour="map" className="h-[420px] sm:h-[480px] rounded-xl overflow-hidden border border-[var(--border)]">
      <MapContainer bounds={[[15.6, 72.6], [22.1, 80.9]]} scrollWheelZoom={false} attributionControl={false} style={{ height: "100%" }}>
        {/* key forces a restyle and fresh tooltips when the layer or language changes */}
        <GeoJSON key={`${layer}-${lang}`} data={geo} style={style} onEachFeature={onEach} />
      </MapContainer>
    </div>
  );
}
