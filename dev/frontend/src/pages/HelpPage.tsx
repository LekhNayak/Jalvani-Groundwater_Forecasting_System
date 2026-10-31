import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { Card } from "../components/Bits";
import { useApp } from "../context";
import { api, useApi } from "../lib/api";

export default function HelpPage() {
  const { t } = useTranslation();
  const nav = useNavigate();
  const { startTour, savedDistrict, setRole } = useApp();
  const ov = useApi(api.overview, "overview-help");
  const tour = (name: "farmer" | "government") => {
    setRole(name);
    nav(name === "farmer" ? (savedDistrict ? `/farmer/${savedDistrict}` : "/farmer") : "/gov");
    setTimeout(() => startTour(name), 900);
  };
  const list = (base: string, keys: string[]) => (
    <ul className="list-disc pl-5 space-y-2">{keys.map((k) => <li key={k}>{t(`${base}.${k}`)}</li>)}</ul>
  );
  return (
    <div className="space-y-4 max-w-3xl">
      <h1 className="text-2xl font-semibold">{t("help.title")}</h1>
      <p className="text-ink2">{t("help.intro")}</p>
      <Card title={t("help.tours_title")}>
        <div className="flex flex-wrap gap-2">
          <button className="px-4 py-2 min-h-[44px] rounded-lg bg-water text-white" onClick={() => tour("farmer")}>{t("help.tour_farmer")}</button>
          <button className="px-4 py-2 min-h-[44px] rounded-lg bg-water text-white" onClick={() => tour("government")}>{t("help.tour_government")}</button>
        </div>
      </Card>
      <Card title={t("help.farmer_title")}>{list("help.farmer_items", ["water", "forecast", "rain"])}</Card>
      <Card title={t("help.gov_title")}>{list("help.gov_items", ["kpi", "map", "table", "needs"])}</Card>
      <Card title={t("help.glossary_title")}>{list("help.glossary", ["depth", "usual", "range", "reliability", "extraction", "provisional"])}</Card>
      <Card title={t("help.limits_title")}>{list("help.limits", ["district", "readings", "advice", "nodata"])}</Card>
      {ov.data && (
        <Card title={t("help.weights_title")}>
          <table className="w-full text-sm">
            <tbody>{Object.entries(ov.data.weights).map(([k, w]) => (
              <tr key={k} className="border-t border-[var(--gridline)]">
                <td className="py-1 pr-3">{t(`gov.flag.${k}`, { stage_pct: "…", fall_m: "…", diff_m: "…", pct: "…" })}</td>
                <td className="py-1 pr-3 tabnum">+{w.weight}</td><td className="py-1 text-ink2">{t(`gov.action.${w.action}`)}</td></tr>))}</tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
