import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { BookOpen, Building2, CircleAlert, Compass, PlayCircle, Scale, Sprout } from "lucide-react";
import { Tile } from "../components/Bits";
import { useApp } from "../context";
import { api, useApi } from "../lib/api";

/** "Term: explanation" -> bold term + text (the same strings feed the chat guide, so they stay one sentence each) */
function Entry({ text }: { text: string }) {
  const i = text.indexOf(":");
  if (i < 0 || i > 60) return <>{text}</>;
  return <><span className="font-semibold text-ink">{text.slice(0, i)}</span><span className="text-ink2">{text.slice(i)}</span></>;
}

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
    <ol className="space-y-3">
      {keys.map((k, i) => (
        <li key={k} className="flex gap-3 text-[0.95rem]">
          <span className="inline-flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-[var(--surface-2)] text-xs font-semibold tabnum" aria-hidden="true">{i + 1}</span>
          <span><Entry text={t(`${base}.${k}`)} /></span>
        </li>
      ))}
    </ol>
  );
  const glossary = ["depth", "usual", "most_likely", "range", "reliability", "extraction", "provisional"];
  return (
    <div className="grid grid-cols-1 md:grid-cols-6 xl:grid-cols-12 gap-4">
      <section className="tile p-6 sm:p-8 md:col-span-6 xl:col-span-8 fade-in">
        <h1 className="text-[1.75rem] leading-tight font-semibold tracking-tight">{t("help.title")}</h1>
        <p className="text-ink2 mt-3 max-w-3xl">{t("help.intro")}</p>
      </section>
      <Tile title={t("help.tours_title")} icon={Compass} className="md:col-span-6 xl:col-span-4" bodyClass="flex flex-col gap-2">
        <button className="btn justify-start" onClick={() => tour("farmer")}><Sprout size={17} aria-hidden="true" />{t("help.tour_farmer")}
          <PlayCircle size={17} aria-hidden="true" className="ml-auto text-water" /></button>
        <button className="btn justify-start" onClick={() => tour("government")}><Building2 size={17} aria-hidden="true" />{t("help.tour_government")}
          <PlayCircle size={17} aria-hidden="true" className="ml-auto text-water" /></button>
      </Tile>

      <Tile title={t("help.farmer_title")} icon={Sprout} className="md:col-span-3 xl:col-span-6">{list("help.farmer_items", ["water", "forecast", "rain"])}</Tile>
      <Tile title={t("help.gov_title")} icon={Building2} className="md:col-span-3 xl:col-span-6">{list("help.gov_items", ["kpi", "map", "table", "needs"])}</Tile>

      <Tile title={t("help.glossary_title")} icon={BookOpen} className="col-span-full">
        <dl className="grid sm:grid-cols-2 xl:grid-cols-3 gap-x-8 gap-y-4 text-[0.95rem]">
          {glossary.map((k) => <div key={k}><Entry text={t(`help.glossary.${k}`)} /></div>)}
        </dl>
      </Tile>

      <Tile title={t("help.limits_title")} icon={CircleAlert} className="md:col-span-6 xl:col-span-5">
        <ul className="space-y-2.5 text-[0.95rem] text-ink2 list-disc pl-5">
          {["district", "readings", "advice", "nodata"].map((k) => <li key={k}>{t(`help.limits.${k}`)}</li>)}
        </ul>
      </Tile>
      {ov.data && (
        <Tile title={t("help.weights_title")} icon={Scale} className="md:col-span-6 xl:col-span-7">
          <table className="w-full text-sm">
            <tbody>{Object.entries(ov.data.weights).map(([k, w]) => (
              <tr key={k} className="border-t border-[var(--gridline)] first:border-t-0">
                <td className="py-2 pr-3">{t(`gov.flag.${k}`, { stage_pct: "…", fall_m: "…", diff_m: "…", pct: "…" })}</td>
                <td className="py-2 pr-3 tabnum text-right font-semibold">+{w.weight}</td><td className="py-2 text-ink2">{t(`gov.action.${w.action}`)}</td></tr>))}</tbody>
          </table>
        </Tile>
      )}
    </div>
  );
}
