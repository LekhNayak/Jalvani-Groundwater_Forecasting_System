import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";
import { LangSwitch } from "../components/Header";
import { useApp } from "../context";
import type { Role } from "../lib/types";

export default function StartPage() {
  const { t } = useTranslation();
  const { setRole } = useApp();
  const nav = useNavigate();
  const choose = (r: Role) => { setRole(r); nav(r === "farmer" ? "/farmer" : "/gov"); };
  return (
    <div className="max-w-3xl mx-auto py-6">
      <h1 className="text-3xl font-semibold mb-2">{t("start.welcome")}</h1>
      <p className="text-ink2 text-lg mb-8">{t("start.intro")}</p>
      <h2 className="text-lg font-semibold mb-3">{t("start.choose_language")}</h2>
      <div className="mb-8"><LangSwitch big /></div>
      <h2 className="text-lg font-semibold mb-3">{t("start.choose_role")}</h2>
      <div className="grid sm:grid-cols-2 gap-4">
        {(["farmer", "government"] as Role[]).map((r) => (
          <button key={r} onClick={() => choose(r)} className="card p-5 text-left hover:shadow-md min-h-[120px]">
            <div className="text-4xl mb-2" aria-hidden="true">{r === "farmer" ? "🌾" : "🏛️"}</div>
            <div className="text-xl font-semibold">{t(r === "farmer" ? "common.role_farmer" : "common.role_government")}</div>
            <div className="text-ink2">{t(r === "farmer" ? "start.farmer_desc" : "start.government_desc")}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
