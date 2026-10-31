import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./en.json";
import hi from "./hi.json";
import mr from "./mr.json";
import districts from "./districts.json";
import type { Lang } from "../lib/types";

const KEY = "jalvaani.lang";

function savedLang(): Lang {
  try {
    const v = localStorage.getItem(KEY);
    if (v === "en" || v === "mr" || v === "hi") return v;
  } catch { /* storage blocked: fall back to English */ }
  return "en";
}

i18n.use(initReactI18next).init({
  resources: { en: { translation: en }, mr: { translation: mr }, hi: { translation: hi } },
  lng: savedLang(),
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});
document.documentElement.lang = i18n.language;

export function setLang(lang: Lang) {
  i18n.changeLanguage(lang);
  document.documentElement.lang = lang;
  try { localStorage.setItem(KEY, lang); } catch { /* ignore */ }
}

type DistrictNames = Record<string, { mr: string; hi: string; aka?: Record<Lang, string> }>;
const DN = districts as unknown as DistrictNames;

/** District name in the chosen language, with the official new name when there is one. */
export function districtName(name: string, lang: Lang, withAka = true): string {
  const d = DN[name];
  if (!d) return name;
  const base = lang === "en" ? name : d[lang] ?? name;
  return withAka && d.aka ? `${base} (${d.aka[lang]})` : base;
}

/** Every name a district can be searched by (all languages, old and new). */
export function districtSearchTerms(name: string): string {
  const d = DN[name];
  if (!d) return name.toLowerCase();
  return [name, d.mr, d.hi, ...(d.aka ? Object.values(d.aka) : [])].join(" ").toLowerCase();
}

export default i18n;
