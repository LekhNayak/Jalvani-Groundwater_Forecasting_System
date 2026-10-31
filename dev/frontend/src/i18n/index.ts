import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import en from "./en.json";
import districts from "./districts.json";
import { isLang, type Lang } from "../lib/langs";

const KEY = "jalvaani.lang";

// English is bundled (it is also the fallback); every other language is a separate file, fetched only when chosen.
const LOADERS = import.meta.glob<{ default: Record<string, unknown> }>(["./*.json", "!./en.json", "!./districts.json"]);

function savedLang(): Lang {
  try {
    const v = localStorage.getItem(KEY);
    if (isLang(v)) return v;
  } catch { /* storage blocked: fall back to English */ }
  return "en";
}

i18n.use(initReactI18next).init({
  resources: { en: { translation: en } },
  lng: "en",
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});

async function load(lang: Lang) {
  if (lang === "en" || i18n.hasResourceBundle(lang, "translation")) return;
  const loader = LOADERS[`./${lang}.json`];
  if (!loader) return;
  i18n.addResourceBundle(lang, "translation", (await loader()).default);
}

/** Switch language: fetch its file (first time only), then switch everything, including <html lang>. */
export async function setLang(lang: Lang) {
  await load(lang);
  await i18n.changeLanguage(lang);
  document.documentElement.lang = lang;
  try { localStorage.setItem(KEY, lang); } catch { /* ignore */ }
}

/** Called once before the first render, so the remembered language shows without a flash of English. */
export async function initLang() {
  const lang = savedLang();
  try { await setLang(lang); } catch { await setLang("en"); }
}

type Names = Partial<Record<Lang, string>>;
type DistrictNames = Record<string, Names & { aka?: Names }>;
const DN = districts as unknown as DistrictNames;

/** District name in the chosen language, with the official new name when there is one. */
export function districtName(name: string, lang: Lang, withAka = true): string {
  const d = DN[name];
  if (!d) return name;
  const base = lang === "en" ? name : d[lang] ?? name;
  const aka = d.aka?.[lang] ?? d.aka?.en;
  return withAka && aka ? `${base} (${aka})` : base;
}

/** Every name a district can be searched by (all languages, old and new). */
export function districtSearchTerms(name: string): string {
  const d = DN[name];
  if (!d) return name.toLowerCase();
  const { aka, ...names } = d;
  return [name, ...Object.values(names), ...(aka ? Object.values(aka) : [])].join(" ").toLowerCase();
}

export default i18n;
