/** Every language the site offers. To add one: a row here, `src/i18n/<code>.json` with every English key, the district
 *  names in `districts.json`, the backend `languages` setting, and (for a new script) a Noto font in main.tsx.
 *  Planned next: Kannada, Telugu, Tamil, Malayalam, Bengali, Punjabi, Odia (district names are already in districts.json). */
export const LANGS = [
  { code: "en", native: "English", english: "English", bcp47: "en-IN" },
  { code: "mr", native: "मराठी", english: "Marathi", bcp47: "mr-IN" },
  { code: "hi", native: "हिन्दी", english: "Hindi", bcp47: "hi-IN" },
  { code: "gu", native: "ગુજરાતી", english: "Gujarati", bcp47: "gu-IN" },
] as const;

export type Lang = (typeof LANGS)[number]["code"];
export const LANG_CODES: Lang[] = LANGS.map((l) => l.code);
export const isLang = (v: unknown): v is Lang => typeof v === "string" && (LANG_CODES as string[]).includes(v);
export const bcp47 = (l: string) => LANGS.find((x) => x.code === l)?.bcp47 ?? "en-IN";
