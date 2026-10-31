# Languages and translations

## Languages on the site
| Code | Language | Status | Voice (Gnani) |
|---|---|---|---|
| `en` | English | source text | yes |
| `mr` | मराठी Marathi | rewritten 2026-11-01 in everyday wording; native review pending | yes (round trip tested 2026-10-31) |
| `hi` | हिन्दी Hindi | rewritten 2026-11-01 in everyday wording; native review pending | yes |
| `gu` | ગુજરાતી Gujarati | added 2026-11-01; native review pending | yes (round trip tested 2026-11-01: the same sentence back, word for word) |

**Planned next:** Kannada, Telugu, Tamil, Malayalam, Bengali, Punjabi and Odia. Their district names are already in `districts.json`. To add one, see "How to add a language" below.

Users choose a language from the menu in the header, or from the large language buttons on the start page. The choice is remembered. Each language file is downloaded only when it is chosen (about 7 kB gzipped).

## How the text is written (not word for word)
The first Marathi and Hindi drafts read like English translated word by word. They were rewritten on 2026-11-01 to read as if written in that language for a farmer. The same rules apply to every new language:
- **Say it the way people speak.** Use the everyday word over the official one. Examples:
  - Hindi "अनुमान", not "पूर्वानुमान";
  - Hindi "भरोसा", not "विश्वसनीयता";
  - Marathi "भरवसा", not "विश्वासार्हता";
  - Marathi "जमिनीतील ओल", not "मृदा आर्द्रता".
- **Build the sentence in that language's order.** Do not follow the English structure. For example, "Deeper than usual for this time of year" becomes "या दिवसांत नेहमी असते त्यापेक्षा खोल".
- **Write units in full in sentences:** "मीटर", "મીટર". The short form ("मी.") is kept only in table headers and chart labels.
- **The two forecast ranges get plain names:**
  - "most likely" becomes "बहुतेक" (mr), "ज़्यादा संभावना" (hi), "મોટે ભાગે" (gu);
  - "could be" becomes "…पर्यंत असू शकते", "…तक हो सकता है", "…સુધી હોઈ શકે";
  - the chance is said in words: "दहापैकी आठ वेळा" (eight times in ten).
- **Keep the official terms where officials expect them.** The CGWB categories use the published terms:
  - Marathi (GSDA): सुरक्षित / अंशतः शोषित / शोषित / अतिशोषित;
  - Hindi (CGWB): सुरक्षित / अर्ध-क्रांतिक / क्रांतिक / अतिदोहित;
  - Gujarati: સલામત / અર્ધ-ગંભીર / ગંભીર / અતિ-શોષિત.
- **Keep commonly used loan words:** बोअरवेल, ड्रिप, पंप, CSV, CGWB, GSDA.
- **Numbers and dates use Latin digits everywhere,** for consistency with the charts. To switch to native digits, remove `-u-nu-latn` in `src/lib/format.ts`.

## What a native speaker should review
| File | What | Priority |
|---|---|---|
| `dev/frontend/src/i18n/{mr,hi,gu}.json` | Every label, card, tip, tour step and help text | **high**: the farmer view (`farmer.*`, `advice.*`, `chat.*`) first |
| `dev/backend/app/chat/faq.json` | Offline guide answers and keywords (the words people type or say) | high. The keywords must still match the chat suggestion chips; a test checks this |
| `dev/frontend/src/i18n/districts.json` | District names in each script, plus the official new names (Ahilyanagar, Chhatrapati Sambhajinagar, Dharashiv) | high |

Questions for the reviewer:
- Do farmers in your district say these words?
- Is anything stiff, too formal, or confusing?
- Do the two range names ("most likely" and "could be") make the difference clear?

## How to change a text
1. Edit the value in the language file. Keep the key and every `{{placeholder}}`.
2. Run `npm test` in `dev/frontend`. It checks that every English key exists in each language, with the same placeholders, and that no long text was left in English.
3. Run `python -m pytest` in `dev/backend`. It checks that every chat suggestion chip still finds its answer in the offline guide.
4. Reload the site.

## How to add a language
1. Add a row to `dev/frontend/src/lib/langs.ts` (code, name in its own script, English name, BCP-47 code).
2. Copy `en.json` to `<code>.json` and write every value, following the rules above.
3. Add `<code>` names to every district in `districts.json` (already done for the seven planned languages).
4. For a new script, install and import its Noto font: `@fontsource/noto-sans-<script>` in `main.tsx`, plus `tailwind.config.js`.
5. Backend:
   - add the code to `languages` and `speech_languages` in `app/config.py`, and to `LANG_CODES` in `app/services/speech/gnani.py`;
   - add the language name to `LANG_NAME` in `app/services/chat.py`;
   - add its keywords, answers and fallback to `app/chat/faq.json`.
6. Run both test suites. Check the voice with one live text-to-speech and speech-to-text round trip before listing it in `speech_languages`.
