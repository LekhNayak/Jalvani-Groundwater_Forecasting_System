# HMI checklist: each principle, where the site applies it, and how it was checked

The checks on 2026-10-31 used Playwright on the local app:
- desktop at 1280 px and phone at 390 px;
- all three languages;
- keyboard-only Tab order;
- the chat and tour flows;
- 0 console errors.

## Nielsen's 10 usability heuristics

| # | Heuristic | Where | Check |
|---|---|---|---|
| 1 | **Visibility of system status** | The freshness bar on every page (groundwater / rain, with the provisional date / soil / forecast). Loading and error states (`Loading`, `ErrorBox`). Chat "…" while waiting | screenshots |
| 2 | **Match with the real world** | "m below ground", depth (not a negative level), and a reversed chart axis (higher = water nearer the surface). "Likely between …". District names in the user's script, plus the official new names | screenshots |
| 3 | **User control and freedom** | Role and language switchable everywhere. Tours can be closed (Esc). Chat navigation needs a "Go there" click; it never moves the user by itself | Playwright |
| 4 | **Consistency** | ▼ in red-orange = deeper / falling, ▲ in blue = rising, everywhere. One set of status pills (✓ / ≈ / !) for reliability and the CGWB categories | review |
| 5 | **Error prevention** | District chosen from a filtered list (no free-text submit). Unknown routes go to start. Chat actions are allowlisted on the server | API tests |
| 6 | **Recognition over recall** | Language, role and district remembered. Suggestion chips in chat | Playwright (the remembered district opens directly) |
| 7 | **Flexibility** | Search by any name in any script (e.g. "नाग" → Nagpur). Sortable and filterable government table. CSV download | Playwright |
| 8 | **Minimalist design** | Farmer: three cards, then tips, then details behind "See details". Charts and the map load only when needed | screenshots |
| 9 | **Help recognise and recover from errors** | Plain error message with "Try again". Forecast-unavailable reasons in words | review |
| 10 | **Help and documentation** | Help page (cards, glossary, limits, weights), guided tours, the chat guide | Playwright (tour + highlight) |

## Shneiderman's 8 golden rules

| Rule | Where it is applied |
|---|---|
| Consistency | See Nielsen #4 |
| Shortcuts | Tour arrows ← / →; Enter picks the first district |
| Informative feedback | Pressed states (`aria-pressed`) on language, layer and horizon; highlighted tour targets |
| Closure | Tours end with "Done" |
| Prevent errors | Nielsen #5 |
| Easy reversal | Back links; role switch; "Change district" |
| User in control | Navigation never automatic |
| Low memory load | Nielsen #6; units repeated next to every number |

## Norman's principles
- **Affordances:** 44–52 px buttons that look like buttons.
- **Mapping:** map → district page; horizon buttons → chart.
- **Feedback:** loading text and highlight rings.
- **Constraints:** pick-lists only.
- **Conceptual model:** the same three cards in every district.

## Showing uncertainty honestly
- Forecasts are a range ("likely between"), never a bare number.
- Values are rounded to 0.1 m on the farmer view.
- A reliability badge appears on every forecast.
- Provisional data is tagged.
- "Usual" is labelled with its years.

## Accessibility (WCAG 2.1 AA targets)
- **Contrast:** text uses ink tokens, never series colours. Status colours always carry an icon and a word.
- **Colour-blind-safe:** blue vs red-orange plus ▲ / ▼. Grey + "no data" for missing values. Every map colour is repeated in the legend, the tooltip and the table.
- **Keyboard:**
  - a skip link;
  - a logical Tab order (checked);
  - a visible focus ring (3 px);
  - sortable headers with `aria-sort`;
  - radio groups for the layers and horizons;
  - the tour responds to Esc and the arrow keys.
- **Screen readers:**
  - `lang` set on `<html>` per language;
  - labelled landmarks and sections;
  - `aria-live` on chat and loading;
  - charts have a "Show as table" view.
- **Touch targets** ≥ 44 px.
- **Reduced motion** respected. Text can be scaled.

## Low literacy and low bandwidth (farmers)
- Icons with every tip, and big numbers.
- "Listen" on each card (device voice now, Gnani TTS later).
- The chat guide (voice later).
- First load is about 95 kB of gzipped JS. Charts and the map load only on demand. No map tiles are downloaded.

## Still to do before a public release
- A native-speaker review of the Marathi and Hindi text (`I18N_REVIEW.md`).
- An automated contrast audit (axe / Lighthouse) in CI.
- A dark-mode visual pass. The tokens exist, but this was not screenshotted.
- Usability sessions with 3–5 farmers and 2–3 officials.
