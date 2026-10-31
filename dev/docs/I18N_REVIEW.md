# Translations: what needs a native speaker's review

The developer drafted the Marathi (`mr`) and Hindi (`hi`) text. Before showing the site to farmers or officials, a native speaker on the team should review these files:

| File | What | Priority |
|---|---|---|
| `dev/frontend/src/i18n/mr.json`, `hi.json` | Every label, card, tip, tour step and help text | **high**: the farmer view (`farmer.*`, `advice.*`, `chat.*`) first |
| `dev/frontend/src/i18n/districts.json` | District names in Devanagari, plus the official new names | high |
| `dev/backend/app/chat/faq.json` | Offline guide answers and the keywords people are likely to type or say | high: add the words farmers actually use |

## What to check
- **Plain, spoken words** a farmer uses, not officialese. Examples to confirm:
  - "भूजल पातळी" vs "पाणीपातळी";
  - "उपसा" for extraction;
  - "तात्पुरते" for provisional;
  - "मर्यादा" for range;
  - Hindi "दोहन" vs "निकासी".
- **Units:** "मी." / "मिमी" are used for metres and millimetres. Confirm these are understood, or use "मीटर" / "मिलीमीटर".
- **Numbers and dates:** Latin digits are used everywhere, for consistency with the charts. Confirm, or switch to Devanagari digits. That is one line in `src/lib/format.ts`: remove `-u-nu-latn`.
- **Category names:** the CGWB category names (सुरक्षित / अर्ध-गंभीर / गंभीर / अतिउपसा) should match the terms GSDA uses in Marathi.
- **The new district names** (Ahilyanagar, Chhatrapati Sambhajinagar, Dharashiv) are shown next to the CGWB names. Confirm the spelling.
- **Every `{{placeholder}}` stays in each translation.** A test (`npm test`) fails if one is missing.

## How to change a text
1. Edit the value in `mr.json` or `hi.json`. Keep the key and the `{{placeholders}}`.
2. Run `npm test`. It checks that every English key exists in each language, with the same placeholders.
3. Reload the site.
