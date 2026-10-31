# The JalVaani site guide (chatbot)

**Job.** It helps people use the site:
- where to find things;
- what a card or a number means;
- guided tours.

It is **not** a data oracle and gives no farming, borewell, scheme or legal advice. For those it refers people to the local agriculture office, the GSDA or the district administration.

## How it works
```
ChatWidget (frontend) ──POST /api/chat {message, lang, role, page, district, visible_context, history}──▶ backend
backend: GROQ_API_KEY set?  yes → Groq chat completion (JSON mode) → {reply, actions}
                            no  → offline guide: keyword intents in app/chat/faq.json (en / mr / hi) → {reply, actions}
         actions are checked against an allowlist (clean_actions) → frontend carries them out
```

**`visible_context`** is what the user sees now: the numbers on the current page, from `setVisible` in each page. The model is told to use only those numbers, so it cannot invent values.

**Actions** (allowlisted on the server):

| Action | What it does |
|---|---|
| `navigate(route)` | Shown as a "Go there" button. It never navigates by itself |
| `highlight(element)` | Rings one element and explains it |
| `start_tour(farmer \| government)` | Starts a guided tour |

The routes and element ids are listed in `backend/app/chat/site_guide.md`, and must stay in sync with the `data-tour` attributes in the frontend.

**Languages.** The reply is in the user's selected language (en / mr / hi). In guide mode, keywords match in the selected language first, then in English.

## Turning on Groq
1. Get an API key from the Groq console.
2. Put it in `dev/backend/.env` as `GROQ_API_KEY=...`. Never in the frontend; never commit `.env`.
   - Optional: add `GROQ_API_KEY_2`, `GROQ_API_KEY_3` (up to `_9`).
   - The keys are used round-robin. A key that answers 401, 403, 429 or 5xx passes the request to the next key.
3. Set `GROQ_MODEL` to a model your key can use. List them with `GET https://api.groq.com/openai/v1/models`.
   - **Tested 2026-10-31:** `qwen/qwen3.8-27b` works (JSON replies in Marathi, Hindi and English, about 1.3 s).
   - `llama-3.3-70b-versatile` is no longer offered (404 `model_not_found`).
   - `openai/gpt-oss-120b` rejected the JSON-mode request (400).
4. Restart the backend. `/api/meta` then reports `"chat": "groq"`, and the panel's subtitle changes from "Offline guide" to "Assistant".

**Safety settings:**
- temperature 0.2;
- at most 400 tokens;
- a 20 s timeout;
- history capped at 8 turns and 1,000 characters per message;
- context capped at 4,000 characters.

On any Groq error the backend answers from the offline guide, so the user always gets a reply.

## Prompt rules (in `services/chat.py`)
- Reply in the user's language, in at most 4 short sentences.
- Only explain the site and the numbers in `VISIBLE_CONTEXT`. Never invent numbers, dates or places.
- No crop, borewell, subsidy, legal or medical advice; refer to local officials.
- Be honest about the limits: district level only, the date readings end, forecast uncertainty.
- Do not ask for or repeat personal information.
- Answer only with JSON `{reply, actions}`.

## Testing
- `backend/tests/test_api.py` covers:
  - guide mode in all three languages;
  - the tour action;
  - the action allowlist (rejects external URLs, unknown elements and unknown action types).
- **With a key, check by hand:**
  - ask in each language about each card;
  - ask for a number that is not on screen: it must point to where to find it, not guess;
  - ask a borewell or subsidy question: it must refer to officials.

## Not built yet (ideas)
- **Logging of unanswered questions.** It must be opt-in and store no personal data. It would improve `faq.json`.
- **Streaming replies.**
