"""Chat assistant: navigation help + explaining the current screen.

Two modes:
- "groq":  GROQ_API_KEY (and optionally GROQ_API_KEY_2, _3, ...) set. Groq's OpenAI-compatible chat API, JSON-mode
           replies {reply, actions}. Keys are used in turn; a rate-limited or failing key passes the request on.
- "guide": no key. An offline intent matcher over app/chat/faq.json (en / mr / hi), with the same actions.
Every action is checked against an allowlist before it reaches the browser.
"""
import itertools
import json
import re
from functools import lru_cache
from pathlib import Path

import httpx

from ..config import settings

CHAT_DIR = Path(__file__).resolve().parents[1] / "chat"
LANG_NAME = {"en": "English", "mr": "Marathi (मराठी)", "hi": "Hindi (हिन्दी)", "gu": "Gujarati (ગુજરાતી)"}
ELEMENTS = {"lang-switch", "theme-switch", "role-switch", "help-link", "freshness-bar", "chat-button", "district-picker",
            "card-water-now", "card-forecast", "reliability-badge", "card-rain-soil", "advice-list", "details-chart",
            "listen-button", "kpi-row", "map", "map-layers", "glance-panel", "priority-table", "needs-panel", "download-csv",
            "gov-history-chart", "gov-extraction", "gov-wells", "gov-metrics"}
TOURS = {"farmer", "government"}
ROUTE_RE = re.compile(r"^/(farmer(/[a-z_]+)?|gov(/district/[a-z_]+)?|help)?$")
MAX_CONTEXT_CHARS = 4000
RETRY_STATUS = {401, 403, 429, 500, 502, 503, 504}
_next_key = itertools.count()


@lru_cache(maxsize=1)
def site_guide():
    return (CHAT_DIR / "site_guide.md").read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def faq():
    return json.loads((CHAT_DIR / "faq.json").read_text(encoding="utf-8"))


def clean_actions(actions, role):
    out = []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        t = a.get("type")
        if t == "navigate" and isinstance(a.get("route"), str) and ROUTE_RE.match(a["route"]):
            out.append({"type": "navigate", "route": a["route"]})
        elif t == "highlight" and a.get("element") in ELEMENTS:
            out.append({"type": "highlight", "element": a["element"]})
        elif t == "start_tour":
            tour = a.get("tour")
            if tour == "@role":
                tour = "government" if role == "government" else "farmer"
            if tour in TOURS:
                out.append({"type": "start_tour", "tour": tour})
    return out[:3]


def guide_reply(message, lang, role):
    text = f" {message.lower()} "
    best, best_score = None, 0
    for intent in faq()["intents"]:
        score = 0
        for lk in (lang, "en"):
            for kw in intent["keywords"].get(lk, []):
                if kw.lower() in text:
                    score += 2 if lk == lang else 1
        if score > best_score:
            best, best_score = intent, score
    if best is None:
        return {"reply": faq()["fallback"].get(lang, faq()["fallback"]["en"]), "actions": [], "mode": "guide"}
    return {"reply": best["answer"].get(lang, best["answer"]["en"]),
            "actions": clean_actions(best.get("actions"), role), "mode": "guide", "intent": best["id"]}


def system_prompt(lang, role, page, district, visible_context):
    ctx = json.dumps(visible_context or {}, ensure_ascii=False)[:MAX_CONTEXT_CHARS]
    return f"""You are the JalVaani website guide. You help {"government officers" if role == "government" else "farmers"} use
the site and explain what is on their screen. Reply in {LANG_NAME.get(lang, "English")}, in at most 4 short, plain sentences.

Rules:
- Only explain this website and the numbers in VISIBLE_CONTEXT. Never invent numbers, dates or places; if a value is
  not in VISIBLE_CONTEXT, say where on the site to find it.
- Do not give crop, borewell, subsidy, legal or medical advice: refer to the local agriculture office, the GSDA or
  the district administration.
- Be honest about limits: district level only; groundwater readings end on the date shown; forecasts are uncertain.
- Do not ask for or repeat personal information.
- Never show element ids, routes or JSON in the reply text; name cards by their visible titles (e.g. "the forecast card").

You may also propose up to 3 actions for the site to perform:
  {{"type":"navigate","route":"/farmer/<slug>|/gov|/gov/district/<slug>|/help|/farmer"}},
  {{"type":"highlight","element":"<element id from the guide>"}},
  {{"type":"start_tour","tour":"farmer|government"}}.
Answer ONLY with JSON: {{"reply": "...", "actions": [...]}}.

SITE GUIDE:
{site_guide()}

CURRENT PAGE: {page or "/"}   DISTRICT: {district or "-"}
VISIBLE_CONTEXT (what the user sees now): {ctx}"""


def groq_reply(req):
    msgs = [{"role": "system", "content": system_prompt(req.lang, req.role, req.page, req.district, req.visible_context)}]
    for m in (req.history or [])[-8:]:
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
            msgs.append({"role": m["role"], "content": m["content"][:1000]})
    msgs.append({"role": "user", "content": req.message[:1000]})
    payload = {"model": settings.groq_model, "messages": msgs, "temperature": 0.2, "max_tokens": 400,
               "response_format": {"type": "json_object"}}
    keys = settings.groq_api_keys
    start = next(_next_key) % len(keys)            # round-robin spreads the load over the keys
    resp = None
    with httpx.Client(timeout=20.0) as client:
        for i in range(len(keys)):
            key = keys[(start + i) % len(keys)]
            resp = client.post(settings.groq_url, json=payload, headers={"Authorization": f"Bearer {key}"})
            if resp.status_code not in RETRY_STATUS:
                break
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    try:
        data = json.loads(content)
        reply, actions = str(data.get("reply", "")).strip(), data.get("actions", [])
    except (json.JSONDecodeError, AttributeError):
        reply, actions = content.strip(), []
    return {"reply": reply or guide_reply(req.message, req.lang, req.role)["reply"],
            "actions": clean_actions(actions, req.role), "mode": "groq"}


def chat(req):
    if not settings.groq_api_keys:
        return guide_reply(req.message, req.lang, req.role)
    try:
        return groq_reply(req)
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        out = guide_reply(req.message, req.lang, req.role)
        out["note"] = "assistant unavailable, answered from the offline guide"
        return out
