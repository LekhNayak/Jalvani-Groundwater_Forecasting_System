# Plan: voice (Gnani STT and TTS) for the site guide and the "Listen" buttons

**Goal.** A farmer who prefers speaking or listening can:
- ask the guide by voice;
- hear its answers;
- have any card read aloud in Marathi, Hindi, Gujarati or English.

**Status (2026-10-31): implemented and tested live.**
- **One Gnani API key** covers both directions. It is sent in the `X-API-Key-ID` header to Gnani Vachana (https://docs.gnani.ai).
- **Speech-to-text:** `POST https://api.vachana.ai/stt/v3` (multipart `audio_file` + `language_code`, model `prisma-v3.0`).
- **Text-to-speech:** `POST https://api.vachana.ai/api/v1/tts/inference` (JSON: text, voice, model `timbre-v2.5`, language, `audio_config` = WAV).
- **The browser converts its recording to 16 kHz mono WAV** (`src/lib/wav.ts`). Gnani accepts WAV/MP3/OGG/FLAC/AAC/M4A up to 30 s, and the app stops recording at 25 s.
- **Live round trip in Marathi:** "नमस्कार, जलवाणीमध्ये आपले स्वागत आहे." → TTS (0.9 s, WAV) → STT (0.9 s) → the same sentence back.
- **Live round trip in Gujarati (2026-11-01):** "નમસ્તે, જલવાણીમાં આપનું સ્વાગત છે." → TTS (voice `Nalini`, `gu-IN`) → STT → the same sentence back. `/api/meta` lists the voice languages in `features.speech_languages`; the mic and "Listen" buttons appear only for those.
- **Still to do:**
  - choose a Marathi voice (`GNANI_TTS_VOICE_MR`, from Gnani's voice catalog; the default is `Nalini`);
  - field tests with real speakers (below).

The original plan follows; the parts marked "fill in" are done.

## User flows

**Ask by voice:**
1. Tap 🎤. It turns red while recording; there is a consent notice the first time.
2. Speak. Tap again to stop. The maximum is about 20 s.
3. The audio goes to `POST /api/speech/stt` (`audio`, `lang`), then Gnani STT.
4. Recording **stops by itself** about 1.2 s after the user stops talking (voice-activity check on the microphone level; 7 s with no speech stops it with a "did not hear anything" message; 25 s is the hard limit). Tapping the mic again stops it early.
5. The transcript is **sent to the guide at once and shown as the user's own message**, so nothing is sent unseen and the user can see what was understood. (Until 2026-11-01 the text waited in the input box for a Send tap; removing that extra step made voice about 1 s faster and hands-free.)

**Where the time goes** (measured 2026-11-01, Marathi, one sentence): speech-to-text about 0.6 s + about 0.5 s of browser conversion; the chat model about 0.8 s. Most of the old delay was waiting for the user to tap stop (up to 25 s if forgotten) and tap Send, not the services. Hands-free, a reply arrives about 3 s after the user stops talking. The server reuses its connections to Gnani and Groq, and speech-to-text no longer blocks other requests.
5. The reply is shown as text, always, and read aloud through `POST /api/speech/tts` (Gnani TTS) if the user has turned on "read replies".

**Listen to a card:** "Listen" → `POST /api/speech/tts` with the card's summary text → audio plays → "Stop" ends it.

## Backend (built; fill in the Gnani details)
- **`backend/app/services/speech/base.py`:** `SpeechProvider` with `transcribe(audio, mime, lang) -> text` and `synthesize(text, lang) -> (bytes, mime)`.
- **`backend/app/services/speech/gnani.py`:** a `GnaniProvider` skeleton. It deliberately has **no guessed endpoints**. Steps to finish it:
  1. Put `GNANI_API_KEY`, `GNANI_STT_URL` and `GNANI_TTS_URL` in `dev/backend/.env`.
  2. From Gnani's API documentation, take:
     - the auth header format;
     - the request format (multipart / JSON / streaming);
     - the accepted audio encoding and sample rate;
     - the language codes (`LANG_CODES`: mr-IN, hi-IN, en-IN, to confirm);
     - the voice names for TTS.
  3. Audio conversion. Browsers record WebM/Opus (Chrome) or MP4/AAC (Safari). If Gnani needs, say, 16 kHz mono PCM WAV, convert on the server, e.g. `ffmpeg -i in.webm -ac 1 -ar 16000 out.wav`. This adds ffmpeg as a dependency.
  4. Implement `transcribe` and `synthesize` with `httpx`, with a timeout and one retry.
  5. Run `python -m pytest`, plus a manual test with real recordings.
- **Endpoints:** `POST /api/speech/stt`, `POST /api/speech/tts`. Both answer 501 until configured. `/api/meta` reports `features.speech`.

## Frontend (built)
- **Mic button** in `ChatWidget.tsx`: shown only when `features.speech` is true and the language is in `features.speech_languages`. It uses MediaRecorder with silence detection, shows "Listening…" and "Understanding…", and sends the transcript to the guide.
- **`ListenButton.tsx`:** uses Gnani TTS when configured. Otherwise it uses the browser's `speechSynthesis` if the device has a voice for the language; otherwise it is hidden. Marathi voices are often missing on PCs, which is why Gnani matters.

## Design decisions to confirm with Gnani's docs
| Topic | Plan |
|---|---|
| Batch or streaming STT | Batch first (short questions); streaming later if the latency is poor |
| Latency budget | STT ≤ 2 s, chat ≤ 3 s, TTS first audio ≤ 1.5 s. Show "Listening…" and "…" states |
| Text length for TTS | Split long text into sentences. Cache the audio of fixed help texts (card explanations) by hash |
| Languages | mr-IN first (most farmers), then hi-IN, en-IN. The UI language decides the STT language |
| Privacy | Audio is not stored. It is sent to Gnani only for the request. A consent notice on first use. No personal data in logs |
| Cost and limits | Rate-limit per browser session. Cap the recording length. Check Gnani's pricing and quotas before any public demo |
| Failure | Any STT/TTS error falls back to text ("Sorry, I couldn't hear that, please type"). The chat always works without voice |
| Accessibility | The captions (reply text) are always visible. The mic button has `aria-pressed` and a label. Keyboard: Space toggles recording |

## Testing plan
- **Recordings.** At least 20 short questions per language, from different speakers, in quiet and noisy places (a farm, a road), on two phones. Measure word accuracy on the key terms: district names, पाऊस, पाणीपातळी, अंदाज, …
- **TTS.** Read every card's summary in all three languages. A native speaker rates clarity, and checks numbers and dates ("5.3 मी.", "4 जून").
- **End-to-end.** Voice question → guide → highlight → spoken reply, on a phone over mobile data.
