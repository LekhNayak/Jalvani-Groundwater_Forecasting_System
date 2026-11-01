/** Split a card's text into short pieces for text-to-speech. Speech takes time in proportion to its length (about 5 s
 *  for a whole forecast card), but a short first piece starts playing in about a second while the rest is made.
 *  Splits at sentence ends first, then at commas; decimals like "7.1" are never split. */
export function splitForSpeech(text: string, first = 55, rest = 85): string[] {
  const sentences = text.split(/(?<=[.।!?])\s+/).map((s) => s.trim()).filter(Boolean);
  const out: string[] = [];
  for (const s of sentences) {
    let cur = "";
    for (const piece of s.split(/(?<=[,;:،])\s+/)) {
      const max = out.length === 0 ? first : rest;
      if (cur && cur.length + piece.length + 1 > max) { out.push(cur); cur = piece; }
      else cur = cur ? `${cur} ${piece}` : piece;
    }
    if (cur) out.push(cur);
  }
  return out.length ? out : [text];
}
