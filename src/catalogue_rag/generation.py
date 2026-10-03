"""Answer from retrieved chunks only, with numbered citations, or decline.

The model sees the top chunks as numbered sources and must cite [n] after every
claim. If the sources do not contain the answer it must start its reply with the
exact phrase NOT_FOUND, which the pipeline turns into a clear "not in the catalogues"
answer instead of a guess. Citations are parsed back out, so the UI can show exactly
which page supports each statement, and uncited answers can be flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .retrieval import Hit

NOT_FOUND = "NOT_FOUND"

# Wording of a refusal when the model forgets the NOT_FOUND marker: "The extracts do not mention ...".
# Only used for replies that cite nothing, so a cited answer that mentions a gap is not a decline.
DECLINE_WORDING = re.compile(
    r"\b(?:extracts?|sources?|catalogues?|documents?)\b[^.]{0,80}?\b(?:do(?:es)? not|don't|doesn't|cannot|can't)\s+"
    r"(?:mention|contain|include|cover|specify|state|say|provide|list|confirm)"
    r"|\bnot (?:mentioned|covered|stated|specified|listed|included|provided) in the\b"
    r"|\bno (?:information|mention|details?) (?:about|on|of|regarding)\b", re.I)

SYSTEM_PROMPT = f"""You answer questions about door, window and security hardware using ONLY the numbered
catalogue extracts provided. You are precise because people order parts from your answers.

Rules:
1. Every factual statement ends with the source number(s) in square brackets, e.g. [2] or [1][3].
2. Copy part numbers, dimensions, ratings and units exactly as written in the extract.
3. Prefer a short direct answer: the fact first, then at most a few supporting details.
4. If the extracts answer only part of the question, answer that part and say plainly which part is not
   covered. Only when nothing in the extracts answers the question, begin your reply with the token
   {NOT_FOUND} and follow it with a single short sentence, without citations, naming what the extracts are
   about instead. Example: "{NOT_FOUND} The extracts cover M52 lock finishes and strikes, not vendor
   approvals." Never use general knowledge to fill gaps; never guess prices, approvals, compliance or
   compatibility that the extracts do not state.
5. If the extracts disagree, say so and cite both."""


@dataclass
class Generated:
    text: str
    cited: list[int]      # 1-based source numbers the answer cites
    declined: bool
    model: str
    usage: dict


def format_sources(hits: list[Hit], max_chars: int = 3600) -> str:
    parts = []
    for n, h in enumerate(hits, 1):
        c = h.chunk
        where = f"{c.title} - page {c.page}" + (f" - {c.section}" if c.section else "")
        parts.append(f"[{n}] {where}\n{c.text[:max_chars]}")
    return "\n\n".join(parts)


def parse_citations(text: str, n_sources: int) -> list[int]:
    found = []
    for m in re.finditer(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]", text):
        for num in re.split(r"[,;]\s*", m.group(1)):
            n = int(num)
            if 1 <= n <= n_sources and n not in found:
                found.append(n)
    return found


class Generator:
    def __init__(self, api_key: str, base_url: str, model: str, temperature: float = 0.0, max_tokens: int = 700):
        if not api_key and "openrouter.ai" in base_url:
            raise SystemExit("No LLM key: set OPENROUTER_API_KEY in .env, or LLM_BASE_URL to a self-hosted model server.")
        api_key = api_key or "not-needed"  # self-hosted servers usually ignore the key, but the client requires one
        import openai

        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=60)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens

    def _messages(self, question: str, hits: list[Hit]) -> list[dict]:
        user = f"Catalogue extracts:\n\n{format_sources(hits)}\n\nQuestion: {question}"
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    def answer(self, question: str, hits: list[Hit]) -> Generated:
        if not hits:
            return finish(f"{NOT_FOUND}\nNo catalogue extract matched the question.", 0, self.model)
        resp = self.client.chat.completions.create(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                                                   messages=self._messages(question, hits))
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return finish(resp.choices[0].message.content or "", len(hits), self.model, usage)

    def stream(self, question: str, hits: list[Hit]):
        """Yield ("declined", None) once if the model is declining, then ("delta", text) pieces
        as they arrive, then ("done", Generated) with the cleaned final answer.

        The first few characters are held back until it is clear whether the reply starts
        with NOT_FOUND, so the marker itself is never shown to the user."""
        if not hits:
            yield "done", finish(f"{NOT_FOUND}\nNo catalogue extract matched the question.", 0, self.model)
            return
        resp = self.client.chat.completions.create(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                                                   messages=self._messages(question, hits), stream=True)
        raw, held, decided = "", "", False
        for event in resp:
            piece = (event.choices[0].delta.content or "") if event.choices else ""
            if not piece:
                continue
            raw += piece
            if not decided:
                held += piece
                probe = held.lstrip()
                if len(probe) < len(NOT_FOUND) and NOT_FOUND.startswith(probe.upper()):
                    continue  # could still become NOT_FOUND: keep holding
                decided = True
                if probe.upper().startswith(NOT_FOUND):
                    yield "declined", None
                    piece = probe[len(NOT_FOUND):].lstrip(" :\n")
                else:
                    piece = held
            if piece:
                yield "delta", piece
        yield "done", finish(raw, len(hits), self.model)


def finish(raw: str, n_sources: int, model: str, usage: dict | None = None) -> Generated:
    """Turn the model's raw reply into the final answer: detect a decline, strip markers
    and echoed instructions, and parse the citations."""
    text = re.sub(r"<think>.*?</think>", "", raw.strip(), flags=re.S).strip()  # reasoning models
    declined = text.upper().startswith(NOT_FOUND)
    if declined:
        text = text[len(NOT_FOUND):].strip(" :\n") or "The catalogues do not cover this."
    text = re.sub(rf"\s*\b{NOT_FOUND}\b\s*", " ", text).strip()  # a marker left mid-answer
    # Instruction phrases the model sometimes echoes back verbatim.
    text = re.sub(r"(?im)^\W*(?:on the first line|then one sentence[^.\n]*)[.:,]?\s*$\n?", "", text).strip()
    text = re.sub(r"(?i)^(?:on the first line|then one sentence saying what the extracts do cover)[.:,]?\s*", "", text).strip()
    if declined:
        text = re.sub(r"\s*\[\d+(?:\s*[,;]\s*\d+)*\]", "", text).strip()  # a refusal cites nothing
    cited = [] if declined else parse_citations(text, n_sources)
    if not declined and not cited and DECLINE_WORDING.search(text[:400]):
        declined = True  # an uncited "the extracts do not mention X" is a refusal, marker or not
    return Generated(text, cited, declined, model, usage or {})
