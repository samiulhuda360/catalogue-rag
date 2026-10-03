"""Split parsed catalogues into page-aware chunks that keep tables readable.

Catalogues are mostly specification tables. The parser (LlamaParse) gives Markdown
with HTML tables and one `---` line between pages. A chunk here:

* never spans two pages, so every answer can cite a page number;
* carries a breadcrumb (document title > section headings) so a table row such as
  "Battery | 1 x Lithium CR123A" still says which product it belongs to;
* keeps tables as `cell | cell` rows, and when a table is too big for one chunk it is
  split by rows with the header row repeated in every piece.
"""

from __future__ import annotations

import html
import re
from dataclasses import asdict, dataclass

IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HEADING = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)
ROW = re.compile(r"<tr\b.*?>(.*?)</tr>", re.S | re.I)
CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")


@dataclass
class Chunk:
    id: str
    doc: str        # document id (file stem)
    title: str      # human title of the document
    page: int       # 1-based page number
    section: str    # heading breadcrumb within the document
    text: str       # body text shown to the model and to the user

    def search_text(self) -> str:
        """What gets embedded and keyword-indexed: breadcrumb plus body."""
        return f"{self.title} > {self.section}\n{self.text}" if self.section else f"{self.title}\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def clean_inline(text: str) -> str:
    text = re.sub(r"\[(?:colspan|rowspan)=\d+\]", "", text, flags=re.I)  # parser artefacts
    text = re.sub(r"<br\s*/?>", "; ", text, flags=re.I)
    text = TAG.sub("", text)
    text = html.unescape(text)
    text = text.replace("**", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def table_to_rows(table_html: str) -> list[str]:
    rows = []
    for tr in ROW.findall(table_html):
        cells = [clean_inline(c) for c in CELL.findall(tr)]
        cells = [c for c in cells if c]
        if cells:
            rows.append(" | ".join(cells))
    return rows


def split_header(markdown: str) -> tuple[dict, str]:
    """The ingestion step writes a small metadata header, then `---`, then the document."""
    meta: dict = {}
    if "\n---\n" not in markdown[:2000]:
        return meta, markdown
    head, body = markdown.split("\n---\n", 1)
    for key, value in re.findall(r"\*\*(.+?):\*\*\s*(.+)", head):
        meta[key.strip().lower().replace(" ", "_")] = value.strip()
    return meta, body


def blocks_of(page: str) -> list[tuple[str, str]]:
    """A page as a list of (kind, text): headings, tables (as rows) and paragraphs."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in TABLE.finditer(page):
        out += _text_blocks(page[pos:m.start()])
        rows = table_to_rows(m.group(0))
        if rows:
            out.append(("table", "\n".join(rows)))
        pos = m.end()
    out += _text_blocks(page[pos:])
    return out


def _text_blocks(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            body = clean_inline(" ".join(para)) if not any(p.startswith(("* ", "- ")) for p in para) \
                else "\n".join(clean_inline(p) for p in para)
            if body:
                out.append(("text", body))
            para.clear()

    for line in IMAGE.sub("", text).splitlines():
        line = line.strip()
        h = HEADING.match(line)
        if h:
            flush()
            title = clean_inline(h.group(2))
            if title:
                out.append((f"h{len(h.group(1))}", title))
        elif not line:
            flush()
        else:
            para.append(line)
    flush()
    return out


def chunk_document(doc_id: str, markdown: str, max_chars: int = 1200) -> list[Chunk]:
    meta, body = split_header(markdown)
    pages = re.split(r"\n\s*---\s*\n", body)
    title = _doc_title(doc_id, pages)
    chunks: list[Chunk] = []
    headings: dict[int, str] = {}  # level -> text; carried across pages

    def emit(page_no: int, buf: list[str], section: str) -> None:
        text = "\n".join(buf).strip()
        if text and len(text) > 20:
            chunks.append(Chunk(f"{doc_id}::p{page_no}::c{len(chunks)}", doc_id, title, page_no, section, text))
        buf.clear()

    for page_no, page in enumerate(pages, 1):
        buf: list[str] = []
        buf_section = ""

        for kind, text in blocks_of(page):
            if kind.startswith("h"):
                level = int(kind[1])
                emit(page_no, buf, buf_section)
                headings = {lv: t for lv, t in headings.items() if lv < level}
                headings[level] = text
                continue
            section = " > ".join(headings[lv] for lv in sorted(headings) if headings[lv] != title)[:200]
            if buf and section != buf_section:
                emit(page_no, buf, buf_section)
            buf_section = section
            pieces = _split_table(text, max_chars) if kind == "table" else _split_text(text, max_chars)
            for piece in pieces:
                if buf and sum(len(b) + 1 for b in buf) + len(piece) > max_chars:
                    emit(page_no, buf, buf_section)
                buf.append(piece)
        emit(page_no, buf, buf_section)
    return chunks


def _split_table(rows_text: str, max_chars: int) -> list[str]:
    rows = rows_text.splitlines()
    if len(rows_text) <= max_chars:
        return [rows_text]
    header, out, cur = rows[0], [], [rows[0]]
    for row in rows[1:]:
        if sum(len(r) + 1 for r in cur) + len(row) > max_chars and len(cur) > 1:
            out.append("\n".join(cur))
            cur = [header]
        cur.append(row)
    if len(cur) > 1:
        out.append("\n".join(cur))
    return out


def _split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    out, cur = [], ""
    for s in sentences:
        if cur and len(cur) + len(s) + 1 > max_chars:
            out.append(cur)
            cur = ""
        cur = f"{cur} {s}".strip() if cur else s
        while len(cur) > max_chars:  # one enormous sentence
            out.append(cur[:max_chars])
            cur = cur[max_chars:]
    if cur:
        out.append(cur)
    return out


GENERIC_TITLES = {"product catalogue", "catalogue", "contents"}


def _doc_title(doc_id: str, pages: list[str]) -> str:
    """The first real heading. A brand-only or generic heading ("Acme",
    "Acme Product Catalogue") is joined to, or replaced by, the next one so that
    sibling documents get distinct titles."""
    heads = [t for page in pages[:2] for kind, t in _text_blocks(TABLE.sub("", page))
             if kind in ("h1", "h2") and len(t) > 3 and not t.lower().endswith(".pdf")]
    if not heads:
        return re.sub(r"[-_]+", " ", re.sub(r"^[0-9a-f]{5}-", "", doc_id)).strip().title()
    first = heads[0]
    nxt = heads[1] if len(heads) > 1 else ""
    if first.lower() in GENERIC_TITLES and nxt:
        return f"{first} {nxt}"[:120]
    if first.lower().endswith("product catalogue") and nxt:
        return nxt[:120]
    return first[:120]
