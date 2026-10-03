"""PDF -> Markdown with LlamaParse, which keeps specification tables intact.

Each PDF in the documents folder is parsed once; the Markdown goes to the parsed
folder with a small metadata header, ready for `catalogue-rag index`. Already-parsed
files are skipped, so adding a catalogue only parses the new one.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Brand names to recognise in file names, from CATALOGUE_BRANDS="brand1,brand2" (optional; only used as metadata).
BRANDS = tuple(b.strip() for b in os.getenv("CATALOGUE_BRANDS", "").split(",") if b.strip())


def guess_brand(filename: str) -> str:
    low = filename.lower()
    return next((b for b in BRANDS if b.lower() in low), "")


async def _parse_one(client, pdf: Path, out_dir: Path, tier: str) -> dict:
    uploaded = await client.files.create(file=str(pdf), purpose="parse")
    result = await client.parsing.parse(file_id=uploaded.id, tier=tier, version="latest", expand=["markdown_full", "text_full"])
    meta = {"filename": pdf.name, "brand": guess_brand(pdf.name), "parsed_at": datetime.now().isoformat(timespec="seconds"),
            "tier": tier, "file_id": uploaded.id}
    write_parsed(out_dir, pdf.stem, pdf.name, result.markdown_full or result.text_full or "", meta["brand"], meta["parsed_at"])
    return meta


def write_parsed(out_dir: Path, stem: str, filename: str, body: str, brand: str = "", parsed_at: str = "") -> Path:
    """Save extracted text in the layout the chunker expects: a metadata header, `---`, the document."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = [f"# {filename}", "", f"**Brand:** {brand or '-'}", f"**Parsed:** {parsed_at or datetime.now().isoformat(timespec='seconds')}",
              "", "---", ""]
    out = out_dir / f"{stem}_content.md"
    out.write_text("\n".join(header) + body, encoding="utf-8")
    return out


def parse_pdf(pdf: Path, out_dir: Path, api_key: str, tier: str = "agentic") -> Path:
    """Parse one PDF now (used by uploads and `catalogue-rag add`); returns the Markdown file."""
    if not api_key:
        raise SystemExit("PDF extraction needs LLAMA_CLOUD_API_KEY in .env (or upload the catalogue as Markdown/text).")
    from llama_cloud import AsyncLlamaCloud

    async def run() -> dict:
        return await _parse_one(AsyncLlamaCloud(api_key=api_key), pdf, out_dir, tier)

    out_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(run())
    out = out_dir / f"{pdf.stem}_content.md"
    if not out.exists() or out.stat().st_size < 200:
        raise SystemExit(f"Extraction returned no text for {pdf.name} (a scanned PDF without a text layer?).")
    return out


def add_text(src: Path, out_dir: Path) -> Path:
    """A Markdown or plain-text catalogue goes in as it is."""
    return write_parsed(out_dir, src.stem, src.name, src.read_text(encoding="utf-8", errors="replace"), guess_brand(src.name))


def parse_pdfs(documents_dir: Path, parsed_dir: Path, api_key: str, tier: str = "agentic", force: bool = False, log=print) -> int:
    if not api_key:
        raise SystemExit("Set LLAMA_CLOUD_API_KEY in .env to parse PDFs (https://cloud.llamaindex.ai).")
    from llama_cloud import AsyncLlamaCloud

    parsed_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(documents_dir.glob("*.pdf")) if force or not (parsed_dir / f"{p.stem}_content.md").exists()]
    if not todo:
        log("Nothing new to parse.")
        return 0
    meta_file = parsed_dir / "metadata.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    async def run() -> None:
        client = AsyncLlamaCloud(api_key=api_key)
        for pdf in todo:
            log(f"  parsing {pdf.name} ...")
            try:
                meta[pdf.stem] = await _parse_one(client, pdf, parsed_dir, tier)
            except Exception as exc:  # noqa: BLE001  keep going; report at the end
                log(f"  FAILED {pdf.name}: {exc}")
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    asyncio.run(run())
    return len(todo)
