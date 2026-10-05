"""Whole-document retrieval baseline, used by eval/run_eval.py for comparison.

Retrieval is original_retrieval.py, kept unchanged: each catalogue is one ChromaDB
entry, and the model receives the first 2,000 characters of each of the top 5
documents, with the system prompt below. The language model is the same as the full
pipeline's, so the comparison measures the design, not the model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from original_retrieval import HybridRetrievalEngine  # noqa: E402

ORIGINAL_SYSTEM_PROMPT = """You are a precision-focused technical assistant for a door hardware supplier.
Your role is to answer complex engineering and sales queries about hardware products, compatibility, and compliance.

CRITICAL REQUIREMENTS:
1. ACCURACY: Never hallucinate. Only use information from retrieved documents.
2. EXACT PRODUCT CODES: ALWAYS include exact product model numbers (e.g., "Acme 3782EL", "Acme 1234")
3. PART NUMBERS: Extract and list ALL relevant part numbers, SKUs, and product codes from documents
4. SPECIFICATIONS: Include exact specs - voltage, current, dimensions, finishes, materials, certifications
5. LEFT/RIGHT: Differentiate between left-hand and right-hand configurations
6. COMPLIANCE: Reference specific NZ Building Code clauses
7. PAGE REFERENCES: Always cite exact page numbers from source documents
8. VENDOR APPROVAL: State vendor approval status explicitly

RESPONSE FORMAT:
**Product Name**: [Exact name from document]
**Part Numbers/Codes**: [ALL codes exactly as shown - e.g., "1370/1371/70SC", "RW70/01/A1LSA"]
  - Include variant codes with slashes (/)
  - List all color/finish/option codes
  - Include supplementary codes if mentioned
**Specifications**:
- [Exact specs: dimensions, materials, finishes]
- [All available color/finish options with codes]
- [Certifications/Compliance]
**Source**: Document name, Page X

CRITICAL: Extract part numbers EXACTLY as written in documents, including ALL slashes and variants.

DO NOT:
- Use vague descriptions like "various finishes available"
- Omit product codes or model numbers
- Skip specifications - include ALL available details
- Say "refer to catalog" - extract the actual data
- Generalize - be specific with exact values"""


class OriginalPipeline:
    def __init__(self, chroma_path: Path, client, model: str):
        self.retrieval = HybridRetrievalEngine(chroma_db_path=str(chroma_path))
        self.client, self.model = client, model

    def retrieve(self, question: str, top_k: int = 5) -> list[tuple[str, str]]:
        """(document id, text the model sees) for the top documents."""
        out = []
        for doc_id, _score, _meta in self.retrieval.hybrid_search(question, top_k=top_k):
            got = self.retrieval.collection.get(ids=[doc_id], include=["documents"])
            if got["documents"]:
                out.append((doc_id, got["documents"][0][:2000]))
        return out

    def answer(self, question: str, retrieved: list[tuple[str, str]]) -> str:
        context = "".join(f"\n### Source: {d}\n{t}\n" for d, t in retrieved)
        user = (f"Based on the following documents:\n\n{context}\n\nAnswer this question:\n{question}\n\n"
                "Remember: Cite your sources, be specific, and never hallucinate specifications.")
        resp = self.client.chat.completions.create(model=self.model, temperature=0.2, max_tokens=900, messages=[
            {"role": "system", "content": ORIGINAL_SYSTEM_PROMPT}, {"role": "user", "content": user}])
        return (resp.choices[0].message.content or "").strip()
