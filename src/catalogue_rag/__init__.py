"""Catalogue RAG: cited answers from product catalogues.

Hybrid retrieval (BM25 + embeddings, fused with reciprocal rank fusion) over
page-aware, table-preserving chunks, and an LLM that must cite a source for every
claim or say the catalogues do not cover the question.
"""

__version__ = "1.0.0"
