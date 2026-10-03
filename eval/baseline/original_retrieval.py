"""
Hybrid retrieval engine combining vector search and BM25 keyword search.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
from collections import Counter
import re

import chromadb
from llama_index.core import Document


class HybridRetrievalEngine:
    """Combines vector search and BM25 keyword matching for precise retrieval."""

    def __init__(self, chroma_db_path: str = "./chroma_data", collection_name: str = "catalogue_documents"):
        """
        Initialize the retrieval engine.

        Args:
            chroma_db_path: Path to ChromaDB storage
            collection_name: Collection name in ChromaDB
        """
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.parsed_docs_path = Path("./data/parsed_documents")

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict,
    ) -> None:
        """
        Add a parsed document to the vector store.

        Args:
            doc_id: Unique document identifier
            content: Full document content (text)
            metadata: Document metadata (type, category, brand, etc.)
        """
        # Extract part numbers and technical terms for better indexing
        part_numbers = self._extract_part_numbers(content)
        technical_terms = self._extract_technical_terms(content)

        # Combine metadata with extracted terms, ensure all values are strings
        enhanced_metadata = {
            "filename": str(metadata.get("filename", "")),
            "document_type": str(metadata.get("document_type", "")),
            "category": str(metadata.get("category", "")),
            "brand": str(metadata.get("brand", "") or ""),
            "region": str(metadata.get("region", "NZ")),
            "part_numbers": ",".join(part_numbers),
            "technical_terms": ",".join(technical_terms),
        }

        # Clean content for embedding (remove excessive whitespace)
        clean_content = self._clean_text(content)

        # Add to ChromaDB
        self.collection.add(
            ids=[doc_id],
            documents=[clean_content],
            metadatas=[enhanced_metadata],
        )

        print(f"[Added] {doc_id} with {len(part_numbers)} part numbers")

    def vector_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search using vector embeddings (semantic search).

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, relevance_score, metadata) tuples
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["embeddings", "metadatas", "distances"]
            )

            if not results["ids"] or not results["ids"][0]:
                return []

            # Convert distances to relevance scores (1 - distance for cosine)
            retrieved = []
            for doc_id, distance, metadata in zip(
                results["ids"][0],
                results["distances"][0],
                results["metadatas"][0]
            ):
                relevance_score = 1 - distance  # Convert distance to similarity
                retrieved.append((doc_id, relevance_score, metadata))

            return retrieved
        except Exception as e:
            print(f"[Error] Vector search failed: {e}")
            return []

    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Search using BM25 (keyword-based retrieval).
        Optimized for exact part numbers and technical specifications.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, bm25_score, metadata) tuples
        """
        # Get all documents in collection
        all_docs = self.collection.get(include=["metadatas"])

        if not all_docs["ids"]:
            return []

        # Tokenize query
        query_tokens = self._tokenize(query.lower())

        # Score each document
        scores = {}
        for doc_id, metadata in zip(all_docs["ids"], all_docs["metadatas"]):
            # Check for exact part number matches (highest priority)
            part_numbers = metadata.get("part_numbers", "").split(",")
            exact_match = any(pn.strip() in query for pn in part_numbers if pn.strip())

            # Get document content for BM25
            doc_result = self.collection.get(ids=[doc_id], include=["documents"])
            if doc_result["documents"]:
                doc_tokens = self._tokenize(doc_result["documents"][0])

                # Calculate BM25 score
                score = self._calculate_bm25_score(query_tokens, doc_tokens)

                # Boost score if part number matches
                if exact_match:
                    score *= 2.0

                scores[doc_id] = score

        # Sort by score and return top_k
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        retrieved = []
        for doc_id, score in sorted_results:
            metadata = self.collection.get(ids=[doc_id], include=["metadatas"])["metadatas"][0]
            retrieved.append((doc_id, score, metadata))

        return retrieved

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        force_keyword: bool = False,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Perform hybrid search combining vector and BM25 results.

        Args:
            query: Search query
            top_k: Number of results to return
            vector_weight: Weight for vector search scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)
            force_keyword: If True, prioritize BM25 for part numbers

        Returns:
            List of (doc_id, combined_score, metadata) tuples
        """
        # Check if query looks like a part number (alphanumeric, dashes, underscores)
        is_part_number = bool(re.match(r'^[A-Z0-9\-_]{3,}$', query.strip().upper()))

        if is_part_number or force_keyword:
            # For part numbers, use BM25-first hybrid approach
            bm25_results = self.bm25_search(query, top_k * 2)
            vector_results = self.vector_search(query, top_k)

            # Combine results
            combined = self._merge_results(
                bm25_results, vector_results,
                bm25_weight=0.7, vector_weight=0.3
            )
        else:
            # For conceptual queries, use vector-first hybrid approach
            vector_results = self.vector_search(query, top_k * 2)
            bm25_results = self.bm25_search(query, top_k)

            combined = self._merge_results(
                vector_results, bm25_results,
                weight_a=vector_weight, weight_b=bm25_weight
            )

        return combined[:top_k]

    def _merge_results(
        self,
        results_a: List[Tuple[str, float, Dict]],
        results_b: List[Tuple[str, float, Dict]],
        weight_a: float = 0.6,
        weight_b: float = 0.4,
    ) -> List[Tuple[str, float, Dict]]:
        """Merge two ranked result sets."""
        scores = {}

        # Add results from first set
        max_score_a = max([r[1] for r in results_a], default=1.0)
        for doc_id, score, metadata in results_a:
            normalized_score = (score / max_score_a) if max_score_a > 0 else 0
            scores[doc_id] = (normalized_score * weight_a, metadata)

        # Add/merge results from second set
        max_score_b = max([r[1] for r in results_b], default=1.0)
        for doc_id, score, metadata in results_b:
            normalized_score = (score / max_score_b) if max_score_b > 0 else 0
            if doc_id in scores:
                # Combine scores
                combined_score = scores[doc_id][0] + (normalized_score * weight_b)
                scores[doc_id] = (combined_score, metadata)
            else:
                scores[doc_id] = (normalized_score * weight_b, metadata)

        # Sort by combined score
        sorted_results = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
        return [(doc_id, score, metadata) for doc_id, (score, metadata) in sorted_results]

    # ===== Utility Methods =====

    def _extract_part_numbers(self, text: str) -> List[str]:
        """Extract potential part numbers from text."""
        # Pattern for part numbers: 3-10 alphanumeric chars, may include dashes/underscores
        pattern = r'\b[A-Z0-9]{3,}[A-Z0-9\-_]*\b'
        matches = re.findall(pattern, text.upper())
        # Filter common words
        common_words = {"THE", "AND", "FOR", "WITH", "FROM", "TYPE", "MODEL"}
        return [m for m in matches if m not in common_words][:50]  # Limit to 50

    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms relevant to hardware."""
        technical_keywords = {
            "fail-safe", "fail-secure", "electric strike", "strike", "lockset",
            "voltage", "current", "draw", "IP rating", "outdoor", "indoor",
            "exit door", "egress", "accessible", "compliance", "standard",
            "specification", "dimension", "backset", "latch", "deadbolt"
        }

        text_lower = text.lower()
        found_terms = [term for term in technical_keywords if term in text_lower]
        return found_terms

    def _clean_text(self, text: str) -> str:
        """Clean text for better indexing."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters except for important ones
        text = re.sub(r'[^\w\s\-\./\(\)]', '', text)
        return text.strip()

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return re.findall(r'\b\w+\b', text.lower())

    def _calculate_bm25_score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        Simple BM25-like scoring.
        In production, use rank_bm25 library for proper implementation.
        """
        k1, b = 1.5, 0.75
        avg_doc_length = len(doc_tokens)
        doc_length = len(doc_tokens)

        score = 0
        for token in query_tokens:
            # Count token occurrences
            count = doc_tokens.count(token)
            if count > 0:
                # BM25 formula (simplified)
                idf = 1  # In production, calculate actual IDF
                score += idf * (count * (k1 + 1)) / (count + k1 * (1 - b + b * (doc_length / avg_doc_length)))

        return score

    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        all_docs = self.collection.get(include=["metadatas"])
        return {
            "total_documents": len(all_docs["ids"]),
            "document_types": list(set(m.get("document_type", "Unknown") for m in all_docs["metadatas"])),
            "categories": list(set(m.get("category", "Unknown") for m in all_docs["metadatas"])),
            "brands": list(set(m.get("brand", "Unknown") for m in all_docs["metadatas"] if m.get("brand"))),
        }
