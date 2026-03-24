"""Phase 2 — ChromaDB storage layer for bookmark embeddings."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
import numpy as np

from src.models import Bookmark

logger = logging.getLogger(__name__)


class ChromaStore:
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "bookmarks",
    ) -> None:
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready — collection '%s' has %d items",
            collection_name,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def upsert_bookmarks(self, bookmarks: list[Bookmark], batch_size: int = 500) -> None:
        """Store or update bookmarks that have embeddings."""
        embedded = [b for b in bookmarks if b.embedding is not None]
        if not embedded:
            logger.warning("upsert_bookmarks: no bookmarks with embeddings to store.")
            return

        for start in range(0, len(embedded), batch_size):
            chunk = embedded[start : start + batch_size]
            self._collection.upsert(
                ids=[b.id for b in chunk],
                embeddings=[b.embedding for b in chunk],  # type: ignore[list-item]
                documents=[b.document for b in chunk],
                metadatas=[
                    {
                        "title": b.title,
                        "url": b.url,
                        "domain": b.domain,
                        "original_path": b.original_path,
                        "scraped": str(b.scraped),
                    }
                    for b in chunk
                ],
            )
        logger.info("Upserted %d bookmarks into ChromaDB", len(embedded))

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query_similar(
        self,
        embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return up to n_results items similar to the given embedding.
        Each result dict has: id, document, metadata, distance.
        """
        kwargs: dict = {"query_embeddings": [embedding], "n_results": min(n_results, self._collection.count())}
        if where:
            kwargs["where"] = where
        results = self._collection.query(include=["documents", "metadatas", "distances"], **kwargs)

        out = []
        for i, rid in enumerate(results["ids"][0]):
            out.append(
                {
                    "id": rid,
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return out

    def get_all_embeddings(self) -> dict[str, list[float]]:
        """Return {id: embedding} for every stored bookmark."""
        total = self._collection.count()
        if total == 0:
            return {}

        result = self._collection.get(include=["embeddings"])
        return {rid: emb for rid, emb in zip(result["ids"], result["embeddings"])}

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """Fetch specific documents by their IDs."""
        result = self._collection.get(ids=ids, include=["documents", "metadatas", "embeddings"])
        out = []
        for i, rid in enumerate(result["ids"]):
            out.append(
                {
                    "id": rid,
                    "document": result["documents"][i],
                    "metadata": result["metadatas"][i],
                    "embedding": result["embeddings"][i],
                }
            )
        return out

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        """Drop and recreate the collection (full reset)."""
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' reset.", name)
