"""Main orchestrator — runs all 11 phases in sequence."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import AppConfig
from src.models import Bookmark, TaxonomyTree
from src.ingestion.parser import parse_bookmarks_html
from src.ingestion.enricher import enrich_all
from src.ingestion.scraper import scrape_all
from src.embeddings.embedder import Embedder
from src.storage.chroma_store import ChromaStore
from src.clustering.reducer import UMAPReducer
from src.clustering.hdbscan_clusterer import run_hdbscan
from src.clustering.kmeans_clusterer import run_kmeans
from src.labeling.llm_labeler import LLMLabeler
from src.taxonomy.root_assigner import RootAssigner
from src.taxonomy.hierarchy_builder import HierarchyBuilder
from src.taxonomy.category_validator import CategoryValidator
from src.taxonomy.deduplicator import (
    remove_exact_duplicates,
    find_semantic_duplicates,
    get_active_bookmarks,
)
from src.output.html_exporter import HTMLExporter
from src.output.json_exporter import JSONExporter
from src.output.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config: AppConfig) -> None:
        self.cfg = config
        self._embedder: Optional[Embedder] = None
        self._store: Optional[ChromaStore] = None
        self._labeler: Optional[LLMLabeler] = None

    # ------------------------------------------------------------------
    # Dependency singletons
    # ------------------------------------------------------------------

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder(
                model=self.cfg.ollama.embedding_model,
                base_url=self.cfg.ollama.base_url,
                batch_size=self.cfg.processing.batch_size,
                cache_dir=self.cfg.cache.directory,
                cache_enabled=self.cfg.cache.enabled,
            )
        return self._embedder

    @property
    def store(self) -> ChromaStore:
        if self._store is None:
            self._store = ChromaStore(
                persist_directory=self.cfg.chromadb.persist_directory,
                collection_name=self.cfg.chromadb.collection_name,
            )
        return self._store

    @property
    def labeler(self) -> LLMLabeler:
        if self._labeler is None:
            self._labeler = LLMLabeler(
                model=self.cfg.ollama.llm_model,
                base_url=self.cfg.ollama.base_url,
            )
        return self._labeler

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    def preflight(self) -> None:
        logger.info("Running preflight checks…")
        self.embedder.check_connection()
        self.labeler.check_connection()
        logger.info("Preflight OK")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, bookmarks_file: str | Path) -> TaxonomyTree:
        cfg = self.cfg

        # ── Phase 1: Ingestion ──────────────────────────────────────────
        logger.info("Phase 1: Parsing %s", bookmarks_file)
        bookmarks = parse_bookmarks_html(bookmarks_file)
        bookmarks = enrich_all(bookmarks)

        if cfg.processing.scrape_page:
            logger.info("Phase 1b: Scraping pages (scrape_page=true)…")
            bookmarks = scrape_all(
                bookmarks,
                timeout=cfg.processing.scrape_timeout,
                max_chars=cfg.processing.scrape_max_chars,
            )

        # ── Phase 9a: Exact deduplication (early pass) ─────────────────
        bookmarks, exact_removed = remove_exact_duplicates(bookmarks)
        logger.info("Removed %d exact duplicates", exact_removed)

        if not bookmarks:
            logger.warning("No bookmarks after deduplication. Exiting.")
            return TaxonomyTree()

        # ── Phase 2: Embed + store ──────────────────────────────────────
        logger.info("Phase 2: Embedding %d bookmarks…", len(bookmarks))
        bookmarks = self.embedder.embed_bookmarks(bookmarks)
        self.store.upsert_bookmarks(bookmarks)

        # ── Phase 9b: Semantic deduplication ───────────────────────────
        bookmarks, sem_removed = find_semantic_duplicates(
            bookmarks,
            threshold=cfg.validation.duplicate_semantic_threshold,
        )
        logger.info("Marked %d semantic duplicates", sem_removed)
        active = get_active_bookmarks(bookmarks)

        if len(active) < cfg.validation.min_cluster_size:
            logger.warning(
                "Only %d active bookmarks — too few to cluster. Assigning all to MISC.",
                len(active),
            )
            for bm in active:
                bm.category_path = ["MISC"]
            tree = self._build_minimal_tree(bookmarks, cfg)
            self._write_outputs(tree)
            return tree

        # ── Phase 3: Root assignment ────────────────────────────────────
        logger.info("Phase 3: Assigning bookmarks to root buckets…")
        assigner = RootAssigner(
            root_buckets=cfg.root_buckets,
            embedder=self.embedder,
            threshold=cfg.validation.root_assignment_threshold,
        )
        pre_assigned, unsorted = assigner.assign(active)

        # ── Phase 4: Cluster unsorted bookmarks ─────────────────────────
        topic_clusters = []
        if unsorted:
            logger.info("Phase 4: Clustering %d unsorted bookmarks…", len(unsorted))
            embeddings = np.array([b.embedding for b in unsorted], dtype=np.float32)
            ids = [b.id for b in unsorted]

            reducer = UMAPReducer(
                n_components=cfg.clustering.umap_n_components,
                n_neighbors=cfg.clustering.umap_n_neighbors,
                min_dist=cfg.clustering.umap_min_dist,
            )
            reduced = reducer.fit_transform(embeddings)

            if cfg.clustering.algorithm == "hdbscan":
                topic_clusters = run_hdbscan(
                    embeddings_reduced=reduced,
                    embeddings_full=embeddings,
                    ids=ids,
                    min_cluster_size=cfg.clustering.hdbscan_min_cluster_size,
                    min_samples=cfg.clustering.hdbscan_min_samples,
                )
                # Fallback: if HDBSCAN puts everything in 1 cluster, use K-Means
                real_clusters = [c for c in topic_clusters if c.cluster_id >= 0]
                if len(real_clusters) <= 1:
                    logger.warning("HDBSCAN produced ≤1 cluster — falling back to K-Means")
                    topic_clusters = run_kmeans(
                        embeddings_full=embeddings,
                        ids=ids,
                        k_factor=cfg.clustering.kmeans_k_factor,
                        min_cluster_size=cfg.clustering.hdbscan_min_cluster_size,
                    )
            else:
                topic_clusters = run_kmeans(
                    embeddings_full=embeddings,
                    ids=ids,
                    k_factor=cfg.clustering.kmeans_k_factor,
                    min_cluster_size=cfg.clustering.hdbscan_min_cluster_size,
                )

        # ── Phases 5–8: Hierarchy + labeling ───────────────────────────
        logger.info("Phases 5–8: Building hierarchy…")
        validator = CategoryValidator(
            min_cluster_size=cfg.validation.min_cluster_size,
            min_confidence=cfg.validation.min_confidence,
            merge_threshold=cfg.validation.merge_threshold,
            max_categories=cfg.validation.max_categories,
        )

        bookmark_map = {b.id: b for b in active}
        builder = HierarchyBuilder(
            labeler=self.labeler,
            validator=validator,
            root_buckets=cfg.root_buckets,
            root_embeddings=assigner.root_embeddings(),
            root_assignment_threshold=cfg.validation.root_assignment_threshold,
        )
        tree = builder.build(
            topic_clusters=topic_clusters,
            bookmark_map=bookmark_map,
            pre_assigned=pre_assigned,
        )

        # Add duplicate bookmarks to tree (marked, not exported in main output)
        for bm in bookmarks:
            if bm.is_duplicate and bm not in tree.bookmarks:
                tree.bookmarks.append(bm)

        logger.info(
            "Tree built: %d categories, %d active bookmarks",
            len(tree.categories),
            sum(1 for b in tree.bookmarks if not b.is_duplicate),
        )

        # ── Phase 10: Output ────────────────────────────────────────────
        self._write_outputs(tree)

        return tree

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_minimal_tree(self, bookmarks: list[Bookmark], cfg: AppConfig) -> TaxonomyTree:
        """Build a minimal tree when there are too few bookmarks to cluster."""
        from src.taxonomy.category_validator import _category_id

        tree = TaxonomyTree()
        for name in cfg.root_buckets:
            from src.models import Category
            cat = Category(
                id=_category_id(name, 1),
                name=name,
                level=1,
                parent_id=None,
                centroid=[],
                member_count=0,
                confidence=1.0,
                is_preset=True,
            )
            tree.categories[cat.id] = cat
        tree.bookmarks = bookmarks
        return tree

    def _write_outputs(self, tree: TaxonomyTree) -> None:
        cfg = self.cfg
        out_dir = Path(cfg.output.directory)
        formats = cfg.output.formats

        if "html" in formats:
            HTMLExporter(tree).export(out_dir / "bookmarks_organized.html")
        if "json" in formats:
            exp = JSONExporter(tree)
            exp.export(out_dir / "taxonomy.json")
            exp.export_graph(out_dir / "graph.json")
        if "report" in formats:
            ReportGenerator(tree).generate(out_dir / "report.txt")
