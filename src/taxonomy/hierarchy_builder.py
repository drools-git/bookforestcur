"""Phase 6 — Build the multi-level taxonomy bottom-up from raw clusters."""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from src.models import Bookmark, Category, ClusterResult, LabelResult, TaxonomyTree
from src.labeling.llm_labeler import LLMLabeler
from src.taxonomy.category_validator import CategoryValidator

logger = logging.getLogger(__name__)

MISC_NAME = "MISC"


def _category_id(name: str, level: int) -> str:
    return hashlib.sha256(f"L{level}::{name.lower()}".encode()).hexdigest()[:12]


def _centroid_of(embeddings_list: list[list[float]]) -> list[float]:
    if not embeddings_list:
        return []
    return np.array(embeddings_list, dtype=np.float32).mean(axis=0).tolist()


def _agglomerative_cluster(
    centroids: list[list[float]],
    n_clusters: Optional[int] = None,
    distance_threshold: float = 0.4,
) -> list[int]:
    """
    Cluster a small set of centroids (topic/subdomain level) using agglomerative clustering.
    If n_clusters is None, use distance_threshold for automatic k selection.
    """
    if len(centroids) < 2:
        return [0] * len(centroids)

    matrix = np.array(centroids, dtype=np.float32)

    if n_clusters is not None:
        k = min(n_clusters, len(centroids))
        agg = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    else:
        agg = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
    return agg.fit_predict(matrix).tolist()


class HierarchyBuilder:
    """
    Orchestrates Phases 6.1 → 6.3: Topics → Subdomains → Domains → Roots.
    """

    def __init__(
        self,
        labeler: LLMLabeler,
        validator: CategoryValidator,
        root_buckets: list[str],
        root_embeddings: dict[str, list[float]],
        root_assignment_threshold: float = 0.75,
    ) -> None:
        self.labeler = labeler
        self.validator = validator
        self.root_buckets = root_buckets
        self.root_embeddings = root_embeddings
        self.root_threshold = root_assignment_threshold

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def build(
        self,
        topic_clusters: list[ClusterResult],
        bookmark_map: dict[str, Bookmark],
        pre_assigned: list[Bookmark],
    ) -> TaxonomyTree:
        """
        Build the full TaxonomyTree.

        Args:
            topic_clusters:  ClusterResult list from Phase 4 (HDBSCAN/K-Means)
            bookmark_map:    id → Bookmark for all active bookmarks
            pre_assigned:    Bookmarks already assigned in Phase 3 (L1 roots)
        """
        tree = TaxonomyTree()

        # Create the preset L1 roots first
        for root_name in self.root_buckets:
            cat = Category(
                id=_category_id(root_name, 1),
                name=root_name,
                level=1,
                parent_id=None,
                centroid=self.root_embeddings.get(root_name, []),
                member_count=0,
                confidence=1.0,
                is_preset=True,
            )
            tree.categories[cat.id] = cat

        # Attach pre-assigned bookmarks to their root
        for bm in pre_assigned:
            if bm.category_path:
                root_name = bm.category_path[0]
                root_id = _category_id(root_name, 1)
                if root_id in tree.categories:
                    tree.categories[root_id].member_count += 1
            tree.bookmarks.append(bm)

        # --- Step 6.1: Label topics (L4) --------------------------------
        noise_ids: list[str] = []
        topic_categories: list[Category] = []

        for cluster in topic_clusters:
            if cluster.cluster_id == -1:
                noise_ids.extend(cluster.bookmark_ids)
                continue

            label = self.labeler.label_topic(cluster, bookmark_map)

            if not self.validator.can_create(
                len(cluster.bookmark_ids), label.confidence, label.category_name
            ):
                noise_ids.extend(cluster.bookmark_ids)
                continue

            # Check for merge with existing topic
            existing_topics = [c for c in topic_categories]
            merge_target = self.validator.find_merge_target(
                cluster.centroid, label.category_name, existing_topics
            )
            if merge_target:
                source_bms = [bookmark_map[bid] for bid in cluster.bookmark_ids if bid in bookmark_map]
                self.validator.merge_into(merge_target, source_bms, cluster.centroid)
                for bid in cluster.bookmark_ids:
                    if bid in bookmark_map:
                        bookmark_map[bid].category_path = [
                            *bookmark_map[bid].category_path[:3],
                            merge_target.name,
                        ]
                continue

            topic_cat = self.validator.make_category(
                name=label.category_name,
                level=4,
                centroid=cluster.centroid,
                member_count=len(cluster.bookmark_ids),
                confidence=label.confidence,
            )
            topic_categories.append(topic_cat)
            tree.categories[topic_cat.id] = topic_cat

            for bid in cluster.bookmark_ids:
                if bid in bookmark_map:
                    bookmark_map[bid].category_path = [topic_cat.name]

        # --- Step 6.2: Group Topics → Subdomains (L3) -------------------
        subdomain_categories = self._build_level(
            child_categories=topic_categories,
            level=3,
            tree=tree,
            label_fn=lambda names: self.labeler.label_subdomain(names),
        )

        # --- Step 6.3: Group Subdomains → Domains (L2) ------------------
        domain_categories = self._build_level(
            child_categories=subdomain_categories,
            level=2,
            tree=tree,
            label_fn=lambda names: self.labeler.label_domain(names),
        )

        # --- Step 6.4: Map Domains → Roots (L1) -------------------------
        self._map_to_roots(domain_categories, tree)

        # --- Propagate full paths ----------------------------------------
        self._propagate_paths(tree, bookmark_map)

        # Noise → MISC
        misc_id = _category_id(MISC_NAME, 1)
        for bid in noise_ids:
            if bid in bookmark_map:
                bm = bookmark_map[bid]
                bm.category_path = [MISC_NAME]
                tree.bookmarks.append(bm)
        if misc_id in tree.categories:
            tree.categories[misc_id].member_count += len(noise_ids)

        # Add clustered bookmarks
        for bm in bookmark_map.values():
            if bm not in tree.bookmarks:
                tree.bookmarks.append(bm)

        return tree

    # ------------------------------------------------------------------
    # Level builder helper
    # ------------------------------------------------------------------

    def _build_level(
        self,
        child_categories: list[Category],
        level: int,
        tree: TaxonomyTree,
        label_fn,
    ) -> list[Category]:
        """Cluster child centroids and label each meta-cluster at `level`."""
        if len(child_categories) < 2:
            # Not enough children to cluster — skip this level
            return child_categories

        centroids = [c.centroid for c in child_categories if c.centroid]
        if len(centroids) < 2:
            return child_categories

        labels = _agglomerative_cluster(centroids, distance_threshold=0.4)

        parent_categories: list[Category] = []
        unique_labels = sorted(set(labels))

        for lbl in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == lbl]
            children = [child_categories[i] for i in indices]
            child_names = [c.name for c in children]
            total_members = sum(c.member_count for c in children)
            centroid = _centroid_of([c.centroid for c in children if c.centroid])

            label_result: LabelResult = label_fn(child_names)

            if not self.validator.can_create(total_members, label_result.confidence, label_result.category_name):
                continue

            existing_at_level = [c for c in parent_categories]
            merge_target = self.validator.find_merge_target(centroid, label_result.category_name, existing_at_level)
            if merge_target:
                merge_target.member_count += total_members
                for child in children:
                    if child.id not in merge_target.children_ids:
                        merge_target.children_ids.append(child.id)
                    child.parent_id = merge_target.id
                continue

            parent_cat = self.validator.make_category(
                name=label_result.category_name,
                level=level,
                centroid=centroid,
                member_count=total_members,
                confidence=label_result.confidence,
            )
            parent_cat.children_ids = [c.id for c in children]
            parent_categories.append(parent_cat)
            tree.categories[parent_cat.id] = parent_cat

            for child in children:
                child.parent_id = parent_cat.id

        return parent_categories

    # ------------------------------------------------------------------
    # Map L2 domains → L1 roots
    # ------------------------------------------------------------------

    def _map_to_roots(
        self,
        domain_categories: list[Category],
        tree: TaxonomyTree,
    ) -> None:
        if not domain_categories:
            return

        root_names = self.root_buckets
        root_matrix = np.array(
            [self.root_embeddings.get(r, [0.0]) for r in root_names],
            dtype=np.float32,
        )

        for domain in domain_categories:
            if not domain.centroid:
                best_root = MISC_NAME
            else:
                vec = np.array(domain.centroid, dtype=np.float32)
                norms = np.linalg.norm(root_matrix, axis=1) * np.linalg.norm(vec)
                with np.errstate(invalid="ignore"):
                    sims = np.where(norms > 0, root_matrix @ vec / norms, 0.0)
                best_idx = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                best_root = root_names[best_idx] if best_score >= self.root_threshold else MISC_NAME

            root_id = _category_id(best_root, 1)
            if root_id in tree.categories:
                root_cat = tree.categories[root_id]
                if domain.id not in root_cat.children_ids:
                    root_cat.children_ids.append(domain.id)
                root_cat.member_count += domain.member_count
                domain.parent_id = root_id

    # ------------------------------------------------------------------
    # Path propagation
    # ------------------------------------------------------------------

    def _propagate_paths(
        self,
        tree: TaxonomyTree,
        bookmark_map: dict[str, Bookmark],
    ) -> None:
        """Walk the tree and set full category_path on each bookmark."""

        def get_path(cat_id: str) -> list[str]:
            path = []
            current = tree.categories.get(cat_id)
            while current:
                path.insert(0, current.name)
                current = tree.categories.get(current.parent_id) if current.parent_id else None
            return path

        # Build reverse map: topic_name → category_id
        topic_name_to_id: dict[str, str] = {
            c.name: c.id for c in tree.categories.values() if c.level == 4
        }

        for bm in bookmark_map.values():
            leaf = bm.leaf_category()
            if leaf and leaf in topic_name_to_id:
                bm.category_path = get_path(topic_name_to_id[leaf])
