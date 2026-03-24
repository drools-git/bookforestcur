from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2"


class ChromaDBConfig(BaseModel):
    persist_directory: str = "./data/chroma_db"
    collection_name: str = "bookmarks"


class ProcessingConfig(BaseModel):
    scrape_page: bool = False
    scrape_timeout: int = 10
    scrape_max_chars: int = 2000
    batch_size: int = 32


class ClusteringConfig(BaseModel):
    algorithm: Literal["hdbscan", "kmeans"] = "hdbscan"
    umap_n_components: int = 10
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: int = 5
    kmeans_k_factor: int = 50


class ValidationConfig(BaseModel):
    root_assignment_threshold: float = 0.75
    min_cluster_size: int = 10
    min_confidence: float = 0.75
    merge_threshold: float = 0.85
    duplicate_semantic_threshold: float = 0.95
    max_categories: int = 50


class CacheConfig(BaseModel):
    directory: str = "./data/cache"
    enabled: bool = True


class OutputConfig(BaseModel):
    directory: str = "./data/output"
    formats: list[str] = Field(default_factory=lambda: ["html", "json", "report"])


class GuiConfig(BaseModel):
    enabled: bool = False
    host: str = "localhost"
    port: int = 8080


class AppConfig(BaseModel):
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    chromadb: ChromaDBConfig = Field(default_factory=ChromaDBConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    root_buckets: list[str] = Field(
        default_factory=lambda: ["AI", "DEV", "FINANCE", "FOOD", "GPS", "LEARNING", "MISC"]
    )
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    gui: GuiConfig = Field(default_factory=GuiConfig)

    def resolve_paths(self, base: Path) -> None:
        """Make relative paths absolute using the project root."""
        def _abs(p: str) -> str:
            path = Path(p)
            return str(base / path) if not path.is_absolute() else p

        self.chromadb.persist_directory = _abs(self.chromadb.persist_directory)
        self.cache.directory = _abs(self.cache.directory)
        self.output.directory = _abs(self.output.directory)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load AppConfig from a YAML file, falling back to defaults."""
    if config_path is None:
        config_path = os.environ.get("BOOKFOREST_CONFIG", "config.yaml")

    path = Path(config_path)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = AppConfig.model_validate(raw)
    else:
        cfg = AppConfig()

    cfg.resolve_paths(path.parent.resolve())
    return cfg
