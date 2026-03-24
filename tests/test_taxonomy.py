"""Tests for CategoryValidator and config loading."""
import pytest

from src.taxonomy.category_validator import CategoryValidator
from src.models import Category
from src.config import load_config, AppConfig


def _make_category(name: str, level: int, centroid=None, member_count=10) -> Category:
    from src.taxonomy.category_validator import _category_id
    return Category(
        id=_category_id(name, level),
        name=name,
        level=level,
        centroid=centroid or [0.5] * 8,
        member_count=member_count,
        confidence=0.9,
    )


# ------------------------------------------------------------------
# CategoryValidator
# ------------------------------------------------------------------

class TestCategoryValidator:
    def setup_method(self):
        self.v = CategoryValidator(
            min_cluster_size=10,
            min_confidence=0.75,
            merge_threshold=0.85,
            max_categories=5,
        )

    def test_can_create_valid(self):
        assert self.v.can_create(15, 0.9, "Machine Learning") is True

    def test_rejects_small_cluster(self):
        assert self.v.can_create(5, 0.9, "Tiny") is False

    def test_rejects_low_confidence(self):
        assert self.v.can_create(20, 0.5, "Low Conf") is False

    def test_rejects_banned_name(self):
        # Banned names pass can_create (name check is separate)
        assert self.v.is_valid_name("misc") is False
        assert self.v.is_valid_name("stuff") is False
        assert self.v.is_valid_name("Machine Learning") is True

    def test_cap_reached(self):
        for _ in range(5):
            self.v.register()
        assert self.v.cap_reached is True
        assert self.v.can_create(100, 1.0, "AnyName") is False

    def test_find_merge_target_similar(self):
        existing = [_make_category("Neural Networks", 4, centroid=[1.0] * 8)]
        # Same centroid → similarity = 1.0 ≥ 0.85
        result = self.v.find_merge_target([1.0] * 8, "Deep Learning", existing)
        assert result is not None
        assert result.name == "Neural Networks"

    def test_find_merge_target_dissimilar(self):
        existing = [_make_category("Cooking", 4, centroid=[1.0, 0.0] * 4)]
        result = self.v.find_merge_target([0.0, 1.0] * 4, "Machine Learning", existing)
        assert result is None

    def test_find_merge_target_exact_name(self):
        existing = [_make_category("Machine Learning", 4)]
        result = self.v.find_merge_target([0.0] * 8, "Machine Learning", existing)
        assert result is not None


# ------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------

def test_default_config():
    cfg = AppConfig()
    assert "AI" in cfg.root_buckets
    assert "MISC" in cfg.root_buckets
    assert len(cfg.root_buckets) == 7
    assert cfg.validation.max_categories == 50


def test_config_resolve_paths(tmp_path):
    import yaml
    config_data = {
        "output": {"directory": "./data/output"},
        "chromadb": {"persist_directory": "./data/chroma_db"},
        "cache": {"directory": "./data/cache"},
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(config_data))

    cfg = load_config(str(config_file))
    assert cfg.output.directory.startswith(str(tmp_path))
    assert not cfg.output.directory.startswith(".")
