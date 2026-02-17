import pytest

from bud_model_catalog.config import CatalogConfig


def test_rejects_zero_timeout():
    with pytest.raises(ValueError, match="timeout must be positive"):
        CatalogConfig(timeout=0)


def test_rejects_negative_timeout():
    with pytest.raises(ValueError, match="timeout must be positive"):
        CatalogConfig(timeout=-5)


def test_rejects_invalid_litellm_url():
    with pytest.raises(ValueError, match="litellm_url must be an HTTP"):
        CatalogConfig(litellm_url="ftp://bad")


def test_rejects_invalid_ai_models_url():
    with pytest.raises(ValueError, match="ai_models_url must be an HTTP"):
        CatalogConfig(ai_models_url="/local/path")


def test_rejects_zero_max_retries():
    with pytest.raises(ValueError, match="max_retries must be at least 1"):
        CatalogConfig(max_retries=0)


def test_rejects_negative_max_retries():
    with pytest.raises(ValueError, match="max_retries must be at least 1"):
        CatalogConfig(max_retries=-1)


def test_accepts_valid_config():
    c = CatalogConfig(timeout=10, max_retries=3)
    assert c.timeout == 10
    assert c.max_retries == 3
