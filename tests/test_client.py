import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from bud_model_catalog import CatalogClient, CatalogConfig
from bud_model_catalog.exceptions import SourceFetchError
from conftest import SAMPLE_LITELLM_DATA, build_ai_models_zip

LITELLM_URL = "https://example.com/litellm.json"
AI_MODELS_URL = "https://example.com/ai-models.zip"


@pytest.fixture
def config():
    return CatalogConfig(litellm_url=LITELLM_URL, ai_models_url=AI_MODELS_URL, max_retries=1)


@pytest.mark.asyncio
async def test_end_to_end_both_sources(config):
    zip_bytes = build_ai_models_zip()
    with respx.mock:
        respx.get(LITELLM_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA)
        )
        respx.get(AI_MODELS_URL).mock(
            return_value=httpx.Response(200, content=zip_bytes)
        )

        client = CatalogClient(config)
        result = await client.fetch_catalog()

    assert len(result.models) > 0
    assert result.stats.total_litellm > 0
    assert result.litellm_fetched_at is not None
    assert result.ai_models_fetched_at is not None

    # Check a model got cost overlay
    if "openai/gpt-4o" in result.models:
        entry = result.models["openai/gpt-4o"]
        assert entry["input_cost_per_token"] == 2.5e-06  # From ai-models


@pytest.mark.asyncio
async def test_fallback_on_ai_models_failure(config):
    with respx.mock:
        respx.get(LITELLM_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA)
        )
        respx.get(AI_MODELS_URL).mock(return_value=httpx.Response(500))

        client = CatalogClient(config)
        result = await client.fetch_catalog()

    assert len(result.models) > 0
    assert result.ai_models_fetched_at is None
    assert result.stats.matched == 0
    assert result.stats.unmatched == result.stats.total_litellm

    # Costs should be original LiteLLM values
    if "openai/gpt-4o" in result.models:
        assert result.models["openai/gpt-4o"]["input_cost_per_token"] == 5e-06


@pytest.mark.asyncio
async def test_raises_on_litellm_failure(config):
    zip_bytes = build_ai_models_zip()
    with respx.mock:
        respx.get(LITELLM_URL).mock(return_value=httpx.Response(500))
        respx.get(AI_MODELS_URL).mock(
            return_value=httpx.Response(200, content=zip_bytes)
        )

        client = CatalogClient(config)
        with pytest.raises(SourceFetchError):
            await client.fetch_catalog()


def test_fetch_catalog_sync_works(config):
    """Sync wrapper returns correct result outside an event loop."""
    zip_bytes = build_ai_models_zip()
    with respx.mock:
        respx.get(LITELLM_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA)
        )
        respx.get(AI_MODELS_URL).mock(
            return_value=httpx.Response(200, content=zip_bytes)
        )

        client = CatalogClient(config)
        result = client.fetch_catalog_sync()

    assert len(result.models) > 0
    assert result.stats.total_litellm > 0
    assert result.litellm_fetched_at is not None


@pytest.mark.asyncio
async def test_end_to_end_empty_litellm(config):
    """Empty LiteLLM data (after removing sample_spec) produces empty catalog."""
    empty_data = {"sample_spec": {"sample_key": "sample_value"}}
    zip_bytes = build_ai_models_zip()
    with respx.mock:
        respx.get(LITELLM_URL).mock(
            return_value=httpx.Response(200, json=empty_data)
        )
        respx.get(AI_MODELS_URL).mock(
            return_value=httpx.Response(200, content=zip_bytes)
        )

        client = CatalogClient(config)
        result = await client.fetch_catalog()

    assert result.models == {}
    assert result.stats.total_litellm == 0
    assert result.stats.total_output == 0
    assert result.stats.matched == 0
    assert result.stats.unmatched == 0


# ---------- BUG-2: non-SourceFetchError in ai-models is caught ----------


@pytest.mark.asyncio
async def test_non_source_fetch_error_in_ai_models_is_caught(config):
    """A TypeError (or any non-SourceFetchError) from ai-models doesn't cancel LiteLLM."""
    with respx.mock:
        respx.get(LITELLM_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA)
        )
        # ai-models will never be called because we patch the source directly
        respx.get(AI_MODELS_URL).mock(return_value=httpx.Response(200, content=b""))

        client = CatalogClient(config)
        # Patch the ai_models source to raise a TypeError
        client._ai_models_source.fetch = AsyncMock(side_effect=TypeError("unexpected"))
        result = await client.fetch_catalog()

    # LiteLLM still worked — we get models
    assert len(result.models) > 0
    # ai-models was not available
    assert result.ai_models_fetched_at is None


# ---------- BUG-7: fetch_catalog_sync from a running event loop ----------


def test_fetch_catalog_sync_from_running_loop(config):
    """fetch_catalog_sync works when called from within a running event loop (ThreadPoolExecutor path)."""
    zip_bytes = build_ai_models_zip()

    async def _run_in_thread() -> None:
        with respx.mock:
            respx.get(LITELLM_URL).mock(
                return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA)
            )
            respx.get(AI_MODELS_URL).mock(
                return_value=httpx.Response(200, content=zip_bytes)
            )
            client = CatalogClient(config)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, client.fetch_catalog_sync)
            assert len(result.models) > 0
            assert result.stats.total_litellm > 0

    asyncio.run(_run_in_thread())
