import io
import zipfile

import httpx
import pytest
import respx
from conftest import build_ai_models_zip

from bud_model_catalog.config import CatalogConfig
from bud_model_catalog.exceptions import SourceFetchError
from bud_model_catalog.mappings import extract_model_name
from bud_model_catalog.sources import ai_models as ai_models_mod
from bud_model_catalog.sources.ai_models import AiModelsSource

TEST_URL = "https://example.com/ai-models.zip"


@pytest.fixture
def config():
    return CatalogConfig(ai_models_url=TEST_URL, max_retries=1)


@pytest.mark.asyncio
async def test_builds_lookup_correctly(config):
    zip_bytes = build_ai_models_zip()
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, content=zip_bytes))
        source = AiModelsSource(config)
        result = await source.fetch()

    lookup = result.data
    assert ("openai", "gpt-4o") in lookup
    assert ("anthropic", "claude-sonnet-4-20250514") in lookup
    assert ("google-gemini", "gemini-2.0-flash") in lookup
    assert lookup[("openai", "gpt-4o")]["costs"][0]["input_cost_per_token"] == 2.5e-06
    assert result.source_name == "ai_models"


def test_extract_model_name_strips_prefixes():
    assert extract_model_name("azure", "azure/gpt-4") == "gpt-4"
    assert extract_model_name("gemini", "gemini/gemini-pro") == "gemini-pro"
    assert extract_model_name("xai", "xai/grok-1") == "grok-1"
    assert extract_model_name("mistral", "mistral/mistral-large") == "mistral-large"
    assert extract_model_name("together_ai", "together_ai/llama-3") == "llama-3"


def test_extract_model_name_no_prefix():
    assert extract_model_name("openai", "gpt-4o") == "gpt-4o"
    assert extract_model_name("anthropic", "claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"


def test_extract_model_name_vertex_gemini():
    assert (
        extract_model_name("vertex_ai-gemini-models", "vertex_ai/gemini-1.5-pro")
        == "gemini-1.5-pro"
    )


def test_extract_model_name_vertex_anthropic():
    assert (
        extract_model_name("vertex_ai-anthropic_models", "vertex_ai/claude-3-5-haiku")
        == "anthropic/claude-3-5-haiku"
    )
    assert extract_model_name("vertex_ai-anthropic_models", "some-other-key") == "some-other-key"


@pytest.mark.asyncio
async def test_raises_on_http_error(config):
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(500))
        source = AiModelsSource(config)
        with pytest.raises(SourceFetchError, match="Failed to fetch models archive"):
            await source.fetch()


@pytest.mark.asyncio
async def test_raises_on_invalid_zip(config):
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, content=b"not a zip file"))
        source = AiModelsSource(config)
        with pytest.raises(SourceFetchError, match="not a valid ZIP"):
            await source.fetch()


@pytest.mark.asyncio
async def test_skips_malformed_yaml_in_zip(config):
    """ZIP containing invalid YAML logs warning and returns other valid entries."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Valid YAML entry
        zf.writestr(
            "models-main/providers/openai/gpt-4o.yaml",
            "model: gpt-4o\ncosts:\n  input_cost_per_token: 2.5e-06\nisDeprecated: false\n",
        )
        # Malformed YAML entry
        zf.writestr(
            "models-main/providers/openai/bad-model.yaml",
            "model: bad\n  invalid: yaml: content:\n    - [broken",
        )
    zip_bytes = buf.getvalue()

    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, content=zip_bytes))
        source = AiModelsSource(config)
        result = await source.fetch()

    # The valid entry should be present
    assert ("openai", "gpt-4o") in result.data
    # Only one entry (the malformed one was skipped)
    assert len(result.data) == 1


@pytest.mark.asyncio
async def test_empty_zip_returns_empty_lookup(config):
    """ZIP with no model YAML files returns empty lookup."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add a non-model file
        zf.writestr("models-main/README.md", "# Models repo")
    zip_bytes = buf.getvalue()

    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, content=zip_bytes))
        source = AiModelsSource(config)
        result = await source.fetch()

    assert result.data == {}


# ---------- ETag caching tests ----------


@pytest.mark.asyncio
async def test_etag_stored_on_first_fetch(config):
    """After a 200 with an ETag header, source._last_etag is stored."""
    zip_bytes = build_ai_models_zip()
    with respx.mock:
        respx.get(TEST_URL).mock(
            return_value=httpx.Response(200, content=zip_bytes, headers={"etag": '"zip-v1"'})
        )
        source = AiModelsSource(config)
        result = await source.fetch()

    assert source._last_etag == '"zip-v1"'
    assert source._last_result is result


@pytest.mark.asyncio
async def test_304_returns_cached_result(config):
    """Second fetch sends If-None-Match; on 304, the cached result is returned."""
    zip_bytes = build_ai_models_zip()
    call_count = 0

    def _side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, content=zip_bytes, headers={"etag": '"zip-v1"'})
        assert request.headers.get("if-none-match") == '"zip-v1"'
        return httpx.Response(304)

    with respx.mock:
        respx.get(TEST_URL).mock(side_effect=_side_effect)
        source = AiModelsSource(config)
        result1 = await source.fetch()
        result2 = await source.fetch()

    assert result2 is result1
    assert call_count == 2


@pytest.mark.asyncio
async def test_cache_disabled_skips_etag():
    """With cache=False, no If-None-Match is sent on the second request."""
    no_cache_config = CatalogConfig(ai_models_url=TEST_URL, max_retries=1, cache=False)
    zip_bytes = build_ai_models_zip()
    call_count = 0

    def _side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        assert "if-none-match" not in request.headers
        return httpx.Response(200, content=zip_bytes, headers={"etag": '"zip-v1"'})

    with respx.mock:
        respx.get(TEST_URL).mock(side_effect=_side_effect)
        source = AiModelsSource(no_cache_config)
        await source.fetch()
        await source.fetch()

    assert call_count == 2


# ---------- Zip bomb protection ----------


@pytest.mark.asyncio
async def test_rejects_oversized_response(config, monkeypatch):
    """Response exceeding _MAX_RESPONSE_BYTES is rejected."""
    monkeypatch.setattr(ai_models_mod, "_MAX_RESPONSE_BYTES", 100)
    # Build a zip that exceeds 100 bytes
    zip_bytes = build_ai_models_zip()
    assert len(zip_bytes) > 100  # sanity check

    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, content=zip_bytes))
        source = AiModelsSource(config)
        with pytest.raises(SourceFetchError, match="too large"):
            await source.fetch()
