import httpx
import pytest
import respx
from conftest import SAMPLE_LITELLM_DATA

from bud_model_catalog.config import CatalogConfig
from bud_model_catalog.exceptions import SourceFetchError
from bud_model_catalog.sources.litellm import LiteLLMSource

TEST_URL = "https://example.com/litellm.json"


@pytest.fixture
def config():
    return CatalogConfig(litellm_url=TEST_URL, max_retries=1)


@pytest.mark.asyncio
async def test_fetches_and_transforms_correctly(config):
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA))
        source = LiteLLMSource(config)
        result = await source.fetch()

    # Should have transformed models with valid providers
    assert "openai/gpt-4o" in result.data
    assert "anthropic/claude-sonnet-4-20250514" in result.data
    assert "gemini/gemini-2.0-flash" in result.data

    # Check transform fields
    entry = result.data["openai/gpt-4o"]
    assert entry["litellm_provider"] == "openai"
    assert entry["metadata"]["original_key"] == "gpt-4o"
    assert entry["license_id"] == "openai-api"
    assert entry["max_tokens"] == 16384

    assert result.source_name == "litellm"
    assert result.fetched_at is not None


@pytest.mark.asyncio
async def test_filters_unmapped_providers(config):
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA))
        source = LiteLLMSource(config)
        result = await source.fetch()

    # unknown_provider should be filtered out
    for key in result.data:
        assert "unknown_provider" not in key


@pytest.mark.asyncio
async def test_vertex_ai_gemini_only_filter(config):
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA))
        source = LiteLLMSource(config)
        result = await source.fetch()

    # vertex_ai/gemini-1.5-pro should be included (has "gemini" in key)
    assert "vertex_ai-gemini-models/gemini-1.5-pro" in result.data
    # vertex_ai/text-bison should be filtered (no "gemini" in key)
    assert "vertex_ai-gemini-models/text-bison" not in result.data


@pytest.mark.asyncio
async def test_removes_sample_spec(config):
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA))
        source = LiteLLMSource(config)
        result = await source.fetch()

    for key in result.data:
        assert "sample_spec" not in key


@pytest.mark.asyncio
async def test_filters_models_without_provider(config):
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA))
        source = LiteLLMSource(config)
        result = await source.fetch()

    for key in result.data:
        assert "no-provider-model" not in key


@pytest.mark.asyncio
async def test_raises_on_http_error(config):
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(500))
        source = LiteLLMSource(config)
        with pytest.raises(SourceFetchError, match="Failed to fetch LiteLLM data"):
            await source.fetch()


@pytest.mark.asyncio
async def test_raises_on_invalid_json(config):
    with respx.mock:
        respx.get(TEST_URL).mock(
            return_value=httpx.Response(
                200, content=b"not json", headers={"content-type": "application/json"}
            )
        )
        source = LiteLLMSource(config)
        with pytest.raises(SourceFetchError):
            await source.fetch()


# ---------- ETag caching tests ----------


@pytest.mark.asyncio
async def test_etag_stored_on_first_fetch(config):
    """After a 200 with an ETag header, source._last_etag is stored."""
    with respx.mock:
        respx.get(TEST_URL).mock(
            return_value=httpx.Response(200, json=SAMPLE_LITELLM_DATA, headers={"etag": '"abc123"'})
        )
        source = LiteLLMSource(config)
        result = await source.fetch()

    assert source._last_etag == '"abc123"'
    assert source._last_result is result


@pytest.mark.asyncio
async def test_304_returns_cached_result(config):
    """Second fetch sends If-None-Match; on 304, the cached result is returned."""
    call_count = 0

    def _side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, json=SAMPLE_LITELLM_DATA, headers={"etag": '"v1"'})
        # Second call — should carry If-None-Match
        assert request.headers.get("if-none-match") == '"v1"'
        return httpx.Response(304)

    with respx.mock:
        respx.get(TEST_URL).mock(side_effect=_side_effect)
        source = LiteLLMSource(config)
        result1 = await source.fetch()
        result2 = await source.fetch()

    assert result2 is result1
    assert call_count == 2


@pytest.mark.asyncio
async def test_cache_disabled_skips_etag():
    """With cache=False, no If-None-Match is sent on the second request."""
    no_cache_config = CatalogConfig(litellm_url=TEST_URL, max_retries=1, cache=False)
    call_count = 0

    def _side_effect(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        assert "if-none-match" not in request.headers
        return httpx.Response(200, json=SAMPLE_LITELLM_DATA, headers={"etag": '"v1"'})

    with respx.mock:
        respx.get(TEST_URL).mock(side_effect=_side_effect)
        source = LiteLLMSource(no_cache_config)
        await source.fetch()
        await source.fetch()

    assert call_count == 2


# ---------- Collision handling tests ----------


@pytest.mark.asyncio
async def test_collision_prefers_prefixed_entry(config):
    """When two LiteLLM entries produce the same catalog key, prefer the prefixed one."""
    collision_data = {
        "deepseek-chat": {
            "input_cost_per_token": 6e-07,
            "max_tokens": 131072,
            "litellm_provider": "deepseek",
            "mode": "chat",
        },
        "deepseek/deepseek-chat": {
            "input_cost_per_token": 2.7e-07,
            "max_tokens": 8192,
            "litellm_provider": "deepseek",
            "mode": "chat",
        },
    }
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=collision_data))
        source = LiteLLMSource(config)
        result = await source.fetch()

    entry = result.data["deepseek/deepseek-chat"]
    # The prefixed entry (deepseek/deepseek-chat) should win
    assert entry["metadata"]["original_key"] == "deepseek/deepseek-chat"
    assert entry["input_cost_per_token"] == 2.7e-07


@pytest.mark.asyncio
async def test_collision_prefers_prefixed_entry_reverse_order(config):
    """Prefixed entry wins even if it comes first in iteration order."""
    from collections import OrderedDict

    collision_data = OrderedDict(
        [
            (
                "deepseek/deepseek-chat",
                {
                    "input_cost_per_token": 2.7e-07,
                    "max_tokens": 8192,
                    "litellm_provider": "deepseek",
                    "mode": "chat",
                },
            ),
            (
                "deepseek-chat",
                {
                    "input_cost_per_token": 6e-07,
                    "max_tokens": 131072,
                    "litellm_provider": "deepseek",
                    "mode": "chat",
                },
            ),
        ]
    )
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=collision_data))
        source = LiteLLMSource(config)
        result = await source.fetch()

    entry = result.data["deepseek/deepseek-chat"]
    assert entry["metadata"]["original_key"] == "deepseek/deepseek-chat"
    assert entry["input_cost_per_token"] == 2.7e-07


@pytest.mark.asyncio
async def test_voice_providers_are_included(config):
    """Voice providers present in LiteLLM must survive the provider filter.

    They are keyed ``{provider}/{model}`` upstream, so the prefix is stripped
    rather than doubled (``deepgram/nova-3``, not ``deepgram/deepgram/nova-3``).
    """
    voice_data = {
        "deepgram/nova-3": {
            "input_cost_per_second": 7.167e-05,
            "output_cost_per_second": 0.0,
            "litellm_provider": "deepgram",
            "mode": "audio_transcription",
        },
        "elevenlabs/eleven_multilingual_v2": {
            "input_cost_per_character": 0.0001,
            "litellm_provider": "elevenlabs",
            "mode": "audio_speech",
        },
        "elevenlabs/scribe_v2": {
            "input_cost_per_second": 6.111e-05,
            "litellm_provider": "elevenlabs",
            "mode": "audio_transcription",
        },
        "aws_polly/neural": {
            "input_cost_per_character": 1.6e-05,
            "litellm_provider": "aws_polly",
            "mode": "audio_speech",
        },
        "groq/whisper-large-v3": {
            "input_cost_per_second": 3.083e-05,
            "litellm_provider": "groq",
            "mode": "audio_transcription",
        },
    }
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=voice_data))
        source = LiteLLMSource(config)
        result = await source.fetch()

    assert set(result.data) == {
        "deepgram/nova-3",
        "elevenlabs/eleven_multilingual_v2",
        "elevenlabs/scribe_v2",
        "aws_polly/neural",
        "groq/whisper-large-v3",
    }

    entry = result.data["deepgram/nova-3"]
    assert entry["litellm_provider"] == "deepgram"
    assert entry["metadata"]["original_key"] == "deepgram/nova-3"
    assert entry["mode"] == "audio_transcription"
    assert entry["input_cost_per_second"] == 7.167e-05


@pytest.mark.asyncio
async def test_a_collision_merges_the_loser_into_the_gaps(config):
    """LiteLLM spreads one model's facts across both keys; keeping one wholesale lost the rest.

    Observed: `vertex_ai/gemini-3-pro-image` won and dropped the unprefixed entry's
    modalities and its $0.014 search-grounding price.
    """
    data = {
        "gemini-3-pro-image": {
            "litellm_provider": "vertex_ai-language-models",
            "mode": "image_generation",
            "input_cost_per_token": 9.9e-06,
            "supported_modalities": ["text", "image"],
            "search_context_cost_per_query": {"search_context_size_low": 0.014},
        },
        "vertex_ai/gemini-3-pro-image": {
            "litellm_provider": "vertex_ai",
            "mode": "image_generation",
            "input_cost_per_token": 2e-06,
        },
    }
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=data))
        result = await LiteLLMSource(config).fetch()

    entry = result.data["vertex_ai-gemini-models/gemini-3-pro-image"]
    assert entry["input_cost_per_token"] == 2e-06  # the canonical entry wins what it has
    assert entry["supported_modalities"] == ["text", "image"]  # the other fills the gap
    assert entry["search_context_cost_per_query"] == {"search_context_size_low": 0.014}
    assert entry["metadata"]["original_key"] == "vertex_ai/gemini-3-pro-image"
    assert entry["metadata"]["merged_from"] == "gemini-3-pro-image"


@pytest.mark.asyncio
async def test_every_vertex_gemini_provider_is_read(config):
    """Vertex files Gemini under three LiteLLM providers; only one used to be mapped."""
    data = {
        "vertex_ai/gemini-3.5-transcribe": {
            "litellm_provider": "vertex_ai",
            "mode": "audio_transcription",
        },
        "gemini-embedding-001": {
            "litellm_provider": "vertex_ai-embedding-models",
            "mode": "embedding",
        },
        "gemini-2.5-pro": {"litellm_provider": "vertex_ai-language-models", "mode": "chat"},
        # Same providers, not Gemini: must stay out of `vertex_ai-gemini-models`.
        "vertex_ai/xai/grok-4.3": {"litellm_provider": "vertex_ai", "mode": "chat"},
        "text-embedding-005": {
            "litellm_provider": "vertex_ai-embedding-models",
            "mode": "embedding",
        },
    }
    with respx.mock:
        respx.get(TEST_URL).mock(return_value=httpx.Response(200, json=data))
        result = await LiteLLMSource(config).fetch()

    assert set(result.data) == {
        "vertex_ai-gemini-models/gemini-3.5-transcribe",
        "vertex_ai-gemini-models/gemini-embedding-001",
        "vertex_ai-gemini-models/gemini-2.5-pro",
    }
