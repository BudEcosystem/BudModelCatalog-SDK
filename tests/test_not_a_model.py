"""Entries in LiteLLM's price map that are not models anyone can deploy.

Each of these reached bud-connect as a "model" with no endpoint, which budapp then offered as
deployable. The rules were checked against the live map and each removed exactly the entries
named here and nothing else.
"""

import pytest

from bud_model_catalog.sources.litellm import not_a_model


@pytest.mark.parametrize(
    "key",
    [
        "fireworks-ai-4.1b-to-16b",
        "fireworks-ai-56b-to-176b",
        "fireworks-ai-embedding-up-to-150m",
        "fireworks-ai-default",
    ],
)
def test_pricing_tiers_with_no_mode_are_not_models(key):
    """Size-bucket tiers. With no mode there is nothing to derive a route from."""
    assert not_a_model(key, {"litellm_provider": "fireworks_ai", "input_cost_per_token": 1e-07})


def test_bedrock_guardrails_is_not_a_model():
    """A content-filtering product applied to another model's traffic."""
    assert not_a_model("bedrock/guardrails", {"litellm_provider": "bedrock", "mode": "guardrail"})


@pytest.mark.parametrize(
    "key",
    [
        "deepgram/streaming/nova-3",
        "deepgram/streaming/nova-3-multilingual",
        "deepgram/streaming/diarize",
        "deepgram/streaming/redact",
        "deepgram/streaming/keyterm",
        "deepgram/streaming/detect_entities",
    ],
)
def test_deepgram_streaming_entries_are_pricing_skus(key):
    """Deepgram's /v1/models lists no streaming model and no `nova-3-multilingual`.

    Streaming is a way of calling the nova-3 models already in the catalog; the other four
    are per-feature surcharges.
    """
    assert not_a_model(key, {"litellm_provider": "deepgram", "mode": "audio_transcription"})


@pytest.mark.parametrize(
    ("key", "data"),
    [
        ("deepgram/nova-3", {"litellm_provider": "deepgram", "mode": "audio_transcription"}),
        ("gpt-4o", {"litellm_provider": "openai", "mode": "chat"}),
        (
            "bedrock/stability.stable-fast-upscale-v1:0",
            {"litellm_provider": "bedrock", "mode": "image_edit"},
        ),
        ("gpt-realtime", {"litellm_provider": "openai", "mode": "realtime"}),
    ],
)
def test_real_models_are_kept(key, data):
    """The rules must not reach past the entries they were written for.

    Realtime models are deliberately kept: they ARE models, and whether Bud can route them
    is a question for the consumer, not a reason to erase them from the catalog.
    """
    assert not_a_model(key, data) is None


def test_the_streaming_rule_is_scoped_to_deepgram():
    """Another vendor using `/streaming/` in a key is not assumed to mean the same thing."""
    assert (
        not_a_model("vendor/streaming/model", {"litellm_provider": "openai", "mode": "chat"})
        is None
    )
