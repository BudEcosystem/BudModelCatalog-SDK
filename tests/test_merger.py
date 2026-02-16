from datetime import date, datetime, timedelta, timezone

import pytest

from bud_model_catalog.config import CatalogConfig
from bud_model_catalog.merger import merge
from bud_model_catalog.sources.base import FetchResult

NOW = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


def _litellm_result(models: dict) -> FetchResult:
    return FetchResult(data=models, source_name="litellm", fetched_at=NOW)


def _ai_models_result(lookup: dict) -> FetchResult:
    return FetchResult(data=lookup, source_name="ai_models", fetched_at=NOW)


@pytest.fixture
def litellm_models():
    return {
        "openai/gpt-4o": {
            "litellm_provider": "openai",
            "input_cost_per_token": 5e-06,
            "output_cost_per_token": 1.5e-05,
            "max_tokens": 16384,
            "metadata": {"original_key": "gpt-4o"},
        },
        "anthropic/claude-sonnet-4-20250514": {
            "litellm_provider": "anthropic",
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
            "max_tokens": 8192,
            "metadata": {"original_key": "claude-sonnet-4-20250514"},
        },
        "gemini/gemini/gemini-2.0-flash": {
            "litellm_provider": "gemini",
            "input_cost_per_token": 1e-07,
            "output_cost_per_token": 4e-07,
            "max_tokens": 8192,
            "metadata": {"original_key": "gemini/gemini-2.0-flash"},
        },
        "fireworks_ai-embedding-models/some-model": {
            "litellm_provider": "fireworks_ai-embedding-models",
            "input_cost_per_token": 1e-06,
            "max_tokens": 4096,
            "metadata": {"original_key": "some-model"},
        },
    }


@pytest.fixture
def ai_models_lookup():
    return {
        ("openai", "gpt-4o"): {
            "provider": "openai",
            "model": "gpt-4o",
            "costs": {
                "input_cost_per_token": 2.5e-06,
                "output_cost_per_token": 1e-05,
            },
            "isDeprecated": False,
        },
        ("anthropic", "claude-sonnet-4-20250514"): {
            "provider": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "costs": {
                "input_cost_per_token": 3e-06,
                "output_cost_per_token": 1.5e-05,
            },
            "isDeprecated": False,
        },
        ("google-gemini", "gemini-2.0-flash"): {
            "provider": "google-gemini",
            "model": "gemini-2.0-flash",
            "costs": {
                "input_cost_per_token": 7.5e-08,
                "output_cost_per_token": 3e-07,
            },
            "isDeprecated": False,
        },
    }


def test_matched_model_gets_ai_models_costs(litellm_models, ai_models_lookup):
    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    entry = result.models["openai/gpt-4o"]
    # Cost should be from ai-models (lower)
    assert entry["input_cost_per_token"] == 2.5e-06
    assert entry["output_cost_per_token"] == 1e-05
    # Non-cost fields preserved
    assert entry["max_tokens"] == 16384


def test_unmatched_model_keeps_litellm_costs(litellm_models, ai_models_lookup):
    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    # fireworks has no match in ai-models
    entry = result.models["fireworks_ai-embedding-models/some-model"]
    assert entry["input_cost_per_token"] == 1e-06


def test_deprecated_model_removed(litellm_models, ai_models_lookup):
    # Mark openai/gpt-4o as deprecated
    ai_models_lookup[("openai", "gpt-4o")]["isDeprecated"] = True

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "openai/gpt-4o" not in result.models
    assert result.stats.deprecated_removed == 1


def test_include_deprecated_keeps_model(litellm_models, ai_models_lookup):
    ai_models_lookup[("openai", "gpt-4o")]["isDeprecated"] = True

    config = CatalogConfig(include_deprecated=True)
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "openai/gpt-4o" in result.models
    # Should still get cost overlay
    assert result.models["openai/gpt-4o"]["input_cost_per_token"] == 2.5e-06


def test_fallback_mode_no_ai_models(litellm_models):
    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), None, config)

    assert len(result.models) == len(litellm_models)
    assert result.stats.matched == 0
    assert result.stats.unmatched == len(litellm_models)
    assert result.ai_models_fetched_at is None
    # Costs should be original LiteLLM values
    assert result.models["openai/gpt-4o"]["input_cost_per_token"] == 5e-06


def test_stats_are_correct(litellm_models, ai_models_lookup):
    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert result.stats.total_litellm == 4
    assert result.stats.total_output == 4
    assert result.stats.matched == 3  # openai, anthropic, gemini
    assert result.stats.unmatched == 1  # fireworks
    assert result.stats.deprecated_removed == 0
    assert result.stats.cost_fields_updated > 0  # openai and gemini had different costs


def test_cost_fields_updated_count(litellm_models, ai_models_lookup):
    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    # openai: input_cost changed (5e-06 -> 2.5e-06), output changed (1.5e-05 -> 1e-05) = 2
    # anthropic: same costs, no change = 0
    # gemini: input changed (1e-07 -> 7.5e-08), output changed (4e-07 -> 3e-07) = 2
    assert result.stats.cost_fields_updated == 4


# --- LiteLLM deprecation_date filtering tests ---

PAST_DATE = (date.today() - timedelta(days=30)).isoformat()
FUTURE_DATE = (date.today() + timedelta(days=30)).isoformat()
TODAY_DATE = date.today().isoformat()


def test_matched_model_with_past_deprecation_date_removed(litellm_models, ai_models_lookup):
    """Matched model with past deprecation_date is removed when include_deprecated=False."""
    litellm_models["openai/gpt-4o"]["deprecation_date"] = PAST_DATE

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "openai/gpt-4o" not in result.models
    assert result.stats.deprecated_removed == 1


def test_matched_model_with_future_deprecation_date_kept(litellm_models, ai_models_lookup):
    """Matched model with future deprecation_date is kept."""
    litellm_models["openai/gpt-4o"]["deprecation_date"] = FUTURE_DATE

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "openai/gpt-4o" in result.models
    assert result.stats.deprecated_removed == 0


def test_matched_model_with_today_deprecation_date_removed(litellm_models, ai_models_lookup):
    """Matched model with today's deprecation_date is removed (today counts as deprecated)."""
    litellm_models["openai/gpt-4o"]["deprecation_date"] = TODAY_DATE

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "openai/gpt-4o" not in result.models
    assert result.stats.deprecated_removed == 1


def test_matched_model_with_past_deprecation_date_kept_when_include_deprecated(litellm_models, ai_models_lookup):
    """Matched model with past deprecation_date is kept when include_deprecated=True."""
    litellm_models["openai/gpt-4o"]["deprecation_date"] = PAST_DATE

    config = CatalogConfig(include_deprecated=True)
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "openai/gpt-4o" in result.models
    assert result.models["openai/gpt-4o"]["input_cost_per_token"] == 2.5e-06


def test_unmatched_model_with_past_deprecation_date_removed(litellm_models, ai_models_lookup):
    """Unmatched model (no ai-models entry) with past deprecation_date is removed."""
    litellm_models["fireworks_ai-embedding-models/some-model"]["deprecation_date"] = PAST_DATE

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "fireworks_ai-embedding-models/some-model" not in result.models
    assert result.stats.deprecated_removed == 1
    assert result.stats.unmatched == 0


def test_unmatched_model_with_future_deprecation_date_kept(litellm_models, ai_models_lookup):
    """Unmatched model with future deprecation_date is kept."""
    litellm_models["fireworks_ai-embedding-models/some-model"]["deprecation_date"] = FUTURE_DATE

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "fireworks_ai-embedding-models/some-model" in result.models


def test_fallback_mode_filters_by_deprecation_date(litellm_models):
    """Fallback mode (no ai-models) filters by deprecation_date."""
    litellm_models["openai/gpt-4o"]["deprecation_date"] = PAST_DATE

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), None, config)

    assert "openai/gpt-4o" not in result.models
    assert result.stats.deprecated_removed == 1
    assert result.stats.total_output == len(litellm_models) - 1


def test_fallback_mode_keeps_deprecated_when_include_deprecated(litellm_models):
    """Fallback mode keeps deprecated models when include_deprecated=True."""
    litellm_models["openai/gpt-4o"]["deprecation_date"] = PAST_DATE

    config = CatalogConfig(include_deprecated=True)
    result = merge(_litellm_result(litellm_models), None, config)

    assert "openai/gpt-4o" in result.models
    assert result.stats.deprecated_removed == 0
    assert result.stats.total_output == len(litellm_models)


def test_both_is_deprecated_and_deprecation_date_counts_once(litellm_models, ai_models_lookup):
    """Model with both isDeprecated=True and past deprecation_date counts as one removal."""
    ai_models_lookup[("openai", "gpt-4o")]["isDeprecated"] = True
    litellm_models["openai/gpt-4o"]["deprecation_date"] = PAST_DATE

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "openai/gpt-4o" not in result.models
    assert result.stats.deprecated_removed == 1


def test_invalid_deprecation_date_treated_as_not_deprecated(litellm_models, ai_models_lookup):
    """Model with invalid deprecation_date format is treated as not deprecated."""
    litellm_models["openai/gpt-4o"]["deprecation_date"] = "not-a-date"

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    assert "openai/gpt-4o" in result.models


# --- Empty input tests ---


def test_empty_litellm_models():
    """Empty LiteLLM dict produces empty output and zero stats."""
    config = CatalogConfig()
    result = merge(_litellm_result({}), _ai_models_result({}), config)

    assert result.models == {}
    assert result.stats.total_litellm == 0
    assert result.stats.total_output == 0
    assert result.stats.matched == 0
    assert result.stats.unmatched == 0
    assert result.stats.deprecated_removed == 0
    assert result.stats.cost_fields_updated == 0


def test_empty_litellm_models_fallback():
    """Empty LiteLLM dict in fallback mode (no ai-models) produces empty output."""
    config = CatalogConfig()
    result = merge(_litellm_result({}), None, config)

    assert result.models == {}
    assert result.stats.total_litellm == 0
    assert result.stats.total_output == 0


def test_empty_costs_in_research(litellm_models, ai_models_lookup):
    """Matched model with empty costs dict keeps LiteLLM costs."""
    ai_models_lookup[("openai", "gpt-4o")]["costs"] = {}

    config = CatalogConfig()
    result = merge(_litellm_result(litellm_models), _ai_models_result(ai_models_lookup), config)

    entry = result.models["openai/gpt-4o"]
    # Costs should be unchanged from LiteLLM since ai-models has empty costs
    assert entry["input_cost_per_token"] == 5e-06
    assert entry["output_cost_per_token"] == 1.5e-05
