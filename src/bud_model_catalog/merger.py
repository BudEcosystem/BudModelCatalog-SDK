from __future__ import annotations

import logging
from datetime import date

from .config import CatalogConfig
from .mappings import LITELLM_TO_RESEARCH, extract_model_name
from .models import CatalogResult, MergeStats
from .sources.base import FetchResult

logger = logging.getLogger(__name__)

COMMON_COST_FIELDS = (
    "input_cost_per_token",
    "output_cost_per_token",
    "input_cost_per_token_batches",
    "output_cost_per_token_batches",
    "input_cost_per_image",
    "input_cost_per_audio_token",
    "output_cost_per_audio_token",
    "input_cost_per_character",
    "input_cost_per_query",
    "input_cost_per_second",
    "output_cost_per_second",
    "cache_read_input_token_cost",
    "cache_creation_input_token_cost",
    "cache_creation_input_audio_token_cost",
)


def _is_litellm_deprecated(entry: dict) -> bool:
    """Check if a LiteLLM model is deprecated based on its deprecation_date."""
    dep_date = entry.get("deprecation_date")
    if not dep_date:
        return False
    try:
        return date.fromisoformat(dep_date) <= date.today()
    except ValueError:
        logger.warning("Invalid deprecation_date %r, treating as not deprecated", dep_date)
        return False


def merge(
    litellm_result: FetchResult,
    ai_models_result: FetchResult | None,
    config: CatalogConfig,
) -> CatalogResult:
    litellm_models: dict[str, dict] = litellm_result.data
    total_litellm = len(litellm_models)

    # Fallback mode: no ai-models data available
    if ai_models_result is None:
        if config.include_deprecated:
            fallback_models = dict(litellm_models)
            fallback_deprecated = 0
        else:
            fallback_models = {}
            fallback_deprecated = 0
            for key, entry in litellm_models.items():
                if _is_litellm_deprecated(entry):
                    fallback_deprecated += 1
                else:
                    fallback_models[key] = entry
        return CatalogResult(
            models=fallback_models,
            stats=MergeStats(
                total_litellm=total_litellm,
                total_output=len(fallback_models),
                matched=0,
                unmatched=len(fallback_models),
                deprecated_removed=fallback_deprecated,
                cost_fields_updated=0,
            ),
            litellm_fetched_at=litellm_result.fetched_at,
            ai_models_fetched_at=None,
        )

    research_lookup: dict[tuple[str, str], dict] = ai_models_result.data

    output: dict[str, dict] = {}
    matched = 0
    unmatched = 0
    deprecated_removed = 0
    cost_fields_updated = 0

    for tz_key, tz_entry in litellm_models.items():
        litellm_provider = tz_entry.get("litellm_provider", "")

        # Check if provider is mapped to ai-models
        if litellm_provider not in LITELLM_TO_RESEARCH:
            if not config.include_deprecated and _is_litellm_deprecated(tz_entry):
                deprecated_removed += 1
                continue
            output[tz_key] = tz_entry
            unmatched += 1
            continue

        # Extract model name for lookup
        original_key = tz_entry.get("metadata", {}).get("original_key", "")
        model_name = extract_model_name(litellm_provider, original_key)
        research_provider = LITELLM_TO_RESEARCH[litellm_provider]

        # Look up in ai-models
        research_key = (research_provider, model_name)
        research_entry = research_lookup.get(research_key)

        if research_entry is None:
            if not config.include_deprecated and _is_litellm_deprecated(tz_entry):
                deprecated_removed += 1
                continue
            output[tz_key] = tz_entry
            unmatched += 1
            continue

        # Check deprecated (ai-models flag OR LiteLLM deprecation_date)
        is_deprecated = research_entry.get("isDeprecated", False) or _is_litellm_deprecated(
            tz_entry
        )
        if is_deprecated and not config.include_deprecated:
            deprecated_removed += 1
            continue

        # Overlay cost fields from ai-models
        research_costs = research_entry.get("costs", {})
        updated_entry = dict(tz_entry)

        for field in COMMON_COST_FIELDS:
            if field in research_costs:
                new_val = research_costs[field]
                old_val = updated_entry.get(field)
                if old_val != new_val:
                    cost_fields_updated += 1
                updated_entry[field] = new_val

        output[tz_key] = updated_entry
        matched += 1

    logger.info(
        "Merge: %d matched, %d unmatched, %d deprecated removed, %d cost fields updated",
        matched,
        unmatched,
        deprecated_removed,
        cost_fields_updated,
    )

    return CatalogResult(
        models=output,
        stats=MergeStats(
            total_litellm=total_litellm,
            total_output=len(output),
            matched=matched,
            unmatched=unmatched,
            deprecated_removed=deprecated_removed,
            cost_fields_updated=cost_fields_updated,
        ),
        litellm_fetched_at=litellm_result.fetched_at,
        ai_models_fetched_at=ai_models_result.fetched_at,
    )
