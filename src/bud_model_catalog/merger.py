#  -----------------------------------------------------------------------------
#  Copyright (c) 2024 Bud Ecosystem Inc.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  -----------------------------------------------------------------------------

"""Merge strategy for combining LiteLLM and ai-models data.

LiteLLM provides the base catalog (model metadata, context windows, etc.).
ai-models provides cost-accurate pricing from truefoundry/models.  The
merge overlays ai-models cost fields onto LiteLLM entries matched via
provider-aware name mapping, and optionally filters deprecated models.
"""

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


def _extract_flat_costs(costs_field: object) -> dict:
    """Extract a flat cost dict from the research costs field.

    The research YAML stores costs as a list of region-specific dicts like
    ``[{"region": "*", "input_cost_per_token": ...}, ...]``.  This helper
    picks the wildcard (``region: "*"``) entry when available, falls back to
    ``{}`` for multi-region models (so LiteLLM costs are preserved), and
    passes through a plain dict unchanged for backward compatibility.
    """
    if isinstance(costs_field, dict):
        return costs_field
    if not isinstance(costs_field, list) or not costs_field:
        return {}

    for entry in costs_field:
        if isinstance(entry, dict) and entry.get("region") == "*":
            return entry

    return {}


def merge(
    litellm_result: FetchResult,
    ai_models_result: FetchResult | None,
    config: CatalogConfig,
) -> CatalogResult:
    """Merge LiteLLM and ai-models data into a unified catalog.

    Uses LiteLLM as the base catalog and overlays cost fields from
    ai-models where a match is found via provider-aware name mapping.
    Deprecated models are filtered unless ``config.include_deprecated``
    is set.  When *ai_models_result* is ``None``, returns LiteLLM data
    as-is (with deprecation filtering only).
    """
    litellm_models: dict[str, dict] = litellm_result.data
    total_litellm = len(litellm_models)

    # Fallback mode: no ai-models data available
    if ai_models_result is None:
        if config.include_deprecated:
            fallback_models = dict(litellm_models)
            fallback_deprecated = 0
        else:
            fallback_models = {
                key: entry
                for key, entry in litellm_models.items()
                if not _is_litellm_deprecated(entry)
            }
            fallback_deprecated = total_litellm - len(fallback_models)
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

        # Try to find a matching ai-models entry
        research_entry: dict | None = None
        if litellm_provider in LITELLM_TO_RESEARCH:
            original_key = tz_entry.get("metadata", {}).get("original_key", "")
            model_name = extract_model_name(litellm_provider, original_key)
            research_provider = LITELLM_TO_RESEARCH[litellm_provider]
            research_entry = research_lookup.get((research_provider, model_name))

        # Determine deprecation status
        is_deprecated = _is_litellm_deprecated(tz_entry) or (
            research_entry is not None and research_entry.get("isDeprecated", False)
        )

        if is_deprecated and not config.include_deprecated:
            deprecated_removed += 1
            continue

        # No match — keep original LiteLLM entry as-is
        if research_entry is None:
            output[tz_key] = tz_entry
            unmatched += 1
            continue

        # Overlay cost fields from ai-models
        research_costs = _extract_flat_costs(research_entry.get("costs", {}))
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
