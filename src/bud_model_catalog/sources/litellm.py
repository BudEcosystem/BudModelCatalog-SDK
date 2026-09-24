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

"""LiteLLM model-prices data source.

Fetches the upstream LiteLLM ``model_prices_and_context_window.json``,
filters to supported TensorZero providers, and transforms each entry
into the internal catalog key format (``{tz_provider}/{original_key}``).
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone

from ..config import CatalogConfig
from ..exceptions import SourceFetchError
from ..mappings import STRIP_PREFIXES, strip_provider_prefix
from .base import BaseSource, FetchResult

logger = logging.getLogger(__name__)

# LiteLLM provider -> TensorZero provider
LITELLM_TO_TENSORZERO = {
    "anthropic": "anthropic",
    "assemblyai": "assemblyai",
    "aws_polly": "aws_polly",
    "azure": "azure",
    "bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "deepgram": "deepgram",
    "deepseek": "deepseek",
    "elevenlabs": "elevenlabs",
    "fireworks_ai": "fireworks_ai-embedding-models",
    "fireworks_ai-embedding-models": "fireworks_ai-embedding-models",
    "gemini": "gemini",
    "groq": "groq",
    "hyperbolic": "hyperbolicai",
    "mistral": "mistral",
    "moonshot": "moonshotai",
    "openai": "openai",
    "sagemaker": "sagemaker",
    "together_ai": "together_ai",
    "transcribe": "aws_transcribe",
    "vertex_ai-anthropic_models": "vertex_ai-anthropic_models",
    "vertex_ai-language-models": "vertex_ai-gemini-models",
    "xai": "xai",
}

# TensorZero provider -> license_id
PROVIDER_LICENSE_MAP = {
    "anthropic": "Anthropic",
    "deepseek": "deepseek-custom",
    "gemini": "Google",
    "mistral": "mistral ai",
    "moonshotai": "moonshotai",
    "openai": "openai-api",
    "vertex_ai-anthropic_models": "Anthropic",
    "vertex_ai-gemini-models": "Google",
    "xai": "xAI",
}

# Valid TensorZero provider keys (from tensorzero_providers.json)
TENSORZERO_PROVIDERS: frozenset[str] = frozenset(
    {
        "anthropic",
        "assemblyai",
        "aws_polly",
        "azure",
        "azure_content_safety",
        "bedrock",
        "bud_sentinel",
        "deepgram",
        "deepseek",
        "elevenlabs",
        "fireworks_ai-embedding-models",
        "gemini",
        "groq",
        "huggingface",
        "hyperbolicai",
        "mistral",
        "moonshotai",
        "openai",
        "sagemaker",
        "aws_transcribe",
        "together_ai",
        "vertex_ai-anthropic_models",
        "vertex_ai-gemini-models",
        "xai",
    }
)


def not_a_model(original_key: str, model_data: dict) -> str | None:
    """Why a LiteLLM entry is not a model anyone can deploy, or None if it is one.

    LiteLLM's price map is a price map. Most entries are models, but some are pricing
    records that happen to sit alongside them, and publishing those as models puts entries
    in bud-connect that no route can ever serve: they arrive with no endpoint, and budapp
    offers them as deployable anyway. Each rule below is here because it removed exactly
    the entries it names and nothing else when checked against the live map.

    Returns:
        A short reason, used for the skip log, or None to keep the entry.
    """
    mode = model_data.get("mode")

    if not mode:
        # Size-bucket pricing tiers such as `fireworks-ai-4.1b-to-16b` and
        # `fireworks-ai-embedding-up-to-150m`. Without a mode there is nothing to derive a
        # modality or an endpoint from, so the entry is unroutable by construction.
        return "no mode (a pricing tier, not a model)"

    if mode == "guardrail":
        # `bedrock/guardrails` prices Bedrock's content-filtering product, which is applied
        # to another model's traffic rather than deployed on its own.
        return "guardrail product, not a model"

    if model_data.get("litellm_provider") == "deepgram" and "/streaming/" in original_key:
        # `deepgram/streaming/*` are PRICING SKUs, not models. Deepgram's own /v1/models
        # lists no streaming model and no `nova-3-multilingual`: streaming is a way of
        # calling the same nova-3 models already in the catalog, and "multilingual" is nova-3
        # with language=multi. The other four -- detect_entities, diarize, keyterm, redact --
        # are per-feature surcharges, the same category as AWS's Call Analytics SKUs. All six
        # arrived in bud-connect with no endpoint, because none of them is something a route
        # can serve.
        return "deepgram streaming pricing SKU or feature add-on"

    return None


def transform_model(original_key: str, model_data: dict, tz_provider: str) -> dict:
    """Transform a raw LiteLLM model entry for the unified catalog.

    Adds ``litellm_provider``, ``metadata.original_key``, and an
    optional ``license_id`` based on the TensorZero provider.
    """
    transformed = dict(model_data)
    transformed["litellm_provider"] = tz_provider
    transformed["metadata"] = {"original_key": original_key}
    if tz_provider in PROVIDER_LICENSE_MAP:
        transformed["license_id"] = PROVIDER_LICENSE_MAP[tz_provider]
    return transformed


class LiteLLMSource(BaseSource):
    """Fetches and transforms model data from the LiteLLM pricing JSON."""

    def __init__(self, config: CatalogConfig) -> None:
        super().__init__(config)

    async def fetch(self) -> FetchResult:
        """Download the LiteLLM JSON, filter to supported providers, and transform.

        Returns a :class:`FetchResult` whose ``data`` is a dict keyed by
        ``{tz_provider}/{original_key}``.
        """
        headers: dict[str, str] = {}
        if self._config.cache and self._last_etag:
            headers["If-None-Match"] = self._last_etag

        response = await self._fetch_url(self._config.litellm_url, headers, label="LiteLLM data")

        # ETag cache hit — return cached result
        if response.status_code == 304 and self._last_result is not None:
            logger.info("LiteLLM: 304 Not Modified, using cached result")
            return self._last_result

        try:
            litellm_data: dict = response.json()
        except ValueError as e:
            raise SourceFetchError(f"Invalid JSON from LiteLLM: {e}") from e

        if not isinstance(litellm_data, dict):
            raise SourceFetchError("LiteLLM response is not a JSON object")

        # Remove sample_spec if present
        litellm_data.pop("sample_spec", None)

        etag = response.headers.get("etag")
        fetched_at = datetime.now(timezone.utc)

        # Transform models
        result: dict[str, dict] = {}
        skipped = 0
        not_models: Counter[str] = Counter()

        for original_key, model_data in litellm_data.items():
            litellm_provider = model_data.get("litellm_provider")

            if not litellm_provider:
                skipped += 1
                continue

            if litellm_provider not in LITELLM_TO_TENSORZERO:
                skipped += 1
                continue

            tz_provider = LITELLM_TO_TENSORZERO[litellm_provider]

            reason = not_a_model(original_key, model_data)
            if reason:
                logger.debug("Skipping %s: %s", original_key, reason)
                not_models[reason] += 1
                skipped += 1
                continue

            if tz_provider not in TENSORZERO_PROVIDERS:
                logger.warning(
                    "Mapped provider %s not in TensorZero providers, skipping", tz_provider
                )
                skipped += 1
                continue

            # vertex_ai-language-models -> only Gemini models
            if (
                litellm_provider == "vertex_ai-language-models"
                and "gemini" not in original_key.lower()
            ):
                skipped += 1
                continue

            stripped_key = strip_provider_prefix(tz_provider, original_key)
            new_key = f"{tz_provider}/{stripped_key}"

            # Handle collision: two LiteLLM entries produce the same catalog key
            if new_key in result:
                prefix = STRIP_PREFIXES.get(tz_provider)
                if prefix:
                    existing_original = result[new_key]["metadata"]["original_key"]
                    # Keep the prefixed entry (canonical LiteLLM routing format)
                    if existing_original.startswith(prefix) and not original_key.startswith(prefix):
                        skipped += 1
                        continue
                logger.warning("Catalog key collision on '%s', overwriting", new_key)

            result[new_key] = transform_model(original_key, model_data, tz_provider)

        logger.info("LiteLLM: transformed %d models, skipped %d", len(result), skipped)
        if not_models:
            logger.info(
                "LiteLLM: excluded %d entries that are not models: %s",
                sum(not_models.values()),
                dict(not_models),
            )

        fetch_result = FetchResult(
            data=result,
            source_name="litellm",
            fetched_at=fetched_at,
            etag=etag,
        )

        # Update cache
        if etag:
            self._last_etag = etag
            self._last_result = fetch_result

        return fetch_result
