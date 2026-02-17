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
from datetime import datetime, timezone

from ..config import CatalogConfig
from ..exceptions import SourceFetchError
from .base import BaseSource, FetchResult

logger = logging.getLogger(__name__)

# LiteLLM provider -> TensorZero provider
LITELLM_TO_TENSORZERO = {
    "anthropic": "anthropic",
    "azure": "azure",
    "bedrock": "bedrock",
    "bedrock_converse": "bedrock",
    "deepseek": "deepseek",
    "fireworks_ai": "fireworks_ai-embedding-models",
    "fireworks_ai-embedding-models": "fireworks_ai-embedding-models",
    "gemini": "gemini",
    "hyperbolic": "hyperbolicai",
    "mistral": "mistral",
    "openai": "openai",
    "sagemaker": "sagemaker",
    "together_ai": "together_ai",
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
    "openai": "openai-api",
    "vertex_ai-anthropic_models": "Anthropic",
    "vertex_ai-gemini-models": "Google",
    "xai": "xAI",
}

# Valid TensorZero provider keys (from tensorzero_providers.json)
TENSORZERO_PROVIDERS: frozenset[str] = frozenset(
    {
        "together_ai",
        "vertex_ai-gemini-models",
        "vertex_ai-anthropic_models",
        "gemini",
        "deepseek",
        "mistral",
        "bedrock",
        "fireworks_ai-embedding-models",
        "azure",
        "openai",
        "sagemaker",
        "xai",
        "anthropic",
        "hyperbolicai",
        "huggingface",
        "bud_sentinel",
        "azure_content_safety",
    }
)


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

        for original_key, model_data in litellm_data.items():
            litellm_provider = model_data.get("litellm_provider")

            if not litellm_provider:
                skipped += 1
                continue

            if litellm_provider not in LITELLM_TO_TENSORZERO:
                skipped += 1
                continue

            tz_provider = LITELLM_TO_TENSORZERO[litellm_provider]

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

            new_key = f"{tz_provider}/{original_key}"
            result[new_key] = transform_model(original_key, model_data, tz_provider)

        logger.info("LiteLLM: transformed %d models, skipped %d", len(result), skipped)

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
