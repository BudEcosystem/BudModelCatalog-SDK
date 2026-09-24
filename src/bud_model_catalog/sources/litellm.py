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
    # Vertex files Gemini under three LiteLLM providers, not one: the language models, the
    # embedding models, and a bare `vertex_ai` that carries the newest entries (Gemini TTS,
    # transcription, Live, gemini-3-pro-preview). Mapping only the first left those out of
    # the catalog entirely. All three are narrowed to Gemini by VERTEX_GEMINI_ONLY below.
    "vertex_ai": "vertex_ai-gemini-models",
    "vertex_ai-anthropic_models": "vertex_ai-anthropic_models",
    "vertex_ai-embedding-models": "vertex_ai-gemini-models",
    "vertex_ai-language-models": "vertex_ai-gemini-models",
    "xai": "xai",
}

#: LiteLLM providers whose entries reach `vertex_ai-gemini-models` only when they are Gemini.
#: The same providers also list Vertex-hosted Grok, Mistral OCR, Imagen, Chirp and Lyria,
#: which the provider name `vertex_ai-gemini-models` does not describe.
VERTEX_GEMINI_ONLY: frozenset[str] = frozenset(
    {"vertex_ai", "vertex_ai-embedding-models", "vertex_ai-language-models"}
)

#: Models a vendor has withdrawn from its API that LiteLLM still lists. Publishing one puts a
#: model in budapp that fails on every request, so each is named here with the evidence.
RETIRED_BY_VENDOR: dict[str, str] = {
    # "scribe_v1 -> Replace with scribe_v2" (elevenlabs.io/docs/models, 2026-09-24); WaaV
    # records the removal date as 2026-07-09 and refuses to default to it.
    "elevenlabs/scribe_v1": "ElevenLabs replaced scribe_v1 with scribe_v2",
    "elevenlabs/scribe_v1_experimental": "ElevenLabs replaced scribe_v1 with scribe_v2",
    # AssemblyAI's transcript API now takes `speech_models` from {universal-3-5-pro,
    # universal-2}; `speech_model` is deprecated and neither `best` nor `nano` is accepted
    # (assemblyai.com/docs/api-reference/transcripts/submit, 2026-09-24).
    "assemblyai/best": "AssemblyAI replaced best/nano with the Universal models",
    "assemblyai/nano": "AssemblyAI replaced best/nano with the Universal models",
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

    if original_key in RETIRED_BY_VENDOR:
        return f"retired by the vendor: {RETIRED_BY_VENDOR[original_key]}"

    if "code_interpreter_cost_per_session" in model_data and not any(
        "cost_per_token" in field for field in model_data
    ):
        # `azure/container` and `openai/container` price the Code Interpreter sandbox by the
        # session. They carry mode=chat, so without this they reached budapp as chat models,
        # with no token price and no endpoint that could serve them.
        return "code-interpreter container pricing, not a model"

    if model_data.get("litellm_provider") == "together_ai" and original_key.startswith(
        "together-ai-"
    ):
        # `together-ai-4.1b-8b`, `together-ai-embedding-up-to-150m` and friends are Together's
        # price-by-parameter-count buckets. Real Together models are `org/model`; these are
        # the rows used to price one, and unlike Fireworks' buckets they carry a mode, so the
        # no-mode rule above does not catch them.
        return "size-bucket pricing tier, not a model"

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


def _is_canonical(original_key: str, tz_provider: str) -> bool:
    """Whether a LiteLLM key is in the provider-prefixed form LiteLLM routes on.

    For a provider with no strip prefix every key is equally canonical, and the entry seen
    first wins -- which, since the merge only fills gaps, decides nothing that matters.
    """
    prefix = STRIP_PREFIXES.get(tz_provider)
    return prefix is not None and original_key.startswith(prefix)


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

            if litellm_provider in VERTEX_GEMINI_ONLY and "gemini" not in original_key.lower():
                skipped += 1
                continue

            stripped_key = strip_provider_prefix(tz_provider, original_key)
            new_key = f"{tz_provider}/{stripped_key}"

            transformed = transform_model(original_key, model_data, tz_provider)

            # Two LiteLLM entries can produce the same catalog key: `gemini-3.8-live` and
            # `gemini/gemini-3.8-live`, say. They describe one model, and LiteLLM tends to
            # spread its facts across both -- one carries the prices, the other the
            # modalities or the search-grounding price. Keeping one wholesale threw the other
            # away, so they are merged: the canonical entry wins every field it has, and the
            # other only fills the gaps.
            if new_key in result:
                existing = result[new_key]
                winner, loser = (
                    (existing, transformed)
                    if _is_canonical(existing["metadata"]["original_key"], tz_provider)
                    or not _is_canonical(original_key, tz_provider)
                    else (transformed, existing)
                )
                merged = {**loser, **winner}
                merged["metadata"] = {
                    **winner["metadata"],
                    "merged_from": loser["metadata"]["original_key"],
                }
                logger.debug(
                    "Catalog key %s: merged %s into %s",
                    new_key,
                    loser["metadata"]["original_key"],
                    winner["metadata"]["original_key"],
                )
                result[new_key] = merged
                skipped += 1
                continue

            result[new_key] = transformed

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
