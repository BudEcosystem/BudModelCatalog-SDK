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

"""High-level client for fetching and merging the model catalog."""

from __future__ import annotations

import asyncio
import logging

from .config import CatalogConfig
from .merger import merge
from .models import CatalogResult
from .sources.ai_models import AiModelsSource
from .sources.base import BaseSource, FetchResult
from .sources.cloud_pricing import AwsPricingSource, AzurePricingSource, apply_pricing_overlay
from .sources.curated import CuratedSource
from .sources.litellm import LiteLLMSource

logger = logging.getLogger(__name__)


class CatalogClient:
    """Main entry point for fetching the unified model catalog.

    Fetches model data from LiteLLM and ai-models concurrently, merges
    the results, and returns a :class:`CatalogResult`.  Reuse a single
    instance to benefit from ETag-based HTTP caching across calls.
    """

    def __init__(self, config: CatalogConfig | None = None) -> None:
        self._config = config or CatalogConfig()
        self._litellm_source = LiteLLMSource(self._config)
        self._ai_models_source = AiModelsSource(self._config)
        self._curated_source = CuratedSource()
        self._aws_pricing_source = AwsPricingSource(self._config)
        self._azure_pricing_source = AzurePricingSource(self._config)

    async def fetch_catalog(self) -> CatalogResult:
        """Fetch from both sources concurrently, merge, return result.

        LiteLLM is required (raises on failure).
        ai-models is best-effort (falls back to LiteLLM-only costs).
        """

        async def _safe_ai_models() -> FetchResult | None:
            try:
                return await self._ai_models_source.fetch()
            except Exception:
                logger.warning(
                    "ai-models fetch failed, falling back to LiteLLM-only costs", exc_info=True
                )
                return None

        async def _safe(source: BaseSource, label: str) -> FetchResult | None:
            try:
                return await source.fetch()
            except Exception:
                logger.warning("%s fetch failed; continuing without it", label, exc_info=True)
                return None

        litellm_result, ai_models_result, aws_result, azure_result = await asyncio.gather(
            self._litellm_source.fetch(),
            _safe_ai_models(),
            _safe(self._aws_pricing_source, "AWS pricing"),
            _safe(self._azure_pricing_source, "Azure pricing"),
        )

        result = merge(litellm_result, ai_models_result, self._config)
        result = self._overlay_curated(result)
        return self._overlay_cloud_pricing(result, aws_result, azure_result)

    @staticmethod
    def _overlay_cloud_pricing(
        result: CatalogResult,
        aws_result: FetchResult | None,
        azure_result: FetchResult | None,
    ) -> CatalogResult:
        """Attach vendor-published billing blocks to entries a feed already listed.

        Best-effort on purpose. These are the only self-refreshing speech prices in the
        catalog, but they are also two more network calls on a nightly sync whose failure
        mode is bud-connect retiring models. Losing a billing block is recoverable on the
        next run; losing the catalog is not.
        """
        enriched = 0
        for fetched in (aws_result, azure_result):
            if fetched is not None:
                enriched += apply_pricing_overlay(result.models, fetched.data)

        if enriched:
            logger.info("Attached %d authoritative cloud billing blocks", enriched)
        return result

    def _overlay_curated(self, result: CatalogResult) -> CatalogResult:
        """Add hand-curated voice prices for vendors no feed covers.

        Additive only: a key already present from LiteLLM wins and the curated entry is
        dropped with a warning. LiteLLM is refreshed upstream continuously while this file
        is refreshed by someone remembering to; letting the stale one silently overwrite
        the live one is the wrong default, and a collision means the curated entry has
        become redundant and should be deleted.
        """
        if not result.models:
            # An empty merge means the upstream fetch failed or returned nothing, and
            # bud-connect's seeder treats "not in this run" as "deactivate". Overlaying
            # here would turn a total outage into a catalog of five voice models that
            # looks like a successful sync, and every other model would be retired on the
            # strength of it. Fail visibly empty instead.
            logger.warning("Merge produced no models; skipping curated overlay")
            return result

        curated = self._curated_source.load()

        added = 0
        for key, entry in curated.data.items():
            if key in result.models:
                logger.warning(
                    "Curated price for %s is shadowed by a live feed entry; delete it from "
                    "curated_voice_pricing.yaml",
                    key,
                )
                continue
            result.models[key] = entry
            added += 1

        if added:
            result.stats.total_output = len(result.models)
            logger.info("Overlaid %d curated voice prices", added)

        return result

    def fetch_catalog_sync(self) -> CatalogResult:
        """Blocking wrapper — works in all environments including Jupyter.

        In async code, prefer ``await fetch_catalog()`` instead.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop — safe to use asyncio.run() directly
            return asyncio.run(self.fetch_catalog())

        # Inside a running event loop (Jupyter, async framework, etc.) —
        # run in a background thread where asyncio.run() can create its own loop.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, self.fetch_catalog())
            return future.result()
