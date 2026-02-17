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
from .sources.base import FetchResult
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

        litellm_result, ai_models_result = await asyncio.gather(
            self._litellm_source.fetch(),
            _safe_ai_models(),
        )

        return merge(litellm_result, ai_models_result, self._config)

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
