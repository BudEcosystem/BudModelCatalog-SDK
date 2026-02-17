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
        """Sync wrapper for fetch_catalog.

        This is a blocking call. In async code, await fetch_catalog() instead.
        """
        return asyncio.run(self.fetch_catalog())
