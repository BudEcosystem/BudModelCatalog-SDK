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
from .scrapers.base import UNIT_TO_COST_FIELD
from .scrapers.validate import exceeds_drift
from .sources.ai_models import AiModelsSource
from .sources.base import BaseSource, FetchResult
from .sources.cloud_pricing import AwsPricingSource, AzurePricingSource, apply_pricing_overlay
from .sources.curated import CuratedSource
from .sources.litellm import LiteLLMSource
from .sources.scraped import ScrapedPricingSource

logger = logging.getLogger(__name__)

#: The cost fields a scraped entry can carry, so the overlay can spot a unit change.
COST_FIELDS = tuple(UNIT_TO_COST_FIELD.values())


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
        self._scraped_source = ScrapedPricingSource(self._config)

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

        (
            litellm_result,
            ai_models_result,
            aws_result,
            azure_result,
            scraped_result,
        ) = await asyncio.gather(
            self._litellm_source.fetch(),
            _safe_ai_models(),
            _safe(self._aws_pricing_source, "AWS pricing"),
            _safe(self._azure_pricing_source, "Azure pricing"),
            _safe(self._scraped_source, "scraped pricing"),
        )

        # Precedence, weakest first. Curated is the committed floor; a scraped page is
        # fresher than that floor so it wins; a vendor's own price API beats any page, so
        # the cloud overlay goes last.
        result = merge(litellm_result, ai_models_result, self._config)
        result = self._overlay_curated(result)
        result = self._overlay_scraped(result, scraped_result)
        return self._overlay_cloud_pricing(result, aws_result, azure_result)

    @staticmethod
    def _overlay_scraped(result: CatalogResult, scraped: FetchResult | None) -> CatalogResult:
        """Apply freshly scraped vendor prices over the committed floor.

        Nobody reviews these between the page changing and bud-connect serving the number,
        so this method is where the last few guards live:

        * an entry already priced by a vendor's own API is never touched
        * a rate that has moved by more than :data:`MAX_DRIFT_FACTOR` from the floor is
          rejected as an extraction bug, because vendors reprice by factors of two, not of
          ten, while a parser reading per-hour as per-second is wrong by 3600
        * a disagreement is always logged with both numbers, since the likeliest cause is a
          parse error rather than a repricing

        A rejected price leaves the curated floor in place, which is why scraping can only
        ever make the catalog staler, never wrong.
        """
        if scraped is None or not scraped.data:
            return result
        if not result.models:
            # Same reasoning as the curated overlay: an empty merge means upstream failed,
            # and bud-connect reads "absent from this run" as "deactivate". Publishing nine
            # scraped voice models over an empty catalog would look like a successful sync
            # and retire everything else.
            logger.warning("Merge produced no models; skipping scraped overlay")
            return result

        added = updated = 0
        for key, entry in scraped.data.items():
            field = next((f for f in COST_FIELDS if f in entry), None)
            if field is None:  # pragma: no cover - the source always sets one
                continue
            new_rate = entry[field]

            existing = result.models.get(key)
            if existing is None:
                result.models[key] = entry
                added += 1
                continue

            existing_billing = existing.get("billing") or {}
            if existing_billing.get("confidence") == "authoritative":
                # The vendor publishes a price feed for this model; a marketing page does
                # not get to override it.
                continue

            other = next((f for f in COST_FIELDS if f != field), None)
            if other in existing and not existing_billing:
                # A live feed priced this model on a different basis and said nothing about
                # billing. Overwriting would leave two cost fields disagreeing about what
                # the request costs, so the mapping needs a human, not a refresh.
                logger.warning(
                    "Scraped %s in %s but the existing entry is priced per %s with no "
                    "billing block; leaving it alone",
                    key,
                    field,
                    other,
                )
                continue

            old_rate = existing.get(field)
            if isinstance(old_rate, (int, float)) and exceeds_drift(new_rate, float(old_rate)):
                logger.error(
                    "Rejecting scraped rate for %s: %.6g -> %.6g is a factor of %.0f, which "
                    "is an extraction bug rather than a repricing. Keeping the committed "
                    "rate; check the unit conversion in the adapter.",
                    key,
                    old_rate,
                    new_rate,
                    max(new_rate / old_rate, old_rate / new_rate) if old_rate else 0,
                )
                continue

            if isinstance(old_rate, (int, float)) and abs(float(old_rate) - new_rate) > 1e-12:
                logger.info(
                    "Scraped price change for %s: %.6g -> %.6g per %s",
                    key,
                    old_rate,
                    new_rate,
                    entry["billing"]["unit"],
                )

            if other in existing:
                # The unit itself changed, which only the billing block can express.
                del existing[other]
            existing[field] = new_rate
            existing["billing"] = entry["billing"]
            existing.setdefault("metadata", {}).update(entry.get("metadata") or {})
            updated += 1

        if added or updated:
            result.stats.total_output = len(result.models)
            logger.info("Scraped pricing: %d entries added, %d refreshed", added, updated)
        return result

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
