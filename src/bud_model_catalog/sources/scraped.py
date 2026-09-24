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


"""Drives the vendor page adapters and turns what they find into catalog entries.

This is the only self-refreshing price source for vendors that publish nothing but a web
page. bud-connect re-seeds every 24h (`cron-tensorzero-sync`), so a price change on a
vendor's page reaches the catalog within a day with nobody in the loop.

That absence of review is what shapes the failure handling here. Every way this can go
wrong ends in *the vendor contributing nothing*, never in a wrong number and never in a
missing one:

* page unreachable, slow, or non-200  -> vendor skipped
* adapter raises                      -> vendor skipped, others unaffected
* adapter returns fewer models than its floor -> nothing taken from that vendor
* a rate fails a sanity gate          -> that model dropped, the rest kept
* the whole source overruns its budget -> whatever finished is used

In every case the committed price in `curated_voice_pricing.yaml` stands, so a broken
scrape degrades to a slightly stale price rather than to no price or a wrong one. The
comment on the cloud overlay applies here word for word: losing a billing block is
recoverable on the next run; losing the catalog is not.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from ..config import CatalogConfig
from ..scrapers.base import UNIT_TO_COST_FIELD, UNIT_TO_METER, ScrapeOutcome, VendorScraper
from ..scrapers.registry import all_scrapers
from ..scrapers.validate import validate_batch
from .base import BaseSource, FetchResult

logger = logging.getLogger(__name__)

#: We read public pricing pages as a reader would, and say who we are while doing it.
#: Verified that every page in the registry serves identical content to this agent, so
#: there is no reason to impersonate a browser.
USER_AGENT = "bud-model-catalog/1.0 (+https://github.com/BudEcosystem/BudModelCatalog-SDK)"

#: Per-vendor ceiling. One slow marketing site must not hold up a catalog sync, and a
#: single attempt is deliberate: these pages are read every 24h, so a retry buys very
#: little and costs the budget below.
DEFAULT_VENDOR_TIMEOUT = 10.0

#: Ceiling for the whole source. bud-connect seeds inside the FastAPI lifespan, before
#: uvicorn serves, and its startup probe already runs close to the limit on a fresh
#: database -- so this source is capped rather than merely timed out per vendor.
DEFAULT_TOTAL_BUDGET = 20.0

#: Pages are fetched a few at a time: enough to stay inside the budget, few enough to not
#: arrive at eight vendors simultaneously every 24 hours.
DEFAULT_CONCURRENCY = 4


class ScrapedPricingSource(BaseSource):
    """Fetches each registered vendor's pricing page and validates what comes back."""

    def __init__(
        self,
        config: CatalogConfig,
        scrapers: list[VendorScraper] | None = None,
        *,
        vendor_timeout: float = DEFAULT_VENDOR_TIMEOUT,
        total_budget: float = DEFAULT_TOTAL_BUDGET,
        concurrency: int = DEFAULT_CONCURRENCY,
    ) -> None:
        super().__init__(config)
        self._scrapers = scrapers if scrapers is not None else all_scrapers()
        self._vendor_timeout = vendor_timeout
        self._total_budget = total_budget
        self._concurrency = max(1, concurrency)

    async def fetch(self) -> FetchResult:
        """Scrape every registered vendor and return validated catalog entries."""
        outcomes = await self._gather()

        data: dict[str, dict[str, Any]] = {}
        today = datetime.now(timezone.utc).date().isoformat()
        for outcome in outcomes:
            for m in outcome.models:
                data[f"{outcome.vendor}/{m.model}"] = _to_entry(outcome, m, today)

        ok = [o.vendor for o in outcomes if o.ok]
        failed = {o.vendor: o.error for o in outcomes if o.error}
        rejected = sum(len(o.rejected) for o in outcomes)
        logger.info(
            "Scraped %d prices from %d/%d vendors (%d rejected)%s",
            len(data),
            len(ok),
            len(self._scrapers),
            rejected,
            f"; failed: {failed}" if failed else "",
        )
        for o in outcomes:
            for reason in o.rejected:
                logger.warning("Rejected scraped rate: %s", reason)

        return FetchResult(data=data, source_name="scraped", fetched_at=datetime.now(timezone.utc))

    async def _gather(self) -> list[ScrapeOutcome]:
        """Run every adapter under a shared time budget, isolating each one."""
        semaphore = asyncio.Semaphore(self._concurrency)

        async with httpx.AsyncClient(
            timeout=self._vendor_timeout,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:

            async def run(scraper: VendorScraper) -> ScrapeOutcome:
                async with semaphore:
                    return await self._scrape_one(client, scraper)

            tasks = [asyncio.create_task(run(s)) for s in self._scrapers]
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=self._total_budget
                )
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning(
                    "Scraped pricing exceeded its %.0fs budget; using whatever finished",
                    self._total_budget,
                )

            outcomes: list[ScrapeOutcome] = []
            for scraper, task in zip(self._scrapers, tasks, strict=True):
                if not task.done():
                    # Cancel the stragglers so they cannot outlive the fetch and log into
                    # a request that has already returned.
                    task.cancel()
                    outcomes.append(
                        ScrapeOutcome(vendor=scraper.vendor, url=scraper.url, error="timed out")
                    )
                    continue
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    outcomes.append(
                        ScrapeOutcome(vendor=scraper.vendor, url=scraper.url, error="cancelled")
                    )
                except Exception as e:  # pragma: no cover - defence in depth
                    outcomes.append(
                        ScrapeOutcome(vendor=scraper.vendor, url=scraper.url, error=repr(e))
                    )
                else:
                    outcomes.append(
                        result
                        if isinstance(result, ScrapeOutcome)
                        else ScrapeOutcome(
                            vendor=scraper.vendor, url=scraper.url, error=repr(result)
                        )
                    )
            return outcomes

    async def _scrape_one(self, client: httpx.AsyncClient, scraper: VendorScraper) -> ScrapeOutcome:
        """Fetch and parse one vendor, converting every failure into an outcome."""
        try:
            response = await client.get(scraper.url)
        except Exception as e:
            return ScrapeOutcome(vendor=scraper.vendor, url=scraper.url, error=f"fetch failed: {e}")

        if response.status_code != 200:
            # A 404 on a pricing page usually means the page moved, which is exactly when
            # an adapter would otherwise "successfully" extract nothing from an error page.
            return ScrapeOutcome(
                vendor=scraper.vendor, url=scraper.url, error=f"HTTP {response.status_code}"
            )

        try:
            extracted = scraper.extract(response.text)
        except Exception as e:
            return ScrapeOutcome(
                vendor=scraper.vendor, url=scraper.url, error=f"extract failed: {e}"
            )

        accepted, rejected = validate_batch(scraper.vendor, extracted, scraper.min_models)
        return ScrapeOutcome(
            vendor=scraper.vendor, url=scraper.url, models=accepted, rejected=rejected
        )


def _to_entry(outcome: ScrapeOutcome, m: Any, checked_on: str) -> dict[str, Any]:
    """Build a catalog entry in the same shape the curated file produces.

    Keeping the shapes identical is what lets the overlay treat a scraped price and a
    hand-curated one the same way, and what makes `confidence` the only thing that
    distinguishes them to a consumer.
    """
    note = m.note or ""
    if m.promotional_rate is not None:
        note = (
            f"{note} Promotional rate at time of scrape: {m.promotional_rate:.6g}/{m.unit}.".strip()
        )
    billing: dict[str, Any] = {
        "unit": m.unit,
        "meter": UNIT_TO_METER[m.unit],
        "currency": "USD",
        "confidence": m.confidence,
    }
    if m.min_billable_units is not None:
        # Kept next to the rate rather than buried in the note: a consumer that does not
        # apply it under-bills every request shorter than the minimum.
        billing["min_billable_units"] = m.min_billable_units
    return {
        "litellm_provider": outcome.vendor,
        "mode": m.mode,
        UNIT_TO_COST_FIELD[m.unit]: m.rate,
        "billing": {
            **billing,
            "source": {
                "url": outcome.url,
                "checked_on": checked_on,
                "note": f"Scraped. Published as {m.published_as}. {note}".strip(),
            },
        },
        "metadata": {"original_key": m.model, "scraped": True},
    }
