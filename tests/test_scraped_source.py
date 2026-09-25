"""The driver and the overlay: every way a scrape can fail, and what happens then.

The invariant under test throughout is that a broken scrape makes the catalog *staler*,
never wrong and never emptier. bud-connect re-seeds every 24h with nobody reviewing the
result, and its seeder reads "absent from this run" as "deactivate", so a partial or wrong
scrape is more damaging than no scrape at all.
"""

import asyncio
import pathlib
from datetime import datetime, timezone

import httpx
import pytest
import respx

from bud_model_catalog.client import CatalogClient
from bud_model_catalog.config import CatalogConfig
from bud_model_catalog.models import CatalogResult, MergeStats
from bud_model_catalog.scrapers.base import ScrapedModel, VendorScraper
from bud_model_catalog.sources.base import FetchResult
from bud_model_catalog.sources.scraped import ScrapedPricingSource

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "scrape"
GOOD_URL = "https://vendor.test/pricing"
OTHER_URL = "https://other.test/pricing"


class GoodScraper(VendorScraper):
    vendor = "goodvendor"
    url = GOOD_URL
    min_models = 1

    def extract(self, html: str) -> list[ScrapedModel]:  # noqa: ARG002
        return [
            ScrapedModel(
                model="fast",
                mode="audio_transcription",
                unit="second",
                rate=1.25e-04,
                published_as="$0.45/hr",
            )
        ]


class OtherGoodScraper(GoodScraper):
    vendor = "othervendor"
    url = OTHER_URL


class RaisingScraper(GoodScraper):
    vendor = "raisingvendor"
    url = OTHER_URL

    def extract(self, html: str) -> list[ScrapedModel]:  # noqa: ARG002
        raise ValueError("page format changed")


class EmptyScraper(GoodScraper):
    vendor = "emptyvendor"
    url = OTHER_URL

    def extract(self, html: str) -> list[ScrapedModel]:  # noqa: ARG002
        return []


class BadRateScraper(GoodScraper):
    vendor = "badratevendor"
    url = OTHER_URL

    def extract(self, html: str) -> list[ScrapedModel]:  # noqa: ARG002
        return [
            ScrapedModel("ok", "audio_transcription", "second", 1.25e-04, "$0.45/hr"),
            # per-hour left unconverted
            ScrapedModel("broken", "audio_transcription", "second", 0.45, "$0.45/hr"),
        ]


async def run(*scrapers, **kw) -> FetchResult:
    return await ScrapedPricingSource(CatalogConfig(), list(scrapers), **kw).fetch()


# --------------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_healthy_vendor_produces_a_catalog_entry():
    with respx.mock:
        respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        result = await run(GoodScraper())
    assert set(result.data) == {"goodvendor/fast"}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [301, 400, 404, 429, 500, 503])
async def test_a_non_200_contributes_nothing(status):
    """A moved pricing page is exactly when an adapter would otherwise parse an error
    page and "succeed" with nothing."""
    with respx.mock:
        respx.get(GOOD_URL).mock(return_value=httpx.Response(status, text="<html/>"))
        result = await run(GoodScraper())
    assert result.data == {}


@pytest.mark.asyncio
async def test_a_connection_failure_contributes_nothing():
    with respx.mock:
        respx.get(GOOD_URL).mock(side_effect=httpx.ConnectError("no route"))
        result = await run(GoodScraper())
    assert result.data == {}


@pytest.mark.asyncio
async def test_a_read_timeout_contributes_nothing():
    with respx.mock:
        respx.get(GOOD_URL).mock(side_effect=httpx.ReadTimeout("slow"))
        result = await run(GoodScraper())
    assert result.data == {}


@pytest.mark.asyncio
async def test_one_vendor_failing_does_not_affect_another():
    """Isolation is the whole point of running these per-vendor."""
    with respx.mock:
        respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        respx.get(OTHER_URL).mock(return_value=httpx.Response(500))
        result = await run(GoodScraper(), RaisingScraper())
    assert set(result.data) == {"goodvendor/fast"}


@pytest.mark.asyncio
async def test_an_adapter_that_raises_is_isolated():
    with respx.mock:
        respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        respx.get(OTHER_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        result = await run(GoodScraper(), RaisingScraper())
    assert set(result.data) == {"goodvendor/fast"}


@pytest.mark.asyncio
async def test_an_adapter_returning_nothing_is_a_failure_not_an_empty_vendor():
    with respx.mock:
        respx.get(OTHER_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        result = await run(EmptyScraper())
    assert result.data == {}


@pytest.mark.asyncio
async def test_a_single_bad_rate_is_dropped_and_its_siblings_kept():
    with respx.mock:
        respx.get(OTHER_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        result = await run(BadRateScraper())
    assert set(result.data) == {"badratevendor/ok"}


@pytest.mark.asyncio
async def test_a_slow_vendor_is_abandoned_without_holding_up_the_rest():
    """The source runs inside bud-connect's startup, before uvicorn serves, and that
    startup already runs close to its probe deadline on a fresh database."""

    async def slow(request):  # noqa: ARG001
        await asyncio.sleep(5)
        return httpx.Response(200, text="<html/>")

    with respx.mock:
        respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        respx.get(OTHER_URL).mock(side_effect=slow)
        result = await run(GoodScraper(), OtherGoodScraper(), total_budget=0.5)
    assert set(result.data) == {"goodvendor/fast"}


@pytest.mark.asyncio
async def test_a_slow_vendor_gives_up_its_slot_at_the_vendor_deadline():
    """httpx's timeout bounds each read, not the request, so a server that dribbles bytes
    never trips it. Without a per-vendor deadline the slow vendor held its slot until the
    whole budget ran out, and every vendor queued behind it went down with it."""

    async def dribble(request):  # noqa: ARG001
        await asyncio.sleep(5)
        return httpx.Response(200, text="<html/>")

    with respx.mock:
        respx.get(OTHER_URL).mock(side_effect=dribble)
        respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        result = await run(
            OtherGoodScraper(), GoodScraper(), concurrency=1, vendor_timeout=0.3, total_budget=3
        )
    # The slow vendor went first and would have held the only slot for the full budget.
    assert set(result.data) == {"goodvendor/fast"}


@pytest.mark.asyncio
async def test_the_budget_is_enforced_even_when_every_vendor_hangs():
    async def slow(request):  # noqa: ARG001
        await asyncio.sleep(10)
        return httpx.Response(200, text="<html/>")

    with respx.mock:
        respx.get(GOOD_URL).mock(side_effect=slow)
        started = asyncio.get_event_loop().time()
        result = await run(GoodScraper(), total_budget=0.4)
        elapsed = asyncio.get_event_loop().time() - started
    assert result.data == {}
    assert elapsed < 4, f"budget not enforced; took {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_concurrency_is_bounded():
    """Eight vendors should not all be hit at the same instant every 24 hours."""
    live = 0
    peak = 0

    async def counted(request):  # noqa: ARG001
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.05)
        live -= 1
        return httpx.Response(200, text="<html/>")

    scrapers = []
    for i in range(6):
        cls = type(f"S{i}", (GoodScraper,), {"vendor": f"v{i}", "url": f"https://v{i}.test/p"})
        scrapers.append(cls())
    with respx.mock:
        for s in scrapers:
            respx.get(s.url).mock(side_effect=counted)
        await run(*scrapers, concurrency=2)
    assert peak <= 2, f"ran {peak} pages at once despite concurrency=2"


# --------------------------------------------------------------------------------- #
# the entry the source builds
# --------------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_entry_has_everything_a_consumer_needs():
    with respx.mock:
        respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        result = await run(GoodScraper())
    entry = result.data["goodvendor/fast"]
    assert entry["litellm_provider"] == "goodvendor"
    assert entry["mode"] == "audio_transcription"
    assert entry["input_cost_per_second"] == pytest.approx(1.25e-04)

    billing = entry["billing"]
    assert billing["unit"] == "second"
    assert billing["meter"] == "input_audio_seconds"
    assert billing["currency"] == "USD"
    assert billing["confidence"] == "curated"
    assert billing["source"]["url"] == GOOD_URL
    # A rate nobody can date is a rate nobody can re-check, and the curated loader
    # refuses entries without this.
    assert billing["source"]["checked_on"] == datetime.now(timezone.utc).date().isoformat()
    assert "$0.45/hr" in billing["source"]["note"]
    assert entry["metadata"]["scraped"] is True


@pytest.mark.asyncio
async def test_a_promotional_rate_is_recorded_in_the_note_not_the_rate():
    class PromoScraper(GoodScraper):
        def extract(self, html):  # noqa: ARG002
            return [
                ScrapedModel(
                    "fast",
                    "audio_transcription",
                    "second",
                    1.25e-04,
                    "$0.45/hr",
                    promotional_rate=6.67e-05,
                )
            ]

    with respx.mock:
        respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
        result = await run(PromoScraper())
    entry = result.data["goodvendor/fast"]
    assert entry["input_cost_per_second"] == pytest.approx(1.25e-04)
    assert "Promotional rate" in entry["billing"]["source"]["note"]


@pytest.mark.asyncio
async def test_real_adapters_run_against_their_fixtures_through_the_source():
    """End to end with the actual adapters, still without touching the network."""
    from bud_model_catalog.scrapers.vendors.cartesia import CartesiaScraper
    from bud_model_catalog.scrapers.vendors.speechmatics import SpeechmaticsScraper

    sm, ca = SpeechmaticsScraper(), CartesiaScraper()
    with respx.mock:
        respx.get(sm.url).mock(
            return_value=httpx.Response(200, text=(FIXTURES / "speechmatics.html").read_text())
        )
        respx.get(ca.url).mock(
            return_value=httpx.Response(200, text=(FIXTURES / "cartesia.html").read_text())
        )
        result = await run(sm, ca)
    assert len(result.data) == 3 + 4
    assert result.data["cartesia/ink-2"]["billing"]["confidence"] == "derived"
    assert result.data["speechmatics/standard"]["input_cost_per_second"] == pytest.approx(
        0.45 / 3600
    )


# --------------------------------------------------------------------------------- #
# the overlay
# --------------------------------------------------------------------------------- #


def catalog(models: dict) -> CatalogResult:
    return CatalogResult(
        models=models,
        stats=MergeStats(
            total_litellm=len(models),
            total_output=len(models),
            matched=0,
            unmatched=0,
            deprecated_removed=0,
            cost_fields_updated=0,
        ),
        litellm_fetched_at=datetime.now(timezone.utc),
    )


def scraped(
    key: str, rate: float, *, unit: str = "second", confidence: str = "curated"
) -> FetchResult:
    field = "input_cost_per_second" if unit == "second" else "input_cost_per_character"
    return FetchResult(
        data={
            key: {
                "litellm_provider": key.split("/")[0],
                "mode": "audio_transcription",
                field: rate,
                "billing": {
                    "unit": unit,
                    "meter": "input_audio_seconds",
                    "currency": "USD",
                    "confidence": confidence,
                    "source": {"url": "u", "checked_on": "2026-09-24", "note": "n"},
                },
                "metadata": {"original_key": key.split("/")[1], "scraped": True},
            }
        },
        source_name="scraped",
        fetched_at=datetime.now(timezone.utc),
    )


def test_a_model_nothing_else_lists_is_not_published(caplog):
    """A page refreshes prices; it does not introduce models.

    Such a key has no committed rate to check drift against, so a mis-parse would publish
    unchecked -- reproduced as a renamed row, "Linden 1.1" at 10x, landing beside the stale
    "Linden 1" as a second model. New models enter through the curated file.
    """
    result = CatalogClient._overlay_scraped(
        catalog({"other/model": {"input_cost_per_second": 1e-05}}),
        scraped("speechmatics/linden-1.1", 8.3e-04),
    )
    assert "speechmatics/linden-1.1" not in result.models
    assert result.stats.total_output == 1
    assert "not published" in caplog.text


def test_the_scraped_billing_block_is_merged_not_replaced():
    """The floor carries rules no page states parseably; a refresh must not drop them.

    Rev AI's `rounding_increment` lived only in the floor, and replacing the whole block
    on every scrape deleted it from all three Rev AI models.
    """
    base = catalog(
        {
            "revai/reverb": {
                "input_cost_per_second": 5.5e-05,
                "billing": {
                    "unit": "second",
                    "confidence": "curated",
                    "min_billable_units": 15,
                    "rounding_increment": 1,
                },
            }
        }
    )
    result = CatalogClient._overlay_scraped(base, scraped("revai/reverb", 5.6e-05))
    billing = result.models["revai/reverb"]["billing"]
    assert billing["rounding_increment"] == 1
    assert billing["min_billable_units"] == 15
    assert billing["source"]["url"] == "u"  # the scrape's provenance does replace the floor's


def test_a_fresh_price_replaces_the_committed_floor():
    """The decision here was explicit: no review, take the new price."""
    base = catalog(
        {
            "speechmatics/linden-1": {
                "input_cost_per_second": 8.3e-05,
                "billing": {"unit": "second", "confidence": "curated"},
            }
        }
    )
    result = CatalogClient._overlay_scraped(base, scraped("speechmatics/linden-1", 1.2e-04))
    entry = result.models["speechmatics/linden-1"]
    assert entry["input_cost_per_second"] == pytest.approx(1.2e-04)
    assert entry["billing"]["source"]["url"] == "u"


def test_an_authoritative_price_is_never_overwritten_by_a_page():
    """AWS and Azure publish price APIs. A marketing page does not outrank one."""
    base = catalog(
        {
            "aws_polly/neural": {
                "input_cost_per_character": 1.6e-05,
                "billing": {"unit": "character", "confidence": "authoritative"},
            }
        }
    )
    result = CatalogClient._overlay_scraped(
        base, scraped("aws_polly/neural", 9.9e-05, unit="character")
    )
    assert result.models["aws_polly/neural"]["input_cost_per_character"] == pytest.approx(1.6e-05)
    assert result.models["aws_polly/neural"]["billing"]["confidence"] == "authoritative"


def test_a_rate_that_moved_by_orders_of_magnitude_is_rejected():
    """3600x is a unit bug, not a repricing, and the floor is the safer value to keep."""
    base = catalog(
        {
            "speechmatics/linden-1": {
                "input_cost_per_second": 8.3e-05,
                "billing": {"unit": "second", "confidence": "curated"},
            }
        }
    )
    result = CatalogClient._overlay_scraped(base, scraped("speechmatics/linden-1", 0.30))
    assert result.models["speechmatics/linden-1"]["input_cost_per_second"] == pytest.approx(8.3e-05)


def test_a_plausible_price_change_is_accepted():
    base = catalog(
        {
            "speechmatics/linden-1": {
                "input_cost_per_second": 8.3e-05,
                "billing": {"unit": "second", "confidence": "curated"},
            }
        }
    )
    result = CatalogClient._overlay_scraped(base, scraped("speechmatics/linden-1", 1.66e-04))
    assert result.models["speechmatics/linden-1"]["input_cost_per_second"] == pytest.approx(
        1.66e-04
    )


def test_an_empty_catalog_is_never_overlaid():
    """An empty merge means upstream failed. Nine voice models over an empty catalog looks
    like a successful sync and would retire everything else."""
    result = CatalogClient._overlay_scraped(catalog({}), scraped("speechmatics/linden-1", 8.3e-05))
    assert result.models == {}


@pytest.mark.parametrize(
    "payload",
    [None, FetchResult(data={}, source_name="scraped", fetched_at=datetime.now(timezone.utc))],
)
def test_nothing_scraped_is_a_no_op(payload):
    base = catalog({"a/b": {"input_cost_per_second": 1e-05}})
    result = CatalogClient._overlay_scraped(base, payload)
    assert result.models == {"a/b": {"input_cost_per_second": 1e-05}}


def test_a_feed_priced_entry_on_a_different_basis_is_left_alone():
    """Writing here would leave two cost fields disagreeing about what a request costs.

    That is a mapping question for a human, not something a refresh should decide.
    """
    base = catalog({"elevenlabs/x": {"input_cost_per_character": 3e-05}})
    result = CatalogClient._overlay_scraped(base, scraped("elevenlabs/x", 1e-04, unit="second"))
    entry = result.models["elevenlabs/x"]
    assert entry["input_cost_per_character"] == pytest.approx(3e-05)
    assert "input_cost_per_second" not in entry


def test_a_vendor_switching_billing_unit_replaces_the_stale_field():
    """When the vendor itself changes basis, leaving the old field behind would keep a
    rate in a unit nothing declares any more."""
    base = catalog(
        {
            "v/m": {
                "input_cost_per_character": 3e-05,
                "billing": {"unit": "character", "confidence": "curated"},
            }
        }
    )
    result = CatalogClient._overlay_scraped(base, scraped("v/m", 1e-04, unit="second"))
    entry = result.models["v/m"]
    assert entry["input_cost_per_second"] == pytest.approx(1e-04)
    assert "input_cost_per_character" not in entry
    assert entry["billing"]["unit"] == "second"


def test_a_minimum_charge_survives_into_the_entry():
    """Regression for a data-loss bug found while writing the Rev AI adapter.

    The overlay replaces an entry's whole billing block, so the first version of this
    source would have scraped Rev AI's rate and silently dropped `min_billable_units: 15`
    from the committed entry -- leaving a rate that under-bills every clip shorter than
    15 seconds, with nothing in the diff to show it had happened.
    """

    class MinimumScraper(GoodScraper):
        def extract(self, html):  # noqa: ARG002
            return [
                ScrapedModel(
                    "reverb",
                    "audio_transcription",
                    "second",
                    5.6e-05,
                    "$0.20 per hour",
                    min_billable_units=15.0,
                )
            ]

    async def go():
        with respx.mock:
            respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
            return await run(MinimumScraper())

    result = asyncio.run(go())
    assert result.data["goodvendor/reverb"]["billing"]["min_billable_units"] == 15.0


def test_no_minimum_means_the_key_is_absent_rather_than_null():
    """A null minimum reads as "charges from zero", which is a different claim from
    "the vendor does not publish one"."""

    async def go():
        with respx.mock:
            respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
            return await run(GoodScraper())

    result = asyncio.run(go())
    assert "min_billable_units" not in result.data["goodvendor/fast"]["billing"]


def test_two_scrapers_claiming_one_key_do_not_silently_overwrite():
    """`validate_batch` catches duplicates inside one extraction, not across two.

    This is how Google would look if its text-to-speech and speech-to-text pages were both
    registered and both produced a model called "standard": whichever ran second would
    quietly win, and the losing rate would never appear anywhere. The first is kept and the
    clash is logged as the registry bug it is.
    """

    class First(GoodScraper):
        vendor = "twinvendor"
        url = GOOD_URL

        def extract(self, html):  # noqa: ARG002
            return [ScrapedModel("standard", "audio_transcription", "second", 1.0e-04, "a")]

    class Second(GoodScraper):
        vendor = "twinvendor"
        url = OTHER_URL

        def extract(self, html):  # noqa: ARG002
            return [ScrapedModel("standard", "audio_transcription", "second", 9.0e-04, "b")]

    async def go():
        with respx.mock:
            respx.get(GOOD_URL).mock(return_value=httpx.Response(200, text="<html/>"))
            respx.get(OTHER_URL).mock(return_value=httpx.Response(200, text="<html/>"))
            return await run(First(), Second())

    result = asyncio.run(go())
    assert list(result.data) == ["twinvendor/standard"]
    assert result.data["twinvendor/standard"]["input_cost_per_second"] == pytest.approx(1.0e-04)
