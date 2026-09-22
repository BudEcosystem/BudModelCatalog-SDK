"""Authoritative cloud speech pricing: AWS Price List and Azure Retail Prices.

The failure modes these guard against are all silent ones. A wrong SKU produces a
plausible number five times too big; a mangled OData filter produces HTTP 200 with an
empty body; an Azure "Free" meter produces a rate of zero. None of them raise.
"""

import httpx
import pytest
import respx

from bud_model_catalog.config import CatalogConfig
from bud_model_catalog.sources.cloud_pricing import (
    AWS_CANONICAL_REGION,
    AZURE_CANONICAL_REGION,
    AZURE_RETAIL_URL,
    AwsPricingSource,
    AzurePricingSource,
    _first_tier_rate,
    apply_pricing_overlay,
)

POLLY_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonPolly/current/index.json"
TRANSCRIBE_URL = (
    "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/transcribe/current/index.json"
)


def _aws_offer(products):
    """Build an offer file in the shape AWS publishes."""
    return {
        "publicationDate": "2026-09-11T12:44:59Z",
        "products": {sku: {"attributes": {"usagetype": ut}} for sku, (ut, _) in products.items()},
        "terms": {
            "OnDemand": {
                sku: {
                    f"{sku}.offer": {
                        "priceDimensions": {
                            f"{sku}.dim{i}": {
                                "beginRange": str(begin),
                                "pricePerUnit": {"USD": f"{price:.10f}"},
                                "unit": "Characters",
                            }
                            for i, (begin, price) in enumerate(tiers)
                        }
                    }
                }
                for sku, (_, tiers) in products.items()
            }
        },
    }


# --------------------------------------------------------------------------------- #
# tier selection
# --------------------------------------------------------------------------------- #


def test_the_first_tier_is_the_one_starting_at_zero():
    """List price, not the cheapest price.

    AWS lists tiers as unordered price dimensions. `min()` would bill everyone the
    highest-volume discount, which is a discount nobody has earned.
    """
    term = {
        "offer": {
            "priceDimensions": {
                "a": {"beginRange": "250000", "pricePerUnit": {"USD": "0.00031"}},
                "b": {"beginRange": "0", "pricePerUnit": {"USD": "0.0005"}},
                "c": {"beginRange": "5000000", "pricePerUnit": {"USD": "0.00019"}},
            }
        }
    }
    assert _first_tier_rate(term) == 0.0005


def test_a_term_with_no_usd_price_yields_nothing():
    assert _first_tier_rate({"offer": {"priceDimensions": {"a": {"beginRange": "0"}}}}) is None


# --------------------------------------------------------------------------------- #
# AWS
# --------------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_aws_picks_the_base_skus_and_ignores_the_variants():
    """`USE1-CallAnalyticsTranscribeAudio` is 5x the base rate and reads just as plausibly."""
    polly = _aws_offer(
        {
            "P1": ("USE1-SynthesizeSpeech-Characters", [(0, 0.000004)]),
            "P2": ("USE1-SynthesizeSpeechNeural-Characters", [(0, 0.000016)]),
            "P3": ("EUW1-SynthesizeSpeech-Characters", [(0, 0.0000045)]),  # wrong region
        }
    )
    transcribe = _aws_offer(
        {
            "T1": ("USE1-TranscribeAudio", [(0, 0.0001)]),
            "T2": ("USE1-CallAnalyticsTranscribeAudio", [(0, 0.0005)]),  # not the base operation
            "T3": ("USE1-RedactionTranscribeAudio", [(0, 0.00004)]),
        }
    )

    with respx.mock:
        respx.get(POLLY_URL).mock(return_value=httpx.Response(200, json=polly))
        respx.get(TRANSCRIBE_URL).mock(return_value=httpx.Response(200, json=transcribe))
        result = await AwsPricingSource(CatalogConfig()).fetch()

    assert set(result.data) == {
        "aws_polly/standard",
        "aws_polly/neural",
        "aws_transcribe/StartTranscriptionJob",
    }
    assert result.data["aws_transcribe/StartTranscriptionJob"]["rate"] == 0.0001
    assert result.data["aws_polly/standard"]["rate"] == 0.000004


@pytest.mark.asyncio
async def test_aws_records_region_and_the_vendors_publication_date():
    """Provenance is what makes an authoritative rate checkable later."""
    with respx.mock:
        respx.get(POLLY_URL).mock(
            return_value=httpx.Response(
                200, json=_aws_offer({"P1": ("USE1-SynthesizeSpeech-Characters", [(0, 0.000004)])})
            )
        )
        respx.get(TRANSCRIBE_URL).mock(return_value=httpx.Response(200, json=_aws_offer({})))
        result = await AwsPricingSource(CatalogConfig()).fetch()

    billing = result.data["aws_polly/standard"]["billing"]
    assert billing["confidence"] == "authoritative"
    assert billing["region"] == AWS_CANONICAL_REGION
    assert billing["source"]["published"] == "2026-09-11T12:44:59Z"
    assert billing["unit"] == "character"
    assert billing["meter"] == "input_characters"


# --------------------------------------------------------------------------------- #
# Azure
# --------------------------------------------------------------------------------- #


def _azure_payload(items):
    return {"Items": items, "NextPageLink": None}


@pytest.mark.asyncio
async def test_the_azure_filter_percent_encodes_the_plus():
    """The service family is literally "AI + Machine Learning".

    A raw `+` in a query string is a SPACE to the server, so the filter becomes
    "AI   Machine Learning", matches nothing, and returns 200 with an empty Items array.
    A success that resolves no meters, which no exception handler will ever notice.
    """
    captured = {}

    def _capture(request):
        captured["url"] = str(request.url)
        return httpx.Response(200, json=_azure_payload([]))

    with respx.mock:
        respx.get(url__startswith=AZURE_RETAIL_URL).mock(side_effect=_capture)
        await AzurePricingSource(CatalogConfig()).fetch()

    assert "%2B" in captured["url"], f"the + was not encoded: {captured['url']}"
    assert "AI%20%2B%20Machine%20Learning" in captured["url"]


@pytest.mark.asyncio
async def test_azure_normalises_per_hour_and_per_million_to_the_stored_unit():
    """Azure quotes STT per hour and synthesis per 1M characters."""
    with respx.mock:
        respx.get(url__startswith=AZURE_RETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_azure_payload(
                    [
                        {
                            "meterName": "S1 Speech To Text",
                            "retailPrice": 1.0,
                            "unitOfMeasure": "1 Hour",
                        },
                        {
                            "meterName": "S1 Neural Text To Speech Characters",
                            "retailPrice": 15.0,
                            "unitOfMeasure": "1M",
                        },
                    ]
                ),
            )
        )
        result = await AzurePricingSource(CatalogConfig()).fetch()

    assert result.data["azure/speech/azure-stt"]["rate"] == pytest.approx(1.0 / 3600)
    assert result.data["azure/speech/azure-tts"]["rate"] == pytest.approx(1.5e-05)
    assert result.data["azure/speech/azure-stt"]["billing"]["region"] == AZURE_CANONICAL_REGION


@pytest.mark.asyncio
async def test_a_zero_priced_azure_meter_is_refused():
    """Azure publishes "Free <meter>" rows beside the paid ones.

    A zero reaching the catalog is a customer billed nothing, silently. Better to have no
    billing block than a free one.
    """
    with respx.mock:
        respx.get(url__startswith=AZURE_RETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_azure_payload(
                    [
                        {
                            "meterName": "S1 Speech To Text",
                            "retailPrice": 0.0,
                            "unitOfMeasure": "1 Hour",
                        }
                    ]
                ),
            )
        )
        result = await AzurePricingSource(CatalogConfig()).fetch()

    assert result.data == {}


@pytest.mark.asyncio
async def test_unmapped_azure_meters_are_ignored():
    """169 meters for one region; the allow-list is the whole point."""
    with respx.mock:
        respx.get(url__startswith=AZURE_RETAIL_URL).mock(
            return_value=httpx.Response(
                200,
                json=_azure_payload(
                    [
                        {
                            "meterName": "Fast Transcription Speech To Text",
                            "retailPrice": 0.36,
                            "unitOfMeasure": "1 Hour",
                        },
                        {
                            "meterName": "Neural HD Text to Speech Characters",
                            "retailPrice": 22.0,
                            "unitOfMeasure": "1M",
                        },
                    ]
                ),
            )
        )
        result = await AzurePricingSource(CatalogConfig()).fetch()

    assert result.data == {}


# --------------------------------------------------------------------------------- #
# the overlay
# --------------------------------------------------------------------------------- #


def test_the_overlay_attaches_billing_and_keeps_the_rate_when_they_agree():
    models = {"aws_polly/standard": {"input_cost_per_character": 4e-06, "mode": "audio_speech"}}
    overlay = {
        "aws_polly/standard": {
            "cost_field": "input_cost_per_character",
            "rate": 4e-06,
            "billing": {"confidence": "authoritative"},
        }
    }

    assert apply_pricing_overlay(models, overlay) == 1
    assert models["aws_polly/standard"]["input_cost_per_character"] == 4e-06
    assert models["aws_polly/standard"]["billing"]["confidence"] == "authoritative"


def test_the_vendor_feed_wins_a_disagreement_but_says_so(caplog):
    """The likeliest cause is a SKU mapped to the wrong entry, not a repricing."""
    models = {"aws_polly/standard": {"input_cost_per_character": 9.9e-06}}
    overlay = {
        "aws_polly/standard": {
            "cost_field": "input_cost_per_character",
            "rate": 4e-06,
            "billing": {"confidence": "authoritative"},
        }
    }

    with caplog.at_level("WARNING"):
        apply_pricing_overlay(models, overlay)

    assert models["aws_polly/standard"]["input_cost_per_character"] == 4e-06
    assert "disagreement" in caplog.text.lower()


def test_a_price_with_no_catalog_entry_warns_rather_than_inventing_one(caplog):
    """An orphaned SKU means the mapping is stale, or the model left the upstream feed.

    Creating the entry here would add a model no feed lists, described by nothing.
    """
    models: dict = {}
    overlay = {
        "aws_polly/standard": {
            "cost_field": "input_cost_per_character",
            "rate": 4e-06,
            "billing": {},
        }
    }

    with caplog.at_level("WARNING"):
        assert apply_pricing_overlay(models, overlay) == 0

    assert models == {}
    assert "no catalog entry" in caplog.text
