"""Authoritative speech pricing from the cloud vendors' own price feeds.

AWS and Azure publish machine-readable price lists with no authentication, which makes
them the only speech prices in this catalog that refresh themselves. Everything else is
either a third-party mirror (LiteLLM) or a hand-read pricing page (``curated.py``).

WHAT THIS SOURCE DOES, AND DOES NOT, DO
---------------------------------------
It does not invent catalog entries. LiteLLM already lists ``aws_polly/standard``,
``azure/speech/azure-stt`` and the rest; this source attaches the ``billing`` block those
entries have no way to carry -- the region the rate belongs to, the vendor's own
publication date, and ``confidence: authoritative`` -- and audits the rate while it is
there.

The audit is the part worth having. Every canonical rate below was checked by hand against
both feeds on 2026-09-22 and agreed to the last digit, so a future disagreement means one
of three things: the vendor repriced, LiteLLM drifted, or the SKU mapping is wrong. All
three deserve a human, which is why a mismatch logs loudly rather than passing quietly.

WHY ONLY SOME SKUS
------------------
AWS Transcribe publishes 167 usage types and Azure Speech 169 meters for one region. The
difference between neighbours is not small: ``USE1-CallAnalyticsTranscribeAudio`` is five
times ``USE1-TranscribeAudio``, and several Azure meters are named ``Free ...`` and priced
zero. An adapter that pattern-matched on names would eventually bill a customer nothing
and no test would catch it. So the mapping is an explicit, hand-verified allow-list, and a
SKU that is not in it is ignored.

GCP is deliberately absent: its Cloud Billing Catalog refuses unauthenticated callers and
this SDK carries no credentials.
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .base import BaseSource, FetchResult

logger = logging.getLogger(__name__)

#: Region whose prices are treated as canonical. Cloud speech prices vary by region and
#: the catalog stores one number, so one region has to be chosen and written down.
AWS_CANONICAL_REGION = "us-east-1"
AWS_REGION_PREFIX = "USE1-"
AZURE_CANONICAL_REGION = "eastus"

AWS_OFFER_URL = "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{offer}/current/index.json"
AZURE_RETAIL_URL = "https://prices.azure.com/api/retail/prices"
AZURE_FILTER = (
    "serviceFamily eq 'AI + Machine Learning' and productName eq 'Azure Speech' "
    "and armRegionName eq '{region}'"
)


@dataclass(frozen=True)
class RateSpec:
    """Where a vendor's published rate lands in the catalog.

    ``divisor`` converts the vendor's display unit into the unit the catalog stores:
    Azure quotes speech-to-text per hour and synthesis per million characters, and storing
    those verbatim would make every consumer repeat the conversion.
    """

    catalog_key: str
    cost_field: str
    unit: str
    meter: str
    divisor: float = 1.0


#: AWS: usage type (without the region prefix) -> where it lands.
#: Verified 2026-09-22 against the offer files published 2026-09-11.
#: Excluded on purpose: CallAnalytics, Redaction, Medical, HealthScribe, Toxicity, Summary
#: and the Clm (custom language model) variants -- all real SKUs, none of them the base
#: operation a user gets by default.
AWS_SKUS: dict[str, dict[str, RateSpec]] = {
    "AmazonPolly": {
        "SynthesizeSpeech-Characters": RateSpec(
            "aws_polly/standard", "input_cost_per_character", "character", "input_characters"
        ),
        "SynthesizeSpeechNeural-Characters": RateSpec(
            "aws_polly/neural", "input_cost_per_character", "character", "input_characters"
        ),
        "SynthesizeSpeechGenerative-Characters": RateSpec(
            "aws_polly/generative", "input_cost_per_character", "character", "input_characters"
        ),
        "SynthesizeSpeechLongForm-Characters": RateSpec(
            "aws_polly/long-form", "input_cost_per_character", "character", "input_characters"
        ),
    },
    "transcribe": {
        "TranscribeAudio": RateSpec(
            "aws_transcribe/StartTranscriptionJob",
            "input_cost_per_second",
            "second",
            "input_audio_seconds",
        ),
    },
}

#: Azure: meter name -> where it lands.
#: ``Neural HD Text to Speech Characters`` ($22/1M) is deliberately NOT mapped. LiteLLM's
#: ``azure/speech/azure-tts-hd`` is $30/1M, and the two probably are not the same product --
#: LiteLLM's HD entry looks like OpenAI's ``tts-1-hd`` hosted on Azure rather than Azure's
#: own Neural HD voice. Mapping them together would overwrite a possibly-correct rate with
#: a confidently wrong one, which is worse than leaving the entry as it was.
AZURE_METERS: dict[str, RateSpec] = {
    "S1 Speech To Text": RateSpec(
        "azure/speech/azure-stt",
        "input_cost_per_second",
        "second",
        "input_audio_seconds",
        divisor=3600.0,
    ),
    "S1 Neural Text To Speech Characters": RateSpec(
        "azure/speech/azure-tts",
        "input_cost_per_character",
        "character",
        "input_characters",
        divisor=1_000_000.0,
    ),
}


def _billing(unit: str, meter: str, region: str, url: str, published: str | None) -> dict[str, Any]:
    return {
        "unit": unit,
        "meter": meter,
        "region": region,
        "currency": "USD",
        "confidence": "authoritative",
        "source": {
            "url": url,
            "checked_on": datetime.now(timezone.utc).date().isoformat(),
            "published": published,
        },
    }


class AwsPricingSource(BaseSource):
    """AWS Price List bulk API. Public, unauthenticated, one JSON file per service."""

    async def fetch(self) -> FetchResult:
        """Read the Polly and Transcribe offer files and extract the canonical SKUs."""
        overlay: dict[str, dict[str, Any]] = {}

        for offer, skus in AWS_SKUS.items():
            url = AWS_OFFER_URL.format(offer=offer)
            response = await self._fetch_url(url, headers={}, label=f"aws-pricing:{offer}")
            payload = response.json()
            published = payload.get("publicationDate")
            terms = payload.get("terms", {}).get("OnDemand", {})

            for sku_id, product in payload.get("products", {}).items():
                usagetype = product.get("attributes", {}).get("usagetype", "")
                if not usagetype.startswith(AWS_REGION_PREFIX):
                    continue
                spec = skus.get(usagetype[len(AWS_REGION_PREFIX) :])
                if spec is None:
                    continue

                rate = _first_tier_rate(terms.get(sku_id, {}))
                if rate is None:
                    logger.warning("No on-demand price dimension for AWS SKU %s", usagetype)
                    continue

                overlay[spec.catalog_key] = {
                    "cost_field": spec.cost_field,
                    "rate": rate,
                    "billing": _billing(
                        spec.unit, spec.meter, AWS_CANONICAL_REGION, url, published
                    ),
                }

        logger.info("AWS pricing: resolved %d canonical SKUs", len(overlay))
        return FetchResult(
            data=overlay, source_name="aws_pricing", fetched_at=datetime.now(timezone.utc)
        )


def _first_tier_rate(term: dict[str, Any]) -> float | None:
    """The undiscounted rate: the cheapest-threshold tier, not the cheapest price.

    AWS lists volume tiers as separate price dimensions under one term, in no guaranteed
    order. Taking ``min()`` would quietly bill everyone the highest-volume discount.
    ``beginRange`` is the tier's lower bound, so the one starting at 0 is list price.
    """
    best: tuple[float, float] | None = None
    for offer in term.values():
        for dimension in offer.get("priceDimensions", {}).values():
            usd = dimension.get("pricePerUnit", {}).get("USD")
            if usd is None:
                continue
            try:
                begin = float(dimension.get("beginRange", 0) or 0)
                price = float(usd)
            except (TypeError, ValueError):
                continue
            if best is None or begin < best[0]:
                best = (begin, price)
    return best[1] if best else None


class AzurePricingSource(BaseSource):
    """Azure Retail Prices API. Public, unauthenticated, OData-filtered.

    Note the filter: ``serviceName eq 'Cognitive Services'`` returns zero rows -- that
    service name is stale -- so the query goes through ``serviceFamily`` and
    ``productName`` instead.
    """

    async def fetch(self) -> FetchResult:
        """Query the retail feed and extract the canonical meters."""
        # `quote` with an empty safe set, because the service family is literally
        # "AI + Machine Learning". A raw `+` in a query string is a SPACE to the server,
        # so the filter silently becomes "AI   Machine Learning", matches nothing, and
        # returns 200 with an empty Items array -- a success that resolves no meters.
        query = urllib.parse.quote(AZURE_FILTER.format(region=AZURE_CANONICAL_REGION), safe="")
        url = f"{AZURE_RETAIL_URL}?$filter={query}"
        response = await self._fetch_url(url, headers={}, label="azure-pricing")
        payload = response.json()

        items = payload.get("Items", [])
        if not items:
            # Distinguishable from "the meters moved": an empty feed for a filter that is
            # known to match is almost always the filter being mangled in transit.
            logger.warning(
                "Azure retail feed returned no rows for %s; check the filter encoding", url
            )

        overlay: dict[str, dict[str, Any]] = {}
        for item in items:
            spec = AZURE_METERS.get(item.get("meterName", ""))
            if spec is None:
                continue

            price = item.get("retailPrice")
            if price is None:
                continue
            if price == 0:
                # Azure publishes "Free <meter>" rows alongside the paid ones. A zero that
                # reaches the catalog is a customer billed nothing, silently.
                logger.warning("Ignoring zero-priced Azure meter %r", item.get("meterName"))
                continue

            overlay[spec.catalog_key] = {
                "cost_field": spec.cost_field,
                "rate": price / spec.divisor,
                "billing": _billing(
                    spec.unit, spec.meter, AZURE_CANONICAL_REGION, AZURE_RETAIL_URL, None
                ),
            }

        if payload.get("NextPageLink"):
            # Every canonical meter is on page one today. If that stops being true the
            # missing ones simply do not get a billing block, which a warning should
            # explain rather than leaving someone to wonder.
            logger.info(
                "Azure retail feed has further pages; canonical meters resolved: %d", len(overlay)
            )

        logger.info("Azure pricing: resolved %d canonical meters", len(overlay))
        return FetchResult(
            data=overlay, source_name="azure_pricing", fetched_at=datetime.now(timezone.utc)
        )


def apply_pricing_overlay(
    models: dict[str, dict[str, Any]],
    overlay: dict[str, dict[str, Any]],
    *,
    tolerance: float = 1e-9,
) -> int:
    """Attach authoritative billing blocks, and audit the rates already there.

    The vendor's own feed wins a disagreement -- it is the vendor's own feed -- but it
    never wins quietly: both numbers go in the log, because the likeliest cause of a
    disagreement is not a repricing but a SKU mapped to the wrong catalog entry.

    Args:
        models: The merged catalog, mutated in place.
        overlay: Canonical rates keyed by catalog key.
        tolerance: Absolute difference below which two rates are considered equal.

    Returns:
        How many entries were enriched.
    """
    enriched = 0
    for key, entry in overlay.items():
        target = models.get(key)
        if target is None:
            logger.warning(
                "Authoritative price for %s has no catalog entry to attach to; "
                "either the SKU mapping is stale or the model left the upstream feed",
                key,
            )
            continue

        field, rate = entry["cost_field"], entry["rate"]
        existing = target.get(field)
        if existing is not None and abs(float(existing) - rate) > tolerance:
            logger.warning(
                "Rate disagreement on %s.%s: catalog has %r, vendor feed says %r -- taking the "
                "vendor's. Check whether they repriced or the SKU is mapped to the wrong model.",
                key,
                field,
                existing,
                rate,
            )

        target[field] = rate
        target["billing"] = entry["billing"]
        enriched += 1

    return enriched
