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


"""The gates a scraped rate passes before it can become a price.

There is no human review between a pricing page changing and bud-connect serving the new
number, by design. That makes these checks the only thing standing between a page redesign
and a wrong invoice, so they are written to catch the failures scraping actually produces
rather than the ones that are easy to test:

* a regex that matches the wrong column and returns a plan price instead of a unit rate
* a per-hour figure stored as if it were per-second (3600x)
* a misplaced decimal (10x, 100x)
* a page that now renders prices in JavaScript, so everything comes back empty
* a table that gained a row, so the same model is extracted twice with different rates

The bounds below are deliberately wide. They are not a judgement about what a fair price
is -- they exist to catch arithmetic that is wrong by orders of magnitude, and a bound
tight enough to argue about is a bound that will reject a real repricing.
"""

from __future__ import annotations

import logging
import math

from .base import UNIT_TO_COST_FIELD, ScrapedModel

logger = logging.getLogger(__name__)

#: Plausible range for a rate, per unit. A second of transcription below $1e-6 would be
#: $0.0036/hr and above $1e-2 would be $36/hr; both are far outside anything any vendor
#: charges, so a value beyond them is arithmetic, not pricing.
UNIT_BOUNDS: dict[str, tuple[float, float]] = {
    "second": (1e-6, 1e-2),
    "character": (1e-7, 1e-3),
}

#: Modes the catalog understands for speech.
VALID_MODES = frozenset({"audio_transcription", "audio_speech"})

#: Confidence levels an adapter may claim. AUTHORITATIVE is reserved for vendor price
#: APIs and is not reachable from a page read.
VALID_CONFIDENCE = frozenset({"curated", "derived"})

#: How far a scraped rate may move from the committed floor before it is treated as an
#: extraction bug rather than a repricing.
#:
#: A vendor cutting a price by half, or doubling it, is ordinary -- Speechmatics currently
#: shows a 46% promotional cut. A vendor changing a price by 10x is not something that
#: happens; a parser reading "per hour" as "per second" is. So this rejects the arithmetic
#: failure without standing in the way of real price movement.
MAX_DRIFT_FACTOR = 10.0

#: Ceiling on a per-request minimum. Rev AI's is 15 seconds; a vendor charging a minimum of
#: hours does not exist, so anything past this is a mis-parse.
_MAX_MIN_BILLABLE_UNITS = 3600.0


def validate_model(m: ScrapedModel) -> str | None:
    """Check one extracted model in isolation.

    Returns:
        A human-readable rejection reason, or None if the model is acceptable.
    """
    if not m.model or m.model.strip() != m.model:
        return f"model name {m.model!r} is empty or has surrounding whitespace"
    if "/" in m.model:
        # The catalog key is "{vendor}/{model}"; a slash in the model half silently
        # reparents the entry under a vendor that does not exist.
        return f"model name {m.model!r} contains '/', which separates vendor from model"
    if m.mode not in VALID_MODES:
        return f"{m.model}: unknown mode {m.mode!r}"
    if m.unit not in UNIT_TO_COST_FIELD:
        return f"{m.model}: unknown unit {m.unit!r}"
    if m.confidence not in VALID_CONFIDENCE:
        return f"{m.model}: confidence {m.confidence!r} is not claimable by a page read"
    if not isinstance(m.rate, (int, float)) or isinstance(m.rate, bool):
        return f"{m.model}: rate {m.rate!r} is not a number"
    if not math.isfinite(m.rate):
        # A NaN rate compares False against every bound, so without this it passes every
        # range check below and lands in the catalog as a price.
        return f"{m.model}: rate is {m.rate}"
    if m.rate <= 0:
        # Zero is the dangerous one: it is a request billed as free, and it looks
        # completely unremarkable in a table.
        return f"{m.model}: rate {m.rate} is not positive"

    low, high = UNIT_BOUNDS[m.unit]
    if not (low <= m.rate <= high):
        return (
            f"{m.model}: rate {m.rate:g}/{m.unit} is outside the plausible range "
            f"[{low:g}, {high:g}] -- published as {m.published_as!r}, so check the unit "
            "conversion"
        )
    if not m.published_as:
        # Without this the stored rate cannot be re-checked against the page by anyone
        # who did not write the adapter.
        return f"{m.model}: no published_as, so the rate cannot be traced back to the page"
    if m.promotional_rate is not None:
        if not math.isfinite(m.promotional_rate) or m.promotional_rate <= 0:
            return f"{m.model}: promotional rate {m.promotional_rate!r} is not a positive number"
        if m.promotional_rate > m.rate:
            # A "promotion" above list means the two were read from the wrong columns.
            return (
                f"{m.model}: promotional rate {m.promotional_rate:g} exceeds list rate "
                f"{m.rate:g}, so the columns are likely swapped"
            )
    if m.min_billable_units is not None:
        if not math.isfinite(m.min_billable_units) or m.min_billable_units <= 0:
            return (
                f"{m.model}: min_billable_units {m.min_billable_units!r} is not a positive number"
            )
        if m.min_billable_units > _MAX_MIN_BILLABLE_UNITS:
            # A "minimum" of thousands of units is a parsed year or a phone number, not a
            # billing floor, and it would multiply every short request's cost.
            return f"{m.model}: min_billable_units {m.min_billable_units:g} is implausibly large"
    return None


def validate_batch(
    vendor: str, models: list[ScrapedModel], min_models: int
) -> tuple[list[ScrapedModel], list[str]]:
    """Apply :func:`validate_model` across one adapter's output, plus batch-level checks.

    Returns:
        ``(accepted, rejections)``. An empty ``accepted`` means the vendor contributes
        nothing and its committed floor stands.
    """
    rejections: list[str] = []

    # Pages repeat themselves -- Deepgram renders its price table once as plan cards and
    # again as a calculator -- so a repeat that says the same thing is collapsed. A repeat
    # that DISAGREES means the parser matched two different things under one name, and
    # which rate would win is an accident of ordering. That rejects the whole vendor: one
    # conflicting row makes every other row from the same parse suspect.
    #
    # This is the only place duplicates are resolved. Adapters return every match they find;
    # one that kept "the first" itself would make this check unreachable, and a page that
    # reordered its tables would then change the price with nothing rejected.
    by_model: dict[str, ScrapedModel] = {}
    conflicts: set[str] = set()
    for m in models:
        first = by_model.setdefault(m.model, m)
        if first is not m and _billing_shape(first) != _billing_shape(m):
            conflicts.add(m.model)
    if conflicts:
        return [], [
            f"{vendor}: model keys {sorted(conflicts)} appear with different rates; "
            "rejecting the whole extraction"
        ]
    models = list(by_model.values())

    accepted: list[ScrapedModel] = []
    for m in models:
        reason = validate_model(m)
        if reason:
            rejections.append(f"{vendor}: {reason}")
        else:
            accepted.append(m)

    if len(accepted) < min_models:
        # Below the floor the page has changed shape. Accepting what survived would
        # publish a partial catalog that is indistinguishable from a vendor retiring
        # models, so nothing is taken.
        rejections.append(
            f"{vendor}: extracted {len(accepted)} usable models, below the floor of "
            f"{min_models}; taking none of them"
        )
        return [], rejections

    return accepted, rejections


def _billing_shape(m: ScrapedModel) -> tuple[object, ...]:
    """What two extractions of one model must agree on to be the same price."""
    return (m.mode, m.unit, m.rate, m.min_billable_units, m.confidence)


def exceeds_drift(scraped_rate: float, floor_rate: float, factor: float = MAX_DRIFT_FACTOR) -> bool:
    """Whether a new rate has moved far enough from the known one to be a bug.

    Guards against the unit-conversion and decimal-point failures, which move a rate by
    factors, while leaving ordinary repricing alone.
    """
    if floor_rate <= 0 or not math.isfinite(floor_rate):
        return False  # nothing trustworthy to compare against
    if not math.isfinite(scraped_rate) or scraped_rate <= 0:
        return True
    ratio = scraped_rate / floor_rate
    return ratio > factor or ratio < 1 / factor
