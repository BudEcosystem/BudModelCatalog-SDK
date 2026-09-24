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

"""Per-vendor pricing page adapters.

Every vendor here publishes prices only as a web page. There is no feed to subscribe to
and no API to query, so the rate is read out of HTML. That makes each adapter a small
parser against a document somebody else controls and can redesign without warning.

The design consequence runs through this whole package: **an adapter that cannot extract
must fail, never return less.** A parser that quietly yields three models out of seven
looks exactly like a vendor that retired four of them, and bud-connect's seeder treats
"absent from this run" as a reason to deactivate. Partial success is the dangerous
outcome, so `min_models` exists to turn it into a failure.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

logger = logging.getLogger(__name__)

#: Catalog modes an adapter may emit.
AUDIO_TRANSCRIPTION = "audio_transcription"
AUDIO_SPEECH = "audio_speech"

#: Billing units, and the catalog cost field each one is carried in.
UNIT_TO_COST_FIELD: dict[str, str] = {
    "second": "input_cost_per_second",
    "character": "input_cost_per_character",
}

#: The meter name that goes in the billing block for each unit. This is what a consumer
#: multiplies by the rate, so it has to name the quantity, not the model's output.
UNIT_TO_METER: dict[str, str] = {
    "second": "input_audio_seconds",
    "character": "input_characters",
}


@dataclass(frozen=True)
class ScrapedModel:
    """One billable model read off a vendor's pricing page.

    Attributes:
        model: Catalog key suffix, e.g. `batch-melia-1`. Must not contain `/`, which
            separates vendor from model in a catalog key.
        mode: :data:`AUDIO_TRANSCRIPTION` or :data:`AUDIO_SPEECH`.
        unit: `second` or `character` -- the unit `rate` is expressed in, which is not
            usually the unit the page quotes.
        rate: Cost per single unit, in USD.
        published_as: What the page literally said, e.g. `"$0.24/hr"`. Carried so a human
            can re-check the arithmetic without re-reading the page, and so a unit
            conversion error is visible next to its result.
        promotional_rate: A discounted rate shown alongside the list price, if any. Never
            used as `rate` -- promotions expire, and a cost estimate that quietly assumes
            one is wrong the day it ends. Recorded so the gap is visible.
        confidence: `curated` for a rate read directly off the page, `derived` for one
            computed from something that is not a price (Cartesia sells credits).
        note: Free text appended to the billing block's source note.
    """

    model: str
    mode: str
    unit: str
    rate: float
    published_as: str
    promotional_rate: float | None = None
    confidence: str = "curated"
    note: str | None = None


@dataclass
class ScrapeOutcome:
    """What one adapter produced, including why anything was dropped.

    Rejections are part of the result rather than a log line because the count is the
    signal: an adapter that suddenly rejects everything it extracts is broken in a way
    that "returned no models" does not distinguish from "vendor removed the table".
    """

    vendor: str
    url: str
    models: list[ScrapedModel] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.models)


class VendorScraper(ABC):
    """Base for a single vendor's pricing page adapter.

    Subclasses implement :meth:`extract` against saved HTML. They do no I/O: fetching,
    timeouts, concurrency and validation all belong to the source that drives them, so an
    adapter can be tested against a fixture with no network at all.
    """

    #: bud-connect provider key. Becomes the first half of the catalog key.
    vendor: ClassVar[str]

    #: The pricing page to read.
    url: ClassVar[str]

    #: Floor on a successful extraction. Set it to the number of models the page listed
    #: when the adapter was written: dropping below it means the page changed shape, not
    #: that the vendor shrank its lineup.
    min_models: ClassVar[int] = 1

    @abstractmethod
    def extract(self, html: str) -> list[ScrapedModel]:
        """Parse *html* into billable models.

        Raise any exception to signal a broken page; the driver isolates it. Returning an
        empty list is also a failure -- it is never a valid reading of a pricing page.
        """
        ...
