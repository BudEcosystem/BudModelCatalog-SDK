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


"""The adapters the scraped source will drive.

Registration is explicit rather than discovered by import scanning: a scraper that starts
running because a file appeared is harder to reason about than one listed here, and the
list doubles as the answer to "which vendors do we scrape".
"""

from __future__ import annotations

from .base import VendorScraper
from .vendors.cartesia import CartesiaScraper
from .vendors.deepgram import DeepgramTtsScraper
from .vendors.gladia import GladiaScraper
from .vendors.google_speech import GoogleSpeechSttScraper, GoogleSpeechTtsScraper
from .vendors.revai import RevAiScraper
from .vendors.speechmatics import SpeechmaticsScraper

#: Every adapter. A list rather than a dict keyed by vendor, because one vendor can need
#: several: Google publishes text-to-speech and speech-to-text on separate pages with
#: different table shapes, and both are `google_speech`.
SCRAPERS: tuple[type[VendorScraper], ...] = (
    SpeechmaticsScraper,
    CartesiaScraper,
    RevAiScraper,
    GladiaScraper,
    GoogleSpeechTtsScraper,
    GoogleSpeechSttScraper,
    DeepgramTtsScraper,
)


def all_scrapers() -> list[VendorScraper]:
    """Instantiate every registered adapter."""
    return [cls() for cls in SCRAPERS]


def slugs() -> list[str]:
    """Every adapter name, for selecting one on the command line."""
    return sorted(cls.slug() for cls in SCRAPERS)


def by_slug(slug: str) -> type[VendorScraper] | None:
    return next((cls for cls in SCRAPERS if cls.slug() == slug), None)
