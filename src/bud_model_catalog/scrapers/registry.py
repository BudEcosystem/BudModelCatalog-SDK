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
from .vendors.gladia import GladiaScraper
from .vendors.revai import RevAiScraper
from .vendors.speechmatics import SpeechmaticsScraper

#: Every adapter, keyed by bud-connect provider key.
SCRAPERS: dict[str, type[VendorScraper]] = {
    SpeechmaticsScraper.vendor: SpeechmaticsScraper,
    CartesiaScraper.vendor: CartesiaScraper,
    RevAiScraper.vendor: RevAiScraper,
    GladiaScraper.vendor: GladiaScraper,
}


def all_scrapers() -> list[VendorScraper]:
    """Instantiate every registered adapter."""
    return [cls() for cls in SCRAPERS.values()]
