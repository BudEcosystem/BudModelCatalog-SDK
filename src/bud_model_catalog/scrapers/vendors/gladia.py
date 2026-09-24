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


"""Gladia: two rates on the entry plan.

The page shows Starter as "Async at $0.61 /hr" and "Real-time at $0.75 /hr", and Growth as
"Async as low as $0.20 /hr". Only Starter is matched: Growth requires an upfront commitment,
which makes it a volume discount rather than a list price, and list is what gets stored
here for the same reason it is for AWS tiers and Speechmatics promotions.

The phrasing difference is what separates them -- "at $X" against "as low as $X" -- so a
pattern anchored on "at" cannot pick up a committed rate by accident.
"""

from __future__ import annotations

import html as html_mod
import re

from ..base import AUDIO_TRANSCRIPTION, ScrapedModel, VendorScraper

#: "Async at $0.61 /hr" -- note `at` immediately before the price, which "as low as $0.20
#: /hr" does not match.
_RATE = re.compile(r"(Async|Real-time)\s+at\s+\$(\d+(?:\.\d+)?)\s*/\s*hr", re.I)

_SECONDS_PER_HOUR = 3600.0


class GladiaScraper(VendorScraper):
    vendor = "gladia"
    url = "https://www.gladia.io/pricing"
    min_models = 2

    def extract(self, html: str) -> list[ScrapedModel]:
        text = re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", " ", html)))

        found = _RATE.findall(text)
        if not found:
            raise ValueError("no 'Async at $X /hr' rows found on the Starter plan; page changed")

        out: list[ScrapedModel] = []
        seen: set[str] = set()
        for label, amount in found:
            key = label.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            per_hour = float(amount)
            out.append(
                ScrapedModel(
                    model=key,
                    mode=AUDIO_TRANSCRIPTION,
                    unit="second",
                    rate=per_hour / _SECONDS_PER_HOUR,
                    published_as=f"${per_hour:g}/hr",
                    note=(
                        f"Starter plan ${per_hour:g}/hr; {per_hour:g} / 3600. Growth tier is "
                        "cheaper but requires an upfront commitment."
                    ),
                )
            )
        return out
