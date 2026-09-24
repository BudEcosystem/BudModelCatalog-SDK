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


"""Cartesia: sells credits, so the rate has to be derived rather than read.

There is no per-unit price on the page. There are three stated facts that together produce
one, which is the difference between deriving a number and inventing one:

    "Sonic text-to-speech: ... 1 credit equals 1 character."
    "Ink speech-to-text: ... 3 credits equals 1 second of audio."
    "At pro tier, it's $65 per 1M credits."

Every one of those is parsed out of the page rather than hardcoded. If Cartesia restates
any of them the adapter fails loudly instead of quietly carrying an assumption that is no
longer true -- which is the whole risk of a derived rate.

Pro is the entry paid tier and therefore the highest unit price, i.e. list, consistent with
how AWS tiers and Speechmatics promotions are handled elsewhere. Startup ($45/1M) and Scale
($38/1M) are volume discounts, not thresholds, so they are not modelled as tiers.
"""

from __future__ import annotations

import html as html_mod
import re

from ..base import AUDIO_SPEECH, AUDIO_TRANSCRIPTION, ScrapedModel, VendorScraper

_CREDITS_PER_CHARACTER = re.compile(r"(\d+(?:\.\d+)?)\s+credits?\s+equals?\s+1\s+character", re.I)
_CREDITS_PER_SECOND = re.compile(
    r"(\d+(?:\.\d+)?)\s+credits?\s+equals?\s+1\s+second\s+of\s+audio", re.I
)
_PRO_TIER = re.compile(r"at\s+pro\s+tier,?\s+it'?s\s+\$(\d+(?:\.\d+)?)\s+per\s+1M\s+credits", re.I)


class CartesiaScraper(VendorScraper):
    vendor = "cartesia"
    url = "https://cartesia.ai/pricing"
    min_models = 2

    def extract(self, html: str) -> list[ScrapedModel]:
        text = re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", " ", html)))

        pro = _PRO_TIER.search(text)
        if not pro:
            raise ValueError(
                "could not find the pro-tier credit price; without it a credit cannot be "
                "converted to dollars and any rate here would be invented"
            )
        usd_per_credit = float(pro.group(1)) / 1_000_000

        per_char = _CREDITS_PER_CHARACTER.search(text)
        per_second = _CREDITS_PER_SECOND.search(text)
        if not per_char and not per_second:
            raise ValueError("found no credit-to-unit conversions; the pricing FAQ changed")

        out: list[ScrapedModel] = []
        if per_char:
            credits = float(per_char.group(1))
            out.append(
                ScrapedModel(
                    model="sonic",
                    mode=AUDIO_SPEECH,
                    unit="character",
                    rate=credits * usd_per_credit,
                    published_as=f"{credits:g} credit/character at ${pro.group(1)} per 1M credits",
                    confidence="derived",
                    note=(
                        f"{credits:g} credit per character x ${pro.group(1)} per 1M credits "
                        f"(pro tier). Startup and Scale tiers are cheaper per credit."
                    ),
                )
            )
        if per_second:
            credits = float(per_second.group(1))
            out.append(
                ScrapedModel(
                    model="ink",
                    mode=AUDIO_TRANSCRIPTION,
                    unit="second",
                    rate=credits * usd_per_credit,
                    published_as=f"{credits:g} credits/second at ${pro.group(1)} per 1M credits",
                    confidence="derived",
                    note=(
                        f"{credits:g} credits per second x ${pro.group(1)} per 1M credits "
                        f"(pro tier)."
                    ),
                )
            )
        return out
