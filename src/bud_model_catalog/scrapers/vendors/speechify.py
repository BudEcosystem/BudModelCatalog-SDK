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


"""Speechify: one metered character rate, and a competitor table that must not be read.

THE TRAP ON THIS PAGE

Speechify publishes a third-party comparison ("Artificial Analysis estimates ...") listing
*other vendors'* prices per million characters, including:

    SpeechifyAI Simba 3.2   $6.60
    Inworld Realtime TTS-2  $20.80
    Alibaba Qwen-Audio-3.0  $27.60
    Cartesia Sonic 3.6      $49.00
    VUI Labs Luna TTS       $80.00

A regex looking for "$X per million characters" finds all of those. Reading Cartesia's rate
off Speechify's marketing page would be bad enough as a number; if a future adapter matched
on vendor names it could overwrite the real Cartesia rate this SDK derives from Cartesia's
own page. So the pattern here anchors on "After that $X per 1M characters" -- the plan
overage phrasing -- which the comparison table does not use. There is a test that feeds the
comparison table in and asserts none of it comes out.

WHY THERE IS ONLY ONE MODEL

Speechify meters by character at a per-plan rate, not per model: Starter $10 per 1M, Pro $8,
Scale $6. The page names no priced model -- "Simba 3.2" appears only inside the third-party
comparison. So this emits a single entry under the key `text-to-speech`, the same convention
Speechmatics' own pricing table uses for its TTS row, and the note says the rate applies to
every voice. It is a stand-in key rather than a deployable model name, which is worth
knowing before it is shown to anyone as a model.

Starter is stored, being the entry paid tier and therefore the highest unit price -- list,
by the same rule applied to AWS tiers, Gladia's Starter plan and Speechmatics' promotions.
"""

from __future__ import annotations

import html as html_mod
import re

from ..base import AUDIO_SPEECH, ScrapedModel, VendorScraper

#: "After that $10 per 1M characters". The comparison table says "per million characters"
#: and never "After that", so this cannot pick a competitor's rate up by accident.
_PLAN_OVERAGE = re.compile(r"After that\s+\$(\d+(?:\.\d+)?)\s*per\s+1M\s+characters", re.I)

_CHARACTERS_PER_UNIT = 1_000_000


class SpeechifyScraper(VendorScraper):
    vendor = "speechify"
    url = "https://speechify.com/text-to-speech-api/"
    min_models = 1

    def extract(self, html: str) -> list[ScrapedModel]:
        text = re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", " ", html)))

        rates = [float(m) for m in _PLAN_OVERAGE.findall(text)]
        if not rates:
            raise ValueError(
                "no 'After that $X per 1M characters' plan overage found; the pricing "
                "page changed shape"
            )

        # The highest published per-unit rate is the undiscounted one. Taking the max rather
        # than the first makes this independent of the order the plans are rendered in.
        per_million = max(rates)
        cheaper = sorted(r for r in rates if r < per_million)
        return [
            ScrapedModel(
                model="text-to-speech",
                mode=AUDIO_SPEECH,
                unit="character",
                rate=per_million / _CHARACTERS_PER_UNIT,
                published_as=f"${per_million:g} per 1M characters",
                note=(
                    f"Entry plan ${per_million:g} per 1M characters; {per_million:g} / 1e6. "
                    "Metered by character across all voices rather than per model."
                    + (
                        f" Higher plans are cheaper: {', '.join(f'${r:g}' for r in cheaper)} per 1M."
                        if cheaper
                        else ""
                    )
                ),
            )
        ]
