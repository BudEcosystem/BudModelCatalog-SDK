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


"""ElevenLabs: flat per-unit API prices, one card per model family.

LiteLLM lists five ElevenLabs entries, and two of them (`scribe_v1`,
`scribe_v1_experimental`) are models ElevenLabs has withdrawn. It does not list Flash at
all, which is the family most TTS traffic goes to. This page does, and it is server-rendered:

    "v3 Text to Speech $0.10 Price per 1K characters"
    "Flash / Turbo Text to Speech $0.05 Price per 1K characters"
    "Scribe v2 Speech to Text $0.22 Price per hour"

WHICH RATE

The card price. The plan table on the same page repeats it for every plan from Free to
Business -- $0.10 per 1K for v3 at each -- so there is no tier to choose between; the plans
differ in how many characters they include, not in the rate beyond them.

WHICH MODELS

A card names a family, and a catalog key has to be an id ElevenLabs' API accepts as
`model_id` (elevenlabs.io/docs/models, read 2026-09-24), so the mapping is written down.
Left out on purpose:

* Turbo -- the card says "Flash / Turbo", but `eleven_turbo_v2_5` and `eleven_turbo_v2` are
  deprecated in favour of the Flash ids.
* v3 Conversational and Scribe v2 Realtime -- realtime models served over WebSocket. WaaV
  reaches ElevenLabs over its HTTP APIs, so a deployment of either would fail, which is the
  same reason realtime models elsewhere in the catalog have no route.
"""

from __future__ import annotations

import html as html_mod
import re

from ..base import AUDIO_SPEECH, AUDIO_TRANSCRIPTION, ScrapedModel, VendorScraper

#: (card label, card kind) -> the model ids that card prices.
CARDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("v3", "Text to Speech"): ("eleven_v3",),
    ("v2 Multilingual", "Text to Speech"): ("eleven_multilingual_v2",),
    ("Flash / Turbo", "Text to Speech"): ("eleven_flash_v2_5", "eleven_flash_v2"),
    ("Scribe v2", "Speech to Text"): ("scribe_v2",),
    ("Scribe v2 Medical", "Speech to Text"): ("scribe_v2_medical",),
}

#: Cards deliberately not emitted, matched anyway so the regex cannot mistake one for a
#: shorter label ("Scribe v2 Realtime" must not read as "Scribe v2").
_SKIPPED = ("v3 Conversational", "Scribe v2 Realtime")

_LABELS = sorted({label for label, _ in CARDS} | set(_SKIPPED), key=len, reverse=True)

_CARD = re.compile(
    r"(?<![\w/])(" + "|".join(re.escape(label) for label in _LABELS) + r")\s+"
    r"(Text to Speech|Speech to Text)\s+\$(\d+(?:\.\d+)?)\s+Price per (1K characters|hour)"
)

_CHARACTERS_PER_UNIT = 1000.0
_SECONDS_PER_HOUR = 3600.0


class ElevenLabsScraper(VendorScraper):
    vendor = "elevenlabs"
    url = "https://elevenlabs.io/pricing/api"
    min_models = sum(len(ids) for ids in CARDS.values())

    def extract(self, html: str) -> list[ScrapedModel]:
        text = re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", " ", html)))

        cards = _CARD.findall(text)
        if not cards:
            raise ValueError(
                "no '<model> Text to Speech $X Price per ...' cards found; page changed"
            )

        out: list[ScrapedModel] = []
        for label, kind, amount, per in cards:
            ids = CARDS.get((label, kind))
            if ids is None:
                continue
            price = float(amount)
            if per == "hour":
                if kind != "Speech to Text":
                    raise ValueError(f"{label} {kind} is priced per hour; the card layout changed")
                mode, unit, rate = AUDIO_TRANSCRIPTION, "second", price / _SECONDS_PER_HOUR
                published = f"${price:g}/hr"
                arithmetic = f"{price:g} / 3600"
            else:
                if kind != "Text to Speech":
                    raise ValueError(
                        f"{label} {kind} is priced per character; the card layout changed"
                    )
                mode, unit, rate = AUDIO_SPEECH, "character", price / _CHARACTERS_PER_UNIT
                published = f"${price:g}/1k characters"
                arithmetic = f"{price:g} / 1000"
            for model in ids:
                out.append(
                    ScrapedModel(
                        model=model,
                        mode=mode,
                        unit=unit,
                        rate=rate,
                        published_as=published,
                        note=f"'{label} {kind}' card, {published}; {arithmetic}. Same rate on every plan.",
                    )
                )
        return out
