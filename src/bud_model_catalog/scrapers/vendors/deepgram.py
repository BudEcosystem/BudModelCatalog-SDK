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


"""Deepgram text-to-speech: Aura, which no feed lists at all.

Deepgram's speech-to-text reaches the catalog through LiteLLM. Its text-to-speech does not:
LiteLLM carries 42 Deepgram entries and every one is `audio_transcription`, so the provider
was advertising TEXT_TO_SPEECH with no TTS model behind it. This adapter supplies them.

WHICH MODELS

Deepgram's own `/v1/models` lists 102 TTS entries, which are VOICES, in two billable
families: `aura-2` (90 voices) and `aura` (12). The price is per family, not per voice, so
the family is the catalog entry -- the same granularity used for Google's WaveNet and
Studio. The pricing page names the families differently from the API ("Aura-1" is the
`aura` architecture), so the mapping below is written down rather than slugified.

Flux TTS is on the pricing page at $0.045 per 1k characters and is deliberately NOT emitted.
It is absent from `/v1/models`, and the only `flux-tts` string in Deepgram's docs is a URL
slug in the sidebar whose pages return 404. Nothing establishes what `model=` value would
select it, and a key this project invented would show up in budapp as a model nobody can
deploy -- which is why Speechify's single entry was removed. Add it when the id is known.

WHICH RATE

Pay As You Go, the undiscounted column. Growth is a volume commitment 10% lower, recorded
in the note rather than used -- the same rule as AWS tier 0 and Gladia's Starter plan.
"""

from __future__ import annotations

import html as html_mod
import re

from ..base import AUDIO_SPEECH, ScrapedModel, VendorScraper

#: Pricing-page family name -> the architecture name Deepgram's own /v1/models uses. Only
#: families whose id is confirmed by that API belong here.
FAMILIES: dict[str, str] = {
    "Aura-2": "aura-2",
    "Aura-1": "aura",
}

#: "Aura-2 $0.030/1k characters $0.027/1k characters" -- Pay As You Go, then Growth. The
#: family names are alternated explicitly so an unlisted family cannot match.
_ROW = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in FAMILIES) + r")\s+"
    r"\$(\d+(?:\.\d+)?)\s*/\s*1k\s+characters"
    r"(?:\s+\$(\d+(?:\.\d+)?)\s*/\s*1k\s+characters)?",
    re.I,
)

_CHARACTERS_PER_UNIT = 1000.0


class DeepgramTtsScraper(VendorScraper):
    vendor = "deepgram"
    name = "deepgram_tts"
    url = "https://deepgram.com/pricing"
    min_models = 2

    def extract(self, html: str) -> list[ScrapedModel]:
        text = re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", " ", html)))

        rows = _ROW.findall(text)
        if not rows:
            raise ValueError(
                "no 'Aura-2 $X/1k characters' rows found; the TTS pricing table changed"
            )

        out: list[ScrapedModel] = []
        seen: set[str] = set()
        for family, payg, growth in rows:
            model = FAMILIES[_canonical(family)]
            if model in seen:
                # The page renders the table twice (plan cards and a calculator). Keep the
                # first; validate_batch would reject a disagreeing duplicate anyway.
                continue
            seen.add(model)
            list_rate = float(payg)
            out.append(
                ScrapedModel(
                    model=model,
                    mode=AUDIO_SPEECH,
                    unit="character",
                    rate=list_rate / _CHARACTERS_PER_UNIT,
                    published_as=f"${list_rate:g}/1k characters",
                    note=(
                        f"Pay As You Go ${list_rate:g} per 1k characters; {list_rate:g} / 1000."
                        + (f" Growth plan ${float(growth):g}/1k." if growth else "")
                        + " Priced per family; applies to every voice in it."
                    ),
                )
            )
        return out


def _canonical(family: str) -> str:
    """Map a case-variant match back to its key in FAMILIES."""
    return next(k for k in FAMILIES if k.lower() == family.lower())
