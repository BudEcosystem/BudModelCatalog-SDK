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


"""Google Cloud Text-to-Speech: a real per-model table, priced per character.

The one page in this package that needs no unit conversion -- Google quotes
"US$0.00003 per character" directly, with the SKU next to each model.

Parsed by table row rather than from stripped text, for the reason the Rev AI adapter
records: stripping tags joins every cell into one line and a model name then absorbs the
column heading in front of it.

Two things are deliberately skipped:

* The Gemini TTS rows (Gemini 2.5 Flash TTS and friends) are priced per million *tokens*,
  input and output separately. That is a different billing basis from per-character and the
  catalog cannot express it, so they are left out rather than stored in a unit they are not
  quoted in.
* Google's free tiers ("0 to 1 million characters"). The stored rate is the one charged
  after the free allowance, because a cost estimate that assumes a free tier is wrong for
  every account that has already used it.

Speech-to-Text is a separate page and is NOT covered here. Its table interleaves list
prices with 1-year and 3-year committed-savings columns, so picking a rate means knowing
which column you are in -- and getting that wrong silently under-bills by 20%. It needs its
own adapter written against that structure rather than a guess.
"""

from __future__ import annotations

import html as html_mod
import re

from ..base import AUDIO_SPEECH, ScrapedModel, VendorScraper

_ROW = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.S | re.I)

#: "US$0.00003 per character", the only form this adapter accepts.
_PER_CHARACTER = re.compile(r"US\$\s*(\d*\.\d+)\s*per\s+character", re.I)

#: Names arrive as "WaveNet voices", "Chirp 3: HD voices", "Instant custom voice". Only the
#: PLURAL is stripped: in "WaveNet voices" it is a category suffix, but in "Instant custom
#: voice" the singular is part of the product name, and dropping it leaves "instant-custom".
_TRAILING_VOICES = re.compile(r"\s+voices$", re.I)
_PARENTHETICAL = re.compile(r"\s*\([^)]*\)")


def _strip_tags(fragment: str) -> str:
    return re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def _slug(name: str) -> str:
    # "Chirp 3: HD voices" -> "chirp-3-hd"; "Polyglot (Preview) voices" -> "polyglot".
    # The strip matters: the caller splits on "(sku", which leaves trailing whitespace that
    # would stop the `voices$` anchor below from matching at all.
    name = _PARENTHETICAL.sub("", name).strip()
    name = _TRAILING_VOICES.sub("", name).strip()
    name = name.replace(":", " ").replace("/", " ")
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")


class GoogleSpeechScraper(VendorScraper):
    vendor = "google_speech"
    url = "https://cloud.google.com/text-to-speech/pricing"
    # Chirp 3 HD, instant custom voice, WaveNet, Studio, Standard, Neural2, Polyglot.
    min_models = 6

    def extract(self, html: str) -> list[ScrapedModel]:
        rows = _ROW.findall(html)
        if not rows:
            raise ValueError("no table rows found; the pricing page structure changed")

        out: list[ScrapedModel] = []
        seen: set[str] = set()
        for row in rows:
            cells = _CELL.findall(row)
            if len(cells) < 2:
                continue

            # The rate lives in the last column; anything without a per-character rate is
            # either a Gemini token row or not a pricing row at all.
            price = _PER_CHARACTER.search(_strip_tags(cells[-1]))
            if not price:
                continue

            # The name is the first cell, up to the SKU that follows it.
            label = _strip_tags(cells[0]).split("(sku")[0].strip()
            key = _slug(label)
            if not key or key in seen:
                continue

            rate = float(price.group(1))
            seen.add(key)
            out.append(
                ScrapedModel(
                    model=key,
                    mode=AUDIO_SPEECH,
                    unit="character",
                    rate=rate,
                    published_as=f"US${rate:g} per character",
                    note=(
                        f"{label.strip()}; quoted per character, charged after the free "
                        "usage allowance."
                    ),
                )
            )
        return out
