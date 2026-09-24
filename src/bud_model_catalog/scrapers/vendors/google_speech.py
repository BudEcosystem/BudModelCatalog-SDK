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

from ..base import AUDIO_SPEECH, AUDIO_TRANSCRIPTION, ScrapedModel, VendorScraper

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


class GoogleSpeechTtsScraper(VendorScraper):
    vendor = "google_speech"
    name = "google_speech_tts"
    url = "https://cloud.google.com/text-to-speech/pricing"
    # Chirp 3 HD, instant custom voice, WaveNet, Studio, Standard, Neural2, Polyglot.
    min_models = 6

    def extract(self, html: str) -> list[ScrapedModel]:
        rows = _ROW.findall(html)
        if not rows:
            raise ValueError("no table rows found; the pricing page structure changed")

        out: list[ScrapedModel] = []
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
            if not key:
                continue

            rate = float(price.group(1))
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


# --------------------------------------------------------------------------------- #
# speech-to-text: the same vendor, a much more dangerous table
# --------------------------------------------------------------------------------- #

#: Column headings for the committed-savings prices, which must never be read as a rate.
_SAVINGS_HEADER = re.compile(r"savings\s+plan", re.I)

#: The list-price column.
_LIST_HEADER = re.compile(r"\bprice\b", re.I)

#: "$0.016 / 1 minute" and "$0.00 (Free) / 1 minute" both appear in one cell.
_PER_MINUTE = re.compile(r"\$\s*(\d+(?:\.\d+)?)\s*(?:\(Free\)\s*)?/\s*1\s*minute", re.I)

#: "(sku:3099-B70F-0949)" and the bare "sku:67F5-A183-E319" form both occur.
_SKU = re.compile(r"\(?\bsku:\s*[0-9A-Za-z-]+\)?", re.I)

#: Footnote markers attached to model names ("Standard¹", "Medical²").
_FOOTNOTES = str.maketrans("", "", "¹²³⁴⁵*†‡")

_SECONDS_PER_MINUTE = 60.0


def _stt_slug(label: str) -> str:
    # Parentheses are meaningful here -- "with data logging" and "without data logging" are
    # different SKUs at different prices -- so unlike the TTS names they are kept.
    label = _SKU.sub("", label).translate(_FOOTNOTES)
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", label.lower())).strip("-")


class GoogleSpeechSttScraper(VendorScraper):
    """Google Cloud Speech-to-Text.

    This table is the reason this adapter was written separately and late. Every row carries
    three price columns -- list, 1-year committed savings, 3-year committed savings -- and
    they look identical in stripped text:

        Recognition (sku:3099-B70F-0949) | Standard
          | 0 to 500,000 min  $0.016 / 1 minute ...    <- list
          | 0 to 500,000 min  $0.0144 / 1 minute ...   <- 1-year commitment
          | 0 to 500,000 min  $0.0128 / 1 minute ...   <- 3-year commitment

    Reading the wrong column under-bills by 10% or 20% and looks entirely plausible while
    doing it, which is exactly the failure no bounds check can catch. So this adapter:

    * locates the list column by its HEADING rather than by position, and refuses a table
      whose heading it cannot identify -- there is no positional fallback, because a guess
      here is the whole risk;
    * asserts the rate it took is not cheaper than the same row's savings columns. List is
      by definition the most expensive, so if it is not, the columns were misread and the
      row is dropped;
    * skips the free tier inside each cell ("0 to 60 minute $0.00 (Free)") and takes the
      first paid tier, which is the lowest-volume and therefore undiscounted rate -- the
      same rule applied to AWS's tier 0.

    One header row on the page lists only three columns while its data rows carry five, so
    the heading search tolerates a short header but still requires a match.
    """

    vendor = "google_speech"
    name = "google_speech_stt"
    url = "https://cloud.google.com/speech-to-text/pricing"
    # Recognition, dynamic batch, speech recognition with/without logging, medical
    # dictation, medical conversation.
    min_models = 5

    def extract(self, html: str) -> list[ScrapedModel]:
        tables = re.findall(r"<table\b.*?</table>", html, re.S | re.I)
        if not tables:
            raise ValueError("no tables found; the pricing page structure changed")

        out: list[ScrapedModel] = []
        for table in tables:
            rows = re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.S | re.I)
            if not rows:
                continue

            header_cells = [_strip_tags(c) for _, c in _iter_cells(rows[0])]
            list_index = _list_column(header_cells)
            if list_index is None:
                continue

            for row in rows[1:]:
                cells = [_strip_tags(c) for _, c in _iter_cells(row)]
                if len(cells) <= list_index:
                    continue

                label = cells[0]
                key = _stt_slug(label)
                if not key:
                    continue

                rate_per_minute = _first_paid_rate(cells[list_index])
                if rate_per_minute is None:
                    continue

                # List must be the most expensive column in its own row. If it is not, the
                # columns were misread, and a 10% error here is invisible downstream.
                savings = [
                    r
                    for cell in cells[list_index + 1 :]
                    if (r := _first_paid_rate(cell)) is not None
                ]
                if any(rate_per_minute < s for s in savings):
                    raise ValueError(
                        f"{key}: took ${rate_per_minute:g}/min as list but a later column is "
                        f"more expensive ({', '.join(f'${s:g}' for s in savings)}); the "
                        "columns are being misread"
                    )

                out.append(
                    ScrapedModel(
                        model=key,
                        mode=AUDIO_TRANSCRIPTION,
                        unit="second",
                        rate=rate_per_minute / _SECONDS_PER_MINUTE,
                        published_as=f"${rate_per_minute:g} per minute",
                        note=(
                            f"{_SKU.sub('', label).strip()}; list price "
                            f"${rate_per_minute:g}/minute / 60, taken from the "
                            f"{header_cells[list_index].split('*')[0].strip()!r} column "
                            "rather than a committed-savings column, and after any free tier."
                        ),
                    )
                )
        return out


def _iter_cells(row: str) -> list[tuple[str, str]]:
    return re.findall(r"<(t[dh])\b[^>]*>(.*?)</\1>", row, re.S | re.I)


def _list_column(header_cells: list[str]) -> int | None:
    """Index of the undiscounted price column, or None if it cannot be identified."""
    for index, text in enumerate(header_cells):
        if _LIST_HEADER.search(text) and not _SAVINGS_HEADER.search(text):
            return index
    return None


def _first_paid_rate(cell: str) -> float | None:
    """The first non-zero per-minute rate in a cell, skipping any free tier."""
    for amount in _PER_MINUTE.findall(cell):
        value = float(amount)
        if value > 0:
            return value
    return None
