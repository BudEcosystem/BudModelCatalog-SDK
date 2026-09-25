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


"""Rev AI: a rate per model, plus the minimum charge that makes short clips expensive.

Each offering sits in its own `payment-offering` block, so this adapter walks those blocks
rather than regexing a tag-stripped blob. That is not stylistic: the blob approach joins
every element into one line, and the first model name came out as
"supports-all-popular-media-types-email-and-chat-support-get-started-view-more-offerings-reverb"
because a non-greedy match had no element boundary to stop at. Parsing structure means a
name cannot absorb the sentence in front of it.

Every offering carries "Rounded up to the nearest second, 15 second minimum". That minimum
is the reason this vendor needs more than a rate: a 2-second clip is billed as 15 seconds,
so a consumer that reads only the rate under-bills nearly every request -- and short clips
are the common case for voice.

Excluded, deliberately:

* Human Transcription ($1.99/min) is people, not a model anyone deploys.
* Whisper Large Transcription ($0.005/min) is priced but unreachable: no documented
  `transcriber` value selects it. The streaming API WaaV calls takes machine / machine_v2,
  both Reverb (docs.rev.ai/api/streaming/transcribers), and the async API takes machine /
  human (docs.rev.ai/api/asynchronous/transcribers). A `whisper-large` model was one budapp
  offered and no Rev AI request could serve.
* Forced Alignment, Language Identification, Translation, Sentiment, Summarization and
  Topic Extraction are add-ons applied to a transcript, the same category as AWS's Call
  Analytics SKUs and Speechmatics' bolt-ons.
"""

from __future__ import annotations

import re

from ..base import AUDIO_TRANSCRIPTION, ScrapedModel, VendorScraper

#: Each model is one of these blocks.
_BLOCK = re.compile(r'<div class="payment-offering"')

#: "<h3 ...>Reverb Transcription</h3>"
_NAME = re.compile(r"<h3[^>]*>([^<]+)</h3>")

#: "<div class="heading-style-h3">$0.20</div>"
_PRICE = re.compile(r'heading-style-h3">\s*\$(\d+(?:\.\d+)?)\s*<')

#: "<div class="heading-style-h5 inline">per hour</div>"
_PER = re.compile(r'inline">\s*per\s+([a-z0-9 ]+?)\s*<', re.I)

#: "Rounded up to the nearest second, 15 second minimum"
_MINIMUM = re.compile(r"(\d+(?:\.\d+)?)\s+second\s+minimum", re.I)

#: Transcription offerings that are not a model a request can select: human transcription is
#: people, and Whisper Large has no `transcriber` value on either Rev AI API (module docstring).
_NOT_A_MODEL = {"human", "whisper-large"}

_PER_UNIT_SECONDS = {"hour": 3600.0, "minute": 60.0}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", name.strip().lower().replace(" ", "-"))


class RevAiScraper(VendorScraper):
    vendor = "revai"
    url = "https://www.rev.ai/pricing"
    # Reverb and Reverb Foreign Language.
    min_models = 2

    def extract(self, html: str) -> list[ScrapedModel]:
        starts = [m.start() for m in _BLOCK.finditer(html)]
        if not starts:
            raise ValueError("no 'payment-offering' blocks found; the page structure changed")

        bounds = list(zip(starts, starts[1:] + [len(html)], strict=True))
        out: list[ScrapedModel] = []
        for start, end in bounds:
            block = html[start:end]

            name_match = _NAME.search(block)
            price_match = _PRICE.search(block)
            per_match = _PER.search(block)
            if not (name_match and price_match and per_match):
                continue

            name = name_match.group(1).strip()
            # Only transcription models. Forced Alignment, Language Identification,
            # Translation, Sentiment, Summarization and Topic Extraction are add-ons
            # applied to a transcript -- the same category as AWS's Call Analytics SKUs
            # and Speechmatics' bolt-ons.
            if not name.endswith("Transcription"):
                continue
            key = _slug(name.removesuffix("Transcription"))
            if not key or key in _NOT_A_MODEL:
                continue

            per = per_match.group(1).strip().lower()
            divisor = _PER_UNIT_SECONDS.get(per)
            if divisor is None:
                # "per 10 words" and similar are real Rev AI units, but not ones the
                # catalog can express, so the model is skipped rather than mis-stored.
                continue

            minimum_match = _MINIMUM.search(block)
            minimum = float(minimum_match.group(1)) if minimum_match else None
            value = float(price_match.group(1))
            out.append(
                ScrapedModel(
                    model=key,
                    mode=AUDIO_TRANSCRIPTION,
                    unit="second",
                    rate=value / divisor,
                    published_as=f"${value:g} per {per}",
                    min_billable_units=minimum,
                    note=(
                        f"${value:g} per {per}; {value:g} / {divisor:g}."
                        + (f" Billed with a {minimum:g} second minimum." if minimum else "")
                    ),
                )
            )
        return out
