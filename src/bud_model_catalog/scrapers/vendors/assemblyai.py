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


"""AssemblyAI: the async (pre-recorded) speech-to-text models.

LiteLLM's two AssemblyAI entries are `best` and `nano`, which AssemblyAI no longer
accepts: its transcript API takes `speech_models` from `universal-3-5-pro` and
`universal-2` (assemblyai.com/docs/api-reference/transcripts/submit, read 2026-09-24). Their
prices were stale too -- `nano` at $0.37/hr sat above what the current top model costs.

WHICH TABLE

The page has one "Models / Pay as you go" block per product: the async API, streaming
(Universal-Streaming, Universal-3.5 Pro Realtime), a Sync API, a Dictation API and a Voice
Agent API. WaaV sends AssemblyAI transcription to the async transcript API, so the async
block is the one read, and it is identified by its content -- the block that names
Universal-2, which no other block does -- rather than by being first.

Add-on features (keyterms, diarization, medical mode) are surcharges on a transcript, not
models, and are excluded the same way Deepgram's and AWS's are.
"""

from __future__ import annotations

import html as html_mod
import re

from ..base import AUDIO_TRANSCRIPTION, ScrapedModel, VendorScraper

#: Label on the page -> the `speech_models` value the API accepts.
MODELS: dict[str, str] = {
    "Universal-3.5 Pro": "universal-3-5-pro",
    "Universal-2": "universal-2",
}

_BLOCK = re.compile(r"Models Pay as you go(.*?)(?=Add-on features|Models Pay as you go|\Z)", re.S)

#: A model label, NOT followed by "Realtime" (the streaming product shares the prefix), then
#: its description, then the first per-hour price.
_ROW = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in MODELS) + r")\b(?!\s+Realtime)[^$]{0,500}?"
    r"\$(\d+(?:\.\d+)?)\s*/\s*hr"
)

_SECONDS_PER_HOUR = 3600.0


class AssemblyAiScraper(VendorScraper):
    vendor = "assemblyai"
    url = "https://www.assemblyai.com/pricing"
    min_models = len(MODELS)

    def extract(self, html: str) -> list[ScrapedModel]:
        text = re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", " ", html)))

        blocks = [b for b in _BLOCK.findall(text) if "Universal-2" in b]
        if not blocks:
            raise ValueError(
                "no async 'Models / Pay as you go' block naming Universal-2; page changed"
            )

        out: list[ScrapedModel] = []
        for block in blocks:
            for label, amount in _ROW.findall(block):
                per_hour = float(amount)
                out.append(
                    ScrapedModel(
                        model=MODELS[label],
                        mode=AUDIO_TRANSCRIPTION,
                        unit="second",
                        rate=per_hour / _SECONDS_PER_HOUR,
                        published_as=f"${per_hour:g}/hr",
                        note=f"'{label}' async, Pay as you go ${per_hour:g}/hr; {per_hour:g} / 3600.",
                    )
                )
        return out
