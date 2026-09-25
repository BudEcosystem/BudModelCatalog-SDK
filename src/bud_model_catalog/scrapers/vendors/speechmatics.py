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


"""Speechmatics: the pricing table is embedded in the page as JSON.

Worth stating why this adapter reads the JSON and not the rendered table, because the two
disagree. The JSON rows carry LIST prices; the visible table shows a promotion roughly 46%
lower (Real-time Standard renders as $0.24/hr against a $0.45 list). List is what gets
stored, for the same reason AWS's tier-0 rate is: a promotion expires, and a cost estimate
that silently assumed one is wrong the day it ends.

One exception inside the JSON itself: a row can carry its own discount. Linden 1's row has
`"Pro Plan value":"0.30"` and `"Crossed-out price":"0.40"`, with a tooltip saying it is
"discounted 25% from a $0.40/hr list rate". There the crossed-out figure is the list price
and the plan value is the promotion, so the adapter reads the pair rather than assuming the
plan value is always list.

The page's rows are pricing LABELS ("Real-time Standard", "Batch Melia 1"), not model ids,
and :data:`ROWS` maps the ones WaaV can reach to the id Speechmatics' Realtime API accepts.
Per docs.speechmatics.com/speech-to-text/models: Batch takes enhanced, standard and melia-1;
Realtime takes enhanced and standard; Agent STT (`/v2/agent`) takes linden-1. WaaV calls the
Realtime API only, so the Batch rows and Linden price nothing it can dispatch.

The bolt-ons on the same page (Translation, Summaries, Chapters, Sentiment, Topics) are
per-feature surcharges rather than models, and are skipped for the same reason AWS's Call
Analytics and Redaction SKUs are: they are not something a user deploys.
"""

from __future__ import annotations

import html as html_mod
import json
import logging
import re

from ..base import AUDIO_SPEECH, AUDIO_TRANSCRIPTION, ScrapedModel, VendorScraper

logger = logging.getLogger(__name__)

#: Pricing rows are emitted as standalone JSON objects inside the page source.
_ROW = re.compile(r'\{"Section":"Pricing".*?\}')

#: The rendered table also shows a discounted figure per model, e.g. "Real-time Standard -
#: $0.24/hr". Captured best-effort so the gap between list and promotion stays visible.
_PROMO = re.compile(
    r"([A-Za-z0-9 \-]+?)\s*-\s*(?:Previous price: \$[\d.]+/hr Current price: )?\$([\d.]+)/hr"
)

_SECONDS_PER_HOUR = 3600.0

#: Pricing row (by :func:`_slug` of its "Line item") -> the catalog model id it prices.
#: `standard` and `enhanced` are what the Realtime API accepts; `real-time-standard`, which
#: this adapter used to publish, is a label no endpoint does.
ROWS: dict[str, str] = {
    "real-time-standard": "standard",
    "real-time-enhanced": "enhanced",
    "text-to-speech": "text-to-speech",
}

#: Billable rows that deliberately map to no model, with the reason. A row in neither this
#: nor :data:`ROWS` is new since the adapter was written, and is logged for a person to map.
ROWS_SKIPPED: dict[str, str] = {
    "batch-standard": "Batch API only; WaaV calls the Realtime API",
    "batch-enhanced": "Batch API only; WaaV calls the Realtime API",
    "batch-melia-1": "melia-1 is Batch-only; WaaV calls the Realtime API",
    "linden-1": "Agent STT API (/v2/agent) only; WaaV calls the Realtime API (/v2)",
}


def _slug(line_item: str) -> str:
    """Normalise a row's "Line item", e.g. "Real-time Standard" -> `real-time-standard`.

    This names the pricing ROW, not a model; :data:`ROWS` is what maps it to one.
    """
    # `.` survives: "Linden 1.1" must not become `linden-11`, a different row's name.
    return re.sub(r"[^a-z0-9.-]", "", line_item.strip().lower().replace(" ", "-"))


class SpeechmaticsScraper(VendorScraper):
    vendor = "speechmatics"
    url = "https://www.speechmatics.com/pricing"
    # Two Realtime STT models plus the TTS row. Below this the table has changed shape.
    min_models = len(ROWS)

    def extract(self, html: str) -> list[ScrapedModel]:
        rows = []
        for m in _ROW.finditer(html):
            try:
                rows.append(json.loads(m.group(0)))
            except json.JSONDecodeError:
                # A row that will not parse is skipped rather than fatal: the page carries
                # unrelated JSON too, and min_models is what catches losing real rows.
                continue
        if not rows:
            raise ValueError("no pricing rows found in page source; the page format changed")

        # Pass one: the billable rows, in page order, with bolt-ons dropped. Bolt-ons
        # (Translation, Summaries, Chapters, Sentiment, Topics) are per-feature surcharges
        # on a transcript rather than models a user deploys, so they are excluded for the
        # same reason AWS's Call Analytics and Redaction SKUs are.
        billable: list[tuple[str, str, str, str, str]] = []
        for row in rows:
            category = (row.get("Category") or "").strip()
            line_item = (row.get("Line item") or "").strip()
            value = (row.get("Pro Plan value") or "").strip()
            crossed = (row.get("Crossed-out price") or "").strip()
            if not line_item or not value or "bolt-on" in category.lower():
                continue
            # Every row is returned, repeats included: validate_batch collapses a repeat that
            # agrees and rejects one that does not, which is the only safe reading of a page
            # that lists one model at two prices.
            billable.append((_slug(line_item), line_item, category, value, crossed))

        # Promotions are attributed over every billable row, mapped or not, so a skipped
        # row's figure can never be mistaken for a kept row's.
        promos = _promotional_rates(html, {row for row, _, _, _, _ in billable})

        out: list[ScrapedModel] = []
        for row, line_item, category, value, crossed in billable:
            key = ROWS.get(row)
            if key is None:
                if row not in ROWS_SKIPPED:
                    logger.warning(
                        "Speechmatics pricing row %r (%s) maps to no model; add it to ROWS "
                        "or ROWS_SKIPPED",
                        line_item,
                        category,
                    )
                continue
            if category == "Speech-to-Text models":
                # Bare numbers, in dollars per hour.
                try:
                    per_hour = float(value)
                except ValueError:
                    continue
                promo = promos.get(row)
                crossed_rate = _as_float(crossed)
                if crossed_rate is not None and crossed_rate > per_hour:
                    # The row's own discount: the struck-through figure is list.
                    per_hour, promo = crossed_rate, per_hour
                discounted = promo if (promo is not None and promo < per_hour) else None
                out.append(
                    ScrapedModel(
                        model=key,
                        mode=AUDIO_TRANSCRIPTION,
                        unit="second",
                        rate=per_hour / _SECONDS_PER_HOUR,
                        published_as=f"${per_hour:g}/hr",
                        promotional_rate=(discounted / _SECONDS_PER_HOUR)
                        if discounted is not None
                        else None,
                        note=f"{line_item!r} row. List ${per_hour:g}/hr; {per_hour:g} / 3600."
                        + (
                            f" Page currently displays a promotional ${discounted:g}/hr."
                            if discounted
                            else ""
                        ),
                    )
                )
            elif category == "Text-to-Speech":
                # Quoted with its unit inline, e.g. "$0.011/1k characters".
                quoted = re.search(r"\$([\d.]+)\s*/\s*1k characters", value)
                if not quoted:
                    continue
                per_1k = float(quoted.group(1))
                out.append(
                    ScrapedModel(
                        model=key,
                        mode=AUDIO_SPEECH,
                        unit="character",
                        rate=per_1k / 1000.0,
                        published_as=f"${per_1k:g}/1k characters",
                        note=f"${per_1k:g} per 1k characters on the Pro plan; {per_1k:g} / 1000.",
                    )
                )
        return out


def _as_float(value: str) -> float | None:
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _promotional_rates(html: str, known: set[str]) -> dict[str, float]:
    """Find the discounted per-model figures in the rendered table.

    Best-effort and deliberately separate from the list prices above: a promotion is
    recorded so the gap is visible, never used as the rate.

    The label capture absorbs whatever text precedes the model name in the same cell -- the
    page renders rows as "Speech-to-Text models Batch Melia 1 - $0.129/hr" in one place and
    " Batch Melia 1 - $0.129/hr" in another -- so a match is attributed by asking which
    known model key the captured label *ends with*, rather than by equality.
    """
    text = re.sub(r"\s+", " ", html_mod.unescape(re.sub(r"<[^>]+>", " ", html)))
    found: dict[str, float] = {}
    for label, rate in _PROMO.findall(text):
        slug = _slug(label)
        for key in known:
            if slug == key or slug.endswith(f"-{key}"):
                found.setdefault(key, float(rate))
                break
    return found
