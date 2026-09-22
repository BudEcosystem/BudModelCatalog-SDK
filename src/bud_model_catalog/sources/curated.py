"""Hand-curated voice pricing, bundled with the package rather than fetched.

Every LLM-catalog aggregator shares one blind spot: dedicated speech vendors are not
OpenAI-compatible chat endpoints, so LiteLLM, models.dev, OpenRouter and Helicone all
list none of them.  bud-connect offers those vendors regardless, and a vendor offered
without a price cannot be billed -- so the rates are transcribed by hand from each
vendor's published pricing page, with the URL and the date recorded next to the number.

This source does no I/O and cannot fail at runtime the way a fetched source can, which
is the main reason it is a bundled file rather than a remote one: a pricing page is not
a stable API, and scraping one on every sync would make the nightly catalog refresh
depend on a vendor's marketing site.

The cost of that choice is staleness, which is why ``source.checked_on`` is mandatory
per entry: it is the only thing that distinguishes a rate confirmed this week from one
nobody has looked at in a year.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .base import FetchResult

logger = logging.getLogger(__name__)

DATA_PATH = Path(__file__).parent.parent / "data" / "curated_voice_pricing.yaml"


class CuratedSource:
    """Loads the bundled curated pricing file.

    Not a :class:`~.base.BaseSource`: that base exists to share HTTP fetching, ETag
    handling and retry behaviour, none of which apply to a file that ships inside the
    wheel.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or DATA_PATH

    def load(self) -> FetchResult:
        """Read the file and flatten it into catalog-key form.

        Returns:
            A FetchResult whose ``data`` maps ``{provider}/{model}`` to an entry carrying
            ``litellm_provider``, ``mode``, the normalised cost fields and a ``billing``
            block.

        Raises:
            ValueError: If an entry omits ``billing.source.checked_on``. A curated rate
                whose age cannot be established is worse than no rate: it looks as
                authoritative as a freshly confirmed one.
        """
        if not self._path.exists():
            logger.warning("Curated pricing file missing at %s; continuing without it", self._path)
            return FetchResult(
                data={}, source_name="curated", fetched_at=datetime.now(timezone.utc)
            )

        raw: dict[str, dict[str, Any]] = yaml.safe_load(self._path.read_text()) or {}

        flattened: dict[str, dict[str, Any]] = {}
        for provider, models in raw.items():
            for model, entry in models.items():
                billing = entry.get("billing") or {}
                checked_on = (billing.get("source") or {}).get("checked_on")
                if not checked_on:
                    raise ValueError(
                        f"curated entry {provider}/{model} has no billing.source.checked_on; "
                        "a rate nobody can date is a rate nobody can re-check"
                    )

                flattened[f"{provider}/{model}"] = {
                    **entry,
                    "litellm_provider": provider,
                    "metadata": {"original_key": model},
                }

        logger.info(
            "Loaded %d curated prices across %d vendors from %s",
            len(flattened),
            len(raw),
            self._path.name,
        )
        return FetchResult(
            data=flattened, source_name="curated", fetched_at=datetime.now(timezone.utc)
        )
