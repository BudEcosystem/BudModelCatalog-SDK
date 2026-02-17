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

"""Abstract base class and shared helpers for data sources."""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import httpx

from ..config import CatalogConfig
from ..exceptions import SourceFetchError

logger = logging.getLogger(__name__)

_MAX_BACKOFF_SECONDS = 30


@dataclass
class FetchResult:
    """Container for the data returned by a single source fetch.

    ``data`` holds the parsed payload (shape varies by source), and
    ``etag`` is stored for conditional-GET caching on subsequent calls.
    """

    data: dict
    source_name: str
    fetched_at: datetime
    etag: str | None = None


class BaseSource(ABC):
    """Abstract base for catalog data sources.

    Subclasses must implement :meth:`fetch` to retrieve and parse their
    upstream data.  The shared :meth:`_fetch_url` helper provides retry
    logic with jittered exponential back-off and ETag support.
    """

    def __init__(self, config: CatalogConfig) -> None:
        self._config = config
        self._last_etag: str | None = None
        self._last_result: FetchResult | None = None

    async def _fetch_url(
        self,
        url: str,
        headers: dict[str, str],
        *,
        label: str,
        follow_redirects: bool = False,
    ) -> httpx.Response:
        """Fetch *url* with retry, jitter back-off, and a single connection pool.

        Returns the final :class:`httpx.Response` (caller handles 304 etc.).
        Raises :class:`SourceFetchError` after *max_retries* failures.
        """
        async with httpx.AsyncClient(
            timeout=self._config.timeout, follow_redirects=follow_redirects
        ) as client:
            last_exc: Exception | None = None
            for attempt in range(1, self._config.max_retries + 1):
                try:
                    response = await client.get(url, headers=headers)
                    if response.status_code == 304:
                        return response
                    response.raise_for_status()
                    return response
                except (httpx.HTTPStatusError, httpx.RequestError) as e:
                    last_exc = e
                    if attempt == self._config.max_retries:
                        break
                    wait = min(random.uniform(0, 2**attempt), _MAX_BACKOFF_SECONDS)
                    logger.warning(
                        "%s fetch attempt %d failed, retrying in %.1fs: %s",
                        label,
                        attempt,
                        wait,
                        e,
                    )
                    await asyncio.sleep(wait)

        raise SourceFetchError(f"Failed to fetch {label}: {last_exc}") from last_exc

    @abstractmethod
    async def fetch(self) -> FetchResult:
        """Fetch and parse data from the upstream source.

        Implementations should use :meth:`_fetch_url` for HTTP requests
        and return a :class:`FetchResult` with the parsed data.
        """
        ...
