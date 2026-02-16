from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timezone

import yaml

from ..config import CatalogConfig
from ..exceptions import SourceFetchError
from .base import BaseSource, FetchResult

logger = logging.getLogger(__name__)

# Use C loader if available for ~10x speedup over pure-Python SafeLoader
_YAMLLoader: type[yaml.SafeLoader] = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

_MAX_RESPONSE_BYTES = 50 * 1024 * 1024  # 50 MB
_MAX_ENTRY_SIZE = 10 * 1024 * 1024  # 10 MB


class AiModelsSource(BaseSource):
    def __init__(self, config: CatalogConfig) -> None:
        super().__init__(config)

    async def fetch(self) -> FetchResult:
        headers: dict[str, str] = {}
        if self._config.cache and self._last_etag:
            headers["If-None-Match"] = self._last_etag

        # Download zipball with retry (single HTTP request, ~77KB, follows 302 redirect)
        response = await self._fetch_url(
            self._config.ai_models_url,
            headers,
            label="models archive",
            follow_redirects=True,
        )

        # ETag cache hit — return cached result
        if response.status_code == 304 and self._last_result is not None:
            logger.info("ai-models: 304 Not Modified, using cached result")
            return self._last_result

        # Guard against oversized responses / zip bombs
        if len(response.content) > _MAX_RESPONSE_BYTES:
            raise SourceFetchError(
                f"Response too large ({len(response.content)} bytes, limit {_MAX_RESPONSE_BYTES})"
            )

        # Extract + parse provider YAML files in-memory (never writes to disk)
        lookup: dict[tuple[str, str], dict] = {}
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                for name in zf.namelist():
                    # Match: models-main/providers/{provider}/{model}.yaml
                    # Skip: default.yaml (provider defaults, not individual models)
                    if not name.endswith(".yaml") or "/providers/" not in name:
                        continue
                    basename = name.rsplit("/", 1)[-1]
                    if basename == "default.yaml":
                        continue

                    # Skip oversized entries (zip bomb protection)
                    entry_size = zf.getinfo(name).file_size
                    if entry_size > _MAX_ENTRY_SIZE:
                        logger.warning("Skipping oversized entry %s (%d bytes)", name, entry_size)
                        continue

                    # Extract provider name from path
                    parts = name.split("/providers/", 1)[1]  # e.g. "anthropic/claude-opus-4-6.yaml"
                    provider_name = parts.split("/", 1)[0]

                    try:
                        with zf.open(name) as f:
                            model_data = yaml.load(f.read(), Loader=_YAMLLoader)
                    except yaml.YAMLError as e:
                        logger.warning("Skipping malformed YAML %s: %s", name, e)
                        continue

                    if not isinstance(model_data, dict) or "model" not in model_data:
                        continue

                    entry = {
                        "provider": provider_name,
                        "model": model_data["model"],
                        "costs": model_data.get("costs", {}),
                        "isDeprecated": model_data.get("isDeprecated", False),
                    }
                    lookup[(provider_name, model_data["model"])] = entry
        except zipfile.BadZipFile as e:
            raise SourceFetchError(f"Downloaded archive is not a valid ZIP: {e}") from e

        etag = response.headers.get("etag")
        fetched_at = datetime.now(timezone.utc)

        logger.info("ai-models: built lookup with %d entries from zipball", len(lookup))

        fetch_result = FetchResult(
            data=lookup,
            source_name="ai_models",
            fetched_at=fetched_at,
            etag=etag,
        )

        # Update cache
        if etag:
            self._last_etag = etag
            self._last_result = fetch_result

        return fetch_result
