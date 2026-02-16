from .client import CatalogClient
from .config import CatalogConfig
from .exceptions import CatalogError, SourceFetchError
from .models import CatalogResult, MergeStats

__all__ = [
    "CatalogClient",
    "CatalogConfig",
    "CatalogResult",
    "MergeStats",
    "CatalogError",
    "SourceFetchError",
    "fetch_catalog",
]


async def fetch_catalog(config: CatalogConfig | None = None) -> CatalogResult:
    """One-shot convenience helper.

    Creates a new :class:`CatalogClient` on every call, so ETag caching
    is **not** available across invocations.  For repeated fetches, create
    a :class:`CatalogClient` once and reuse it.
    """
    client = CatalogClient(config)
    return await client.fetch_catalog()
