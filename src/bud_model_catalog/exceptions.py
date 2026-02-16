class CatalogError(Exception):
    """Base exception for bud-model-catalog."""


class SourceFetchError(CatalogError):
    """HTTP or parse errors from a data source."""
