from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class MergeStats(BaseModel):
    total_litellm: int
    total_output: int
    matched: int
    unmatched: int
    deprecated_removed: int
    cost_fields_updated: int


class CatalogResult(BaseModel):
    models: dict[str, dict[str, Any]]
    stats: MergeStats
    litellm_fetched_at: datetime
    ai_models_fetched_at: datetime | None = None
