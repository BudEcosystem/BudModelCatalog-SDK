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

"""Data models returned by the catalog client."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class MergeStats(BaseModel):
    """Statistics from a single merge operation.

    Counts describe how the LiteLLM source was filtered and enriched
    with ai-models data during :func:`~bud_model_catalog.merger.merge`.
    """

    total_litellm: int
    total_output: int
    matched: int
    unmatched: int
    deprecated_removed: int
    cost_fields_updated: int


class CatalogResult(BaseModel):
    """Result of a catalog fetch-and-merge operation.

    Returned by :meth:`CatalogClient.fetch_catalog` and
    :func:`fetch_catalog`.
    """

    models: dict[str, dict[str, Any]]
    stats: MergeStats
    litellm_fetched_at: datetime
    ai_models_fetched_at: datetime | None = None
