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

"""Bud Model Catalog SDK — multi-source LLM catalog with cost-accurate pricing.

Fetches model metadata from LiteLLM and truefoundry/models, merges them with
cost-accurate pricing, filters deprecated models, and returns a unified catalog
keyed by TensorZero provider/model.
"""

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
