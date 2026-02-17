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

"""Immutable configuration for the catalog client."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogConfig:
    """Configuration for :class:`~bud_model_catalog.CatalogClient`.

    All fields have sensible defaults.  Validation is enforced at
    construction time via ``__post_init__`` — invalid values raise
    :class:`ValueError`.
    """

    litellm_url: str = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
    ai_models_url: str = "https://github.com/truefoundry/models/archive/refs/heads/main.zip"
    timeout: int = 30
    include_deprecated: bool = False
    max_retries: int = 2
    cache: bool = True

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        if not self.litellm_url.startswith(("http://", "https://")):
            raise ValueError(f"litellm_url must be an HTTP(S) URL, got {self.litellm_url!r}")
        if not self.ai_models_url.startswith(("http://", "https://")):
            raise ValueError(f"ai_models_url must be an HTTP(S) URL, got {self.ai_models_url!r}")
        if self.max_retries < 1:
            raise ValueError(f"max_retries must be at least 1, got {self.max_retries}")
