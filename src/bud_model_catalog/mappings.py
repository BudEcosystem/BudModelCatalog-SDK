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

"""Shared provider mappings and model-name helpers.

Consumed by both the merger and individual source modules.
"""

from __future__ import annotations

# TensorZero litellm_provider -> ai-models research provider name
LITELLM_TO_RESEARCH: dict[str, str] = {
    "anthropic": "anthropic",
    "azure": "azure-openai",
    "bedrock": "aws-bedrock",
    "gemini": "google-gemini",
    "mistral": "mistral-ai",
    "moonshotai": "moonshot-ai",
    "openai": "openai",
    "together_ai": "together-ai",
    "vertex_ai-anthropic_models": "google-vertex",
    "vertex_ai-gemini-models": "google-vertex",
    "xai": "x-ai",
}

# Prefixes to strip from metadata.original_key per provider
STRIP_PREFIXES: dict[str, str] = {
    "azure": "azure/",
    "bedrock": "bedrock/",
    "deepseek": "deepseek/",
    "gemini": "gemini/",
    "mistral": "mistral/",
    "moonshotai": "moonshot/",
    "sagemaker": "sagemaker/",
    "together_ai": "together_ai/",
    "vertex_ai-gemini-models": "vertex_ai/",
    "xai": "xai/",
}


def extract_model_name(litellm_provider: str, original_key: str) -> str:
    """Extract model name from original_key for lookup in ai-models.

    Special handling:
    - vertex_ai-anthropic_models: prepend "anthropic/" if key starts with "vertex_ai/"
    - Others: strip provider-specific prefix if present
    """
    if litellm_provider == "vertex_ai-anthropic_models":
        if original_key.startswith("vertex_ai/"):
            return "anthropic/" + original_key[len("vertex_ai/") :]
        return original_key

    prefix = STRIP_PREFIXES.get(litellm_provider)
    if prefix and original_key.startswith(prefix):
        return original_key[len(prefix) :]

    return original_key


def strip_provider_prefix(tz_provider: str, original_key: str) -> str:
    """Strip LiteLLM provider prefix from original_key for URI construction."""
    prefix = STRIP_PREFIXES.get(tz_provider)
    if prefix and original_key.startswith(prefix):
        return original_key[len(prefix):]
    return original_key
