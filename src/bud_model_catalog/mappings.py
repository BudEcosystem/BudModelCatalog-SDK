"""Shared provider mappings and model-name helpers.

Consumed by both the merger and individual source modules.
"""

from __future__ import annotations

# TensorZero litellm_provider -> ai-models research provider name
LITELLM_TO_RESEARCH: dict[str, str] = {
    "anthropic": "anthropic",
    "bedrock": "aws-bedrock",
    "azure": "azure-openai",
    "gemini": "google-gemini",
    "vertex_ai-gemini-models": "google-vertex",
    "vertex_ai-anthropic_models": "google-vertex",
    "openai": "openai",
    "xai": "x-ai",
    "together_ai": "together-ai",
    "mistral": "mistral-ai",
}

# Prefixes to strip from metadata.original_key per provider
STRIP_PREFIXES: dict[str, str] = {
    "azure": "azure/",
    "gemini": "gemini/",
    "vertex_ai-gemini-models": "vertex_ai/",
    "xai": "xai/",
    "mistral": "mistral/",
    "together_ai": "together_ai/",
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
