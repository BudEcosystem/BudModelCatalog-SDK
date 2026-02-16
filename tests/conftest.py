import io
import zipfile

import pytest
import yaml

# Sample LiteLLM model data (subset of real structure)
SAMPLE_LITELLM_DATA = {
    "gpt-4o": {
        "max_tokens": 16384,
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
        "input_cost_per_token": 5e-06,
        "output_cost_per_token": 1.5e-05,
        "cache_read_input_token_cost": 2.5e-06,
        "litellm_provider": "openai",
        "mode": "chat",
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_system_messages": True,
    },
    "claude-sonnet-4-20250514": {
        "max_tokens": 8192,
        "max_input_tokens": 200000,
        "max_output_tokens": 8192,
        "input_cost_per_token": 3e-06,
        "output_cost_per_token": 1.5e-05,
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_system_messages": True,
    },
    "gemini/gemini-2.0-flash": {
        "max_tokens": 8192,
        "max_input_tokens": 1048576,
        "max_output_tokens": 8192,
        "input_cost_per_token": 1e-07,
        "output_cost_per_token": 4e-07,
        "litellm_provider": "gemini",
        "mode": "chat",
        "supports_function_calling": True,
        "supports_vision": True,
        "supports_system_messages": True,
    },
    "some-unknown-provider-model": {
        "max_tokens": 4096,
        "litellm_provider": "unknown_provider",
        "mode": "chat",
    },
    "vertex_ai/gemini-1.5-pro": {
        "max_tokens": 8192,
        "max_input_tokens": 2097152,
        "max_output_tokens": 8192,
        "input_cost_per_token": 1.25e-06,
        "output_cost_per_token": 5e-06,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "chat",
    },
    "vertex_ai/text-bison": {
        "max_tokens": 1024,
        "litellm_provider": "vertex_ai-language-models",
        "mode": "completion",
    },
    "no-provider-model": {
        "max_tokens": 4096,
        "mode": "chat",
    },
    "sample_spec": {
        "sample_key": "sample_value",
    },
}

# Sample ai-models entries (provider YAML file contents)
SAMPLE_AI_MODELS_DATA = [
    {
        "provider": "openai",
        "model": "gpt-4o",
        "costs": {
            "input_cost_per_token": 2.5e-06,
            "output_cost_per_token": 1e-05,
            "cache_read_input_token_cost": 1.25e-06,
        },
        "isDeprecated": False,
    },
    {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "costs": {
            "input_cost_per_token": 3e-06,
            "output_cost_per_token": 1.5e-05,
        },
        "isDeprecated": False,
    },
    {
        "provider": "google-gemini",
        "model": "gemini-2.0-flash",
        "costs": {
            "input_cost_per_token": 7.5e-08,
            "output_cost_per_token": 3e-07,
        },
        "isDeprecated": False,
    },
    {
        "provider": "openai",
        "model": "gpt-3.5-turbo-0301",
        "costs": {
            "input_cost_per_token": 1.5e-06,
            "output_cost_per_token": 2e-06,
        },
        "isDeprecated": True,
    },
]


def build_ai_models_zip(entries: list[dict] | None = None) -> bytes:
    """Build an in-memory ZIP archive mimicking the truefoundry/models repo structure."""
    if entries is None:
        entries = SAMPLE_AI_MODELS_DATA
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for entry in entries:
            # Replicate: models-main/providers/{provider}/{model}.yaml
            path = f"models-main/providers/{entry['provider']}/{entry['model']}.yaml"
            yaml_content = yaml.dump(entry, default_flow_style=False)
            zf.writestr(path, yaml_content)
    return buf.getvalue()


@pytest.fixture
def sample_litellm_data():
    return dict(SAMPLE_LITELLM_DATA)


@pytest.fixture
def sample_ai_models_data():
    return list(SAMPLE_AI_MODELS_DATA)
