# bud-model-catalog

Multi-source LLM model catalog with cost-accurate pricing. Fetches model metadata from [LiteLLM](https://github.com/BerriAI/litellm) and [truefoundry/models](https://github.com/truefoundry/models), merges them with cost-accurate pricing, filters deprecated models, and returns a unified catalog keyed by TensorZero provider/model.

## Install

```bash
pip install bud-model-catalog
```

## Quick Start

```python
from bud_model_catalog import CatalogClient

client = CatalogClient()
result = client.fetch_catalog_sync()

# Inspect merge statistics
print(result.stats)
# MergeStats(total_litellm=847, total_output=812, matched=342, ...)

# Look up a specific model's pricing
model = result.models["openai/gpt-4o"]
print(f"Input:  ${model['input_cost_per_token']}/token")
print(f"Output: ${model['output_cost_per_token']}/token")
```

## Async Usage

```python
import asyncio
from bud_model_catalog import CatalogClient, CatalogConfig

async def main():
    config = CatalogConfig(include_deprecated=True, timeout=60)
    client = CatalogClient(config)
    result = await client.fetch_catalog()

    # Filter to a single provider
    anthropic_models = {
        key: model for key, model in result.models.items()
        if key.startswith("anthropic/")
    }
    for key, model in list(anthropic_models.items())[:3]:
        print(f"{key}: ${model.get('input_cost_per_token', 'N/A')}/token")

asyncio.run(main())
```

## Jupyter / Notebook

`fetch_catalog_sync()` works inside Jupyter's running event loop:

```python
# In a Jupyter notebook cell:
from bud_model_catalog import CatalogClient

client = CatalogClient()
result = client.fetch_catalog_sync()  # works inside Jupyter's event loop
print(f"{len(result.models)} models loaded")

# Or use await directly in a notebook cell:
result = await client.fetch_catalog()
```

## Convenience Function

```python
from bud_model_catalog import fetch_catalog

# One-shot fetch (no ETag caching across calls)
result = await fetch_catalog()
```

## Configuration

All options are passed via `CatalogConfig`:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `litellm_url` | `str` | GitHub raw URL | URL to the LiteLLM model prices JSON |
| `ai_models_url` | `str` | GitHub archive URL | URL to the truefoundry/models ZIP archive |
| `timeout` | `int` | `30` | HTTP request timeout in seconds (must be > 0) |
| `include_deprecated` | `bool` | `False` | Whether to include deprecated models in output |
| `max_retries` | `int` | `2` | Maximum retry attempts per HTTP request (with exponential backoff) |
| `cache` | `bool` | `True` | Enable ETag-based conditional GET caching across calls |

```python
from bud_model_catalog import CatalogConfig

config = CatalogConfig(
    timeout=60,
    include_deprecated=True,
    max_retries=3,
    cache=True,
)
```

Validation is enforced at construction time:

```python
CatalogConfig(timeout=-1)   # ValueError: timeout must be positive
CatalogConfig(litellm_url="not-a-url")  # ValueError: must be an HTTP(S) URL
```

## Error Handling

```python
from bud_model_catalog import CatalogClient, CatalogConfig, SourceFetchError

try:
    result = CatalogClient().fetch_catalog_sync()
except SourceFetchError as e:
    print(f"Failed to fetch data: {e}")
```

- `SourceFetchError` — raised when LiteLLM fetch fails (HTTP error, invalid JSON, timeout)
- ai-models failures are handled gracefully — the SDK falls back to LiteLLM-only costs

## API Reference

### `CatalogClient`

Main entry point for fetching the catalog.

- `CatalogClient(config=None)` — create a client with optional `CatalogConfig`
- `await client.fetch_catalog()` — async fetch, returns `CatalogResult`
- `client.fetch_catalog_sync()` — blocking sync wrapper (works in Jupyter, async frameworks, and plain scripts)

### `CatalogResult`

Pydantic model returned from fetch operations.

- `models: dict[str, dict]` — merged model catalog keyed by `{provider}/{model}`
- `stats: MergeStats` — merge statistics
- `litellm_fetched_at: datetime` — timestamp of LiteLLM fetch
- `ai_models_fetched_at: datetime | None` — timestamp of ai-models fetch (None if failed/skipped)

### `MergeStats`

- `total_litellm` — total models from LiteLLM source
- `total_output` — models in final output
- `matched` — models matched with ai-models data
- `unmatched` — models without ai-models match
- `deprecated_removed` — models filtered as deprecated
- `cost_fields_updated` — individual cost field values updated from ai-models

## Logging

The SDK uses Python's `logging` module. Enable output to see fetch/merge details:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

Key log messages:
- `INFO` — fetch counts, merge statistics, cache hits
- `WARNING` — ai-models fallback, malformed YAML files skipped, retry attempts

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest -v

# Lint
ruff check src/ tests/
```
