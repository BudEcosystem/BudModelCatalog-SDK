"""Example: fetch and inspect the Bud model catalog."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

from bud_model_catalog import CatalogClient, CatalogConfig, SourceFetchError


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch the Bud model catalog and print a summary.")
    parser.add_argument("--output", "-o", help="Write the full model catalog to a JSON file.")
    parser.add_argument(
        "--include-deprecated",
        action="store_true",
        default=False,
        help="Include deprecated models in the catalog (default: exclude them).",
    )
    args = parser.parse_args()

    # ── 1. Fetch catalog with CLI-driven config ──────────────────────────
    config = CatalogConfig(include_deprecated=args.include_deprecated)
    print(f"Fetching catalog (include_deprecated={args.include_deprecated}) ...")
    client = CatalogClient(config)

    try:
        result = client.fetch_catalog_sync()
    except SourceFetchError as exc:
        print(f"Failed to fetch catalog: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── 2. Inspect results ──────────────────────────────────────────────
    stats = result.stats
    print(f"\nTotal models in catalog : {stats.total_output}")
    print(f"LiteLLM source models  : {stats.total_litellm}")
    print(f"Matched with ai-models : {stats.matched}")
    print(f"Unmatched              : {stats.unmatched}")
    print(f"Deprecated removed     : {stats.deprecated_removed}")
    print(f"Cost fields updated    : {stats.cost_fields_updated}")
    print(f"\nLiteLLM fetched at     : {result.litellm_fetched_at}")
    print(f"AI-models fetched at   : {result.ai_models_fetched_at}")

    # ── 3. Browse by provider ───────────────────────────────────────────
    provider_counts: Counter[str] = Counter()
    for info in result.models.values():
        provider_counts[info.get("litellm_provider", "unknown")] += 1

    print(f"\n{'Provider':<30} {'Models':>6}")
    print("-" * 38)
    for provider, count in provider_counts.most_common(10):
        print(f"{provider:<30} {count:>6}")
    if len(provider_counts) > 10:
        print(f"... and {len(provider_counts) - 10} more providers")

    # ── 4. Inspect a single model ───────────────────────────────────────
    first_key = next(iter(result.models))
    model = result.models[first_key]
    print(f"\nSample model: {first_key}")
    for field in ("litellm_provider", "max_tokens", "max_input_tokens", "max_output_tokens",
                  "input_cost_per_token", "output_cost_per_token"):
        if field in model:
            print(f"  {field}: {model[field]}")

    # ── 5. Optional JSON dump ────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(result.models, fh, indent=2, default=str)
        print(f"\nCatalog written to {args.output}")


if __name__ == "__main__":
    main()
