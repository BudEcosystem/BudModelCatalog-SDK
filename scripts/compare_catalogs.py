#!/usr/bin/env python3
"""Compare SDK catalog output vs production seeder file."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_provider(model_data: dict) -> str:
    return model_data.get("litellm_provider", "unknown")


def compare_catalogs(sdk_path: str, seeder_path: str) -> None:
    sdk = load_json(sdk_path)
    seeder = load_json(seeder_path)

    sdk_keys = set(sdk.keys())
    seeder_keys = set(seeder.keys())

    only_in_sdk = sdk_keys - seeder_keys
    only_in_seeder = seeder_keys - sdk_keys
    common = sdk_keys & seeder_keys

    # ── 1. Summary counts ──────────────────────────────────────────
    print("=" * 70)
    print("CATALOG COMPARISON REPORT")
    print("=" * 70)
    print(f"SDK catalog models:        {len(sdk_keys)}")
    print(f"Seeder catalog models:     {len(seeder_keys)}")
    print(f"Models only in SDK:        {len(only_in_sdk)}")
    print(f"Models only in Seeder:     {len(only_in_seeder)}")
    print(f"Models in both:            {len(common)}")

    # ── 2. Classify common models ──────────────────────────────────
    identical = []
    changed = []
    change_details = {}  # model -> list of (field, old, new)

    for key in sorted(common):
        sdk_model = sdk[key]
        seeder_model = seeder[key]
        if sdk_model == seeder_model:
            identical.append(key)
        else:
            changed.append(key)
            diffs = []
            all_fields = set(sdk_model.keys()) | set(seeder_model.keys())
            for field in sorted(all_fields):
                sdk_val = sdk_model.get(field)
                seeder_val = seeder_model.get(field)
                if sdk_val != seeder_val:
                    diffs.append((field, seeder_val, sdk_val))
            change_details[key] = diffs

    print(f"  Identical:               {len(identical)}")
    print(f"  Changed:                 {len(changed)}")
    print()

    # ── 3. New models by provider ──────────────────────────────────
    print("-" * 70)
    print("MODELS ONLY IN SDK (new)")
    print("-" * 70)
    sdk_only_by_provider = defaultdict(list)
    for key in sorted(only_in_sdk):
        provider = get_provider(sdk[key])
        sdk_only_by_provider[provider].append(key)

    for provider in sorted(sdk_only_by_provider):
        models = sdk_only_by_provider[provider]
        print(f"\n  {provider} ({len(models)} models):")
        for m in models:
            print(f"    - {m}")
    print()

    # ── 4. Removed models by provider ─────────────────────────────
    print("-" * 70)
    print("MODELS ONLY IN SEEDER (missing from SDK)")
    print("-" * 70)
    seeder_only_by_provider = defaultdict(list)
    for key in sorted(only_in_seeder):
        provider = get_provider(seeder[key])
        seeder_only_by_provider[provider].append(key)

    for provider in sorted(seeder_only_by_provider):
        models = seeder_only_by_provider[provider]
        print(f"\n  {provider} ({len(models)} models):")
        for m in models:
            print(f"    - {m}")
    print()

    # ── 5. Field change frequency ─────────────────────────────────
    print("-" * 70)
    print("FIELD CHANGE FREQUENCY (across changed models)")
    print("-" * 70)
    field_counter = Counter()
    for diffs in change_details.values():
        for field, _, _ in diffs:
            field_counter[field] += 1

    for field, count in field_counter.most_common():
        print(f"  {field:40s} {count:>5d} models")
    print()

    # ── 6. Fields added/removed in SDK vs seeder ──────────────────
    print("-" * 70)
    print("FIELDS ADDED IN SDK (present in SDK but not seeder for common models)")
    print("-" * 70)
    fields_added = Counter()
    fields_removed = Counter()
    for diffs in change_details.values():
        for field, seeder_val, sdk_val in diffs:
            if seeder_val is None and sdk_val is not None:
                fields_added[field] += 1
            elif seeder_val is not None and sdk_val is None:
                fields_removed[field] += 1

    for field, count in fields_added.most_common():
        print(f"  {field:40s} {count:>5d} models")

    print()
    print("-" * 70)
    print("FIELDS REMOVED IN SDK (present in seeder but not SDK for common models)")
    print("-" * 70)
    for field, count in fields_removed.most_common():
        print(f"  {field:40s} {count:>5d} models")
    print()

    # ── 7. Cost field differences ─────────────────────────────────
    cost_fields = [
        "input_cost_per_token",
        "output_cost_per_token",
        "input_cost_per_pixel",
        "output_cost_per_pixel",
        "input_cost_per_second",
        "output_cost_per_second",
        "input_cost_per_character",
        "output_cost_per_character",
        "input_cost_per_image",
        "output_cost_per_image",
        "input_cost_per_audio_token",
        "output_cost_per_audio_token",
        "cache_creation_input_token_cost",
        "cache_read_input_token_cost",
    ]

    print("-" * 70)
    print("COST FIELD DIFFERENCES (before → after)")
    print("-" * 70)
    cost_changes = []
    for model_key, diffs in sorted(change_details.items()):
        for field, seeder_val, sdk_val in diffs:
            if field in cost_fields:
                cost_changes.append((model_key, field, seeder_val, sdk_val))

    if cost_changes:
        # Group by field
        by_field = defaultdict(list)
        for model_key, field, old, new in cost_changes:
            by_field[field].append((model_key, old, new))

        for field in sorted(by_field):
            entries = by_field[field]
            print(f"\n  {field} ({len(entries)} changes):")
            for model_key, old, new in entries:
                print(f"    {model_key}")
                print(f"      seeder: {old}  →  sdk: {new}")
    else:
        print("  No cost field differences found.")
    print()

    # ── 8. Sample changed models (detailed) ───────────────────────
    print("-" * 70)
    print("SAMPLE CHANGED MODELS (first 20)")
    print("-" * 70)
    for model_key in changed[:20]:
        diffs = change_details[model_key]
        print(f"\n  {model_key}:")
        for field, seeder_val, sdk_val in diffs:
            print(f"    {field}:")
            print(f"      seeder: {seeder_val}")
            print(f"      sdk:    {sdk_val}")
    print()

    # ── 9. Provider summary table ─────────────────────────────────
    print("-" * 70)
    print("PROVIDER SUMMARY")
    print("-" * 70)
    all_providers = set()
    provider_stats = defaultdict(lambda: {"sdk_only": 0, "seeder_only": 0, "common": 0, "changed": 0})

    for key in only_in_sdk:
        p = get_provider(sdk[key])
        all_providers.add(p)
        provider_stats[p]["sdk_only"] += 1
    for key in only_in_seeder:
        p = get_provider(seeder[key])
        all_providers.add(p)
        provider_stats[p]["seeder_only"] += 1
    for key in common:
        p = get_provider(sdk[key])
        all_providers.add(p)
        provider_stats[p]["common"] += 1
    for key in changed:
        p = get_provider(sdk[key])
        provider_stats[p]["changed"] += 1

    header = f"  {'Provider':30s} {'SDK-only':>9s} {'Seeder-only':>12s} {'Common':>7s} {'Changed':>8s}"
    print(header)
    print("  " + "-" * len(header.strip()))
    for p in sorted(all_providers):
        s = provider_stats[p]
        print(
            f"  {p:30s} {s['sdk_only']:>9d} {s['seeder_only']:>12d} "
            f"{s['common']:>7d} {s['changed']:>8d}"
        )
    print()


if __name__ == "__main__":
    sdk_path = sys.argv[1] if len(sys.argv) > 1 else "catalog.json"
    seeder_path = sys.argv[2] if len(sys.argv) > 2 else "tensorzero_v_0_1_0.json"
    compare_catalogs(sdk_path, seeder_path)
