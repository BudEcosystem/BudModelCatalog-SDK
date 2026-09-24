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


"""Run the scrapers by hand and see what they read.

For development and for answering "why did this vendor's price change", not a scheduled
entry point -- bud-connect's own `cron-tensorzero-sync` binding already re-seeds every 24h
and picks up whatever this package extracts.

    python -m bud_model_catalog.scrapers              # every registered vendor
    python -m bud_model_catalog.scrapers speechmatics # one vendor
    python -m bud_model_catalog.scrapers --diff       # against the committed floor
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from ..config import CatalogConfig
from ..sources.curated import CuratedSource
from ..sources.scraped import ScrapedPricingSource
from .registry import SCRAPERS, all_scrapers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bud_model_catalog.scrapers")
    parser.add_argument("vendors", nargs="*", help=f"any of: {', '.join(sorted(SCRAPERS))}")
    parser.add_argument("--diff", action="store_true", help="compare against the committed floor")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show HTTP and rejection detail"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(message)s",
    )

    unknown = set(args.vendors) - set(SCRAPERS)
    if unknown:
        parser.error(f"unknown vendor(s): {', '.join(sorted(unknown))}")

    scrapers = [s for s in all_scrapers() if not args.vendors or s.vendor in args.vendors]
    result = asyncio.run(ScrapedPricingSource(CatalogConfig(), scrapers).fetch())

    floor = CuratedSource().load().data if args.diff else {}
    if not result.data:
        print("nothing extracted -- run with -v to see why")
        return 1

    for key in sorted(result.data):
        entry = result.data[key]
        billing = entry["billing"]
        rate = entry.get("input_cost_per_second") or entry.get("input_cost_per_character")
        line = f"  {key:38s} {rate:<14.8g} per {billing['unit']:<10s} {billing['confidence']}"
        if args.diff:
            committed = floor.get(key)
            if committed is None:
                line += "   NEW"
            else:
                field = (
                    "input_cost_per_second"
                    if billing["unit"] == "second"
                    else "input_cost_per_character"
                )
                old = committed.get(field)
                if old is None:
                    line += "   unit changed"
                elif abs(old - rate) > 1e-12:
                    line += f"   CHANGED from {old:.8g} ({rate / old:.2f}x)"
        print(line)

    if args.diff:
        missing = sorted(
            k
            for k in floor
            if k.split("/")[0] in {s.vendor for s in scrapers} and k not in result.data
        )
        for key in missing:
            # A model the file has and the page no longer lists. Worth a human look: it
            # could be a retired model or a parser that stopped matching a row.
            print(f"  {key:38s} NOT FOUND on the page any more")

    print(f"\n{len(result.data)} prices from {len(scrapers)} vendor(s)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
