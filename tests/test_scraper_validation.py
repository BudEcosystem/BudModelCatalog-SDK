"""The gates between a pricing page changing and bud-connect billing from it.

Nothing reviews a scraped rate before it is served, so these gates are the only guard.
Each test below is a failure scraping actually produces, not a hypothetical: a unit left
unconverted, a decimal in the wrong place, a regex that matched the plan price column, a
page that started rendering in JavaScript, a table that gained a duplicate row.
"""

import math

import pytest

from bud_model_catalog.scrapers.base import ScrapedModel
from bud_model_catalog.scrapers.validate import (
    MAX_DRIFT_FACTOR,
    UNIT_BOUNDS,
    exceeds_drift,
    validate_batch,
    validate_model,
)


def model(**kw) -> ScrapedModel:
    base = {
        "model": "batch-standard",
        "mode": "audio_transcription",
        "unit": "second",
        "rate": 1.25e-04,
        "published_as": "$0.45/hr",
    }
    base.update(kw)
    return ScrapedModel(**base)  # type: ignore[arg-type]


def test_a_correct_model_passes():
    assert validate_model(model()) is None


# --------------------------------------------------------------------------------- #
# rates that are not numbers, or not finite
# --------------------------------------------------------------------------------- #


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rates_are_rejected(rate):
    """NaN is the dangerous one: it compares False against every bound.

    Without an explicit finiteness check a NaN rate passes every range test and lands in
    the catalog as a price.
    """
    assert validate_model(model(rate=rate)) is not None


@pytest.mark.parametrize("rate", [0, 0.0, -1e-05])
def test_non_positive_rates_are_rejected(rate):
    """Zero is a request billed as free, and it looks unremarkable in a table."""
    assert "not positive" in validate_model(model(rate=rate))


@pytest.mark.parametrize("rate", ["0.00012", None, True])
def test_non_numeric_rates_are_rejected(rate):
    """`True` is included on purpose: it is an int in Python and would pass `> 0`."""
    assert "not a number" in validate_model(model(rate=rate))


# --------------------------------------------------------------------------------- #
# the unit-conversion failures, which are the ones that actually happen
# --------------------------------------------------------------------------------- #


def test_a_per_hour_rate_stored_as_per_second_is_rejected():
    """$0.45/hr left unconverted is 3600x too high, and this is what catches it."""
    reason = validate_model(model(rate=0.45))
    assert "outside the plausible range" in reason
    assert "check the unit conversion" in reason


def test_a_per_1k_character_rate_stored_as_per_character_is_rejected():
    reason = validate_model(model(unit="character", rate=0.011, published_as="$0.011/1k chars"))
    assert "outside the plausible range" in reason


@pytest.mark.parametrize("unit", sorted(UNIT_BOUNDS))
def test_the_bounds_themselves_are_inclusive(unit):
    low, high = UNIT_BOUNDS[unit]
    assert validate_model(model(unit=unit, rate=low)) is None
    assert validate_model(model(unit=unit, rate=high)) is None
    assert validate_model(model(unit=unit, rate=low / 1.0001)) is not None
    assert validate_model(model(unit=unit, rate=high * 1.0001)) is not None


# --------------------------------------------------------------------------------- #
# keys, modes, units, confidence
# --------------------------------------------------------------------------------- #


def test_a_slash_in_the_model_name_is_rejected():
    """The catalog key is "{vendor}/{model}"; a slash silently reparents the entry."""
    assert "'/'" in validate_model(model(model="batch/standard"))


@pytest.mark.parametrize("name", ["", " batch", "batch "])
def test_empty_or_padded_model_names_are_rejected(name):
    assert validate_model(model(model=name)) is not None


def test_an_unknown_mode_is_rejected():
    assert "unknown mode" in validate_model(model(mode="chat"))


def test_an_unknown_unit_is_rejected():
    assert "unknown unit" in validate_model(model(unit="minute"))


def test_authoritative_cannot_be_claimed_by_a_page_read():
    """AUTHORITATIVE means a vendor price API. A marketing page is not one."""
    assert "not claimable" in validate_model(model(confidence="authoritative"))


def test_a_rate_with_no_published_form_is_rejected():
    """Without it nobody but the adapter's author can re-check the number."""
    assert "cannot be traced back" in validate_model(model(published_as=""))


# --------------------------------------------------------------------------------- #
# promotional rates
# --------------------------------------------------------------------------------- #


def test_a_promotional_rate_above_list_means_the_columns_are_swapped():
    reason = validate_model(model(rate=1.25e-04, promotional_rate=2.0e-04))
    assert "columns are likely swapped" in reason


@pytest.mark.parametrize("promo", [0, -1e-05, float("nan")])
def test_an_implausible_promotional_rate_is_rejected(promo):
    assert validate_model(model(promotional_rate=promo)) is not None


def test_a_promotional_rate_below_list_is_fine():
    assert validate_model(model(promotional_rate=3.58e-05)) is None


# --------------------------------------------------------------------------------- #
# batch-level checks
# --------------------------------------------------------------------------------- #


def test_duplicate_keys_reject_the_whole_extraction():
    """Two rows for one model means the parser matched twice.

    Which rate wins would be an accident of ordering, and one duplicated row makes every
    other row from the same parse suspect -- so none of it is taken.
    """
    accepted, rejections = validate_batch(
        "speechmatics", [model(), model(rate=2.0e-04)], min_models=1
    )
    assert accepted == []
    assert "duplicate model keys" in rejections[0]


def test_falling_below_the_floor_takes_nothing():
    """A partial extraction is indistinguishable from a vendor retiring models.

    bud-connect reads "absent from this run" as "deactivate", so publishing three of seven
    models is worse than publishing none.
    """
    accepted, rejections = validate_batch("speechmatics", [model()], min_models=7)
    assert accepted == []
    assert "below the floor of 7" in rejections[-1]


def test_an_empty_extraction_is_a_failure_not_an_empty_catalog():
    accepted, rejections = validate_batch("speechmatics", [], min_models=1)
    assert accepted == []
    assert rejections


def test_one_bad_row_does_not_sink_the_others_when_the_floor_still_holds():
    accepted, rejections = validate_batch(
        "speechmatics",
        [model(), model(model="batch-enhanced", rate=0.75)],  # second is per-hour
        min_models=1,
    )
    assert [m.model for m in accepted] == ["batch-standard"]
    assert len(rejections) == 1


# --------------------------------------------------------------------------------- #
# drift against the committed floor
# --------------------------------------------------------------------------------- #


def test_a_per_hour_misread_is_caught_as_drift():
    """3600x. The single most likely adapter bug for a per-hour vendor."""
    assert exceeds_drift(0.24, 6.6667e-05) is True


@pytest.mark.parametrize("factor", [1.0, 2.0, 0.5, MAX_DRIFT_FACTOR, 1 / MAX_DRIFT_FACTOR])
def test_ordinary_repricing_is_not_drift(factor):
    """Vendors halve and double prices. Speechmatics currently shows a 46% cut."""
    assert exceeds_drift(1e-04 * factor, 1e-04) is False


@pytest.mark.parametrize("factor", [MAX_DRIFT_FACTOR * 1.01, 1 / (MAX_DRIFT_FACTOR * 1.01)])
def test_movement_beyond_the_factor_is_drift(factor):
    assert exceeds_drift(1e-04 * factor, 1e-04) is True


@pytest.mark.parametrize("floor", [0, -1, float("nan")])
def test_no_trustworthy_floor_means_no_drift_verdict(floor):
    """With nothing sane to compare against, the gates above are the only judge."""
    assert exceeds_drift(1e-04, floor) is False


@pytest.mark.parametrize("scraped", [float("nan"), 0, -1])
def test_an_unusable_new_rate_counts_as_drift(scraped):
    assert exceeds_drift(scraped, 1e-04) is True


def test_math_is_not_accidentally_symmetric():
    """Guards the ratio logic: 10x up and 10x down must behave the same way."""
    assert exceeds_drift(1e-03, 1e-04) is False  # exactly 10x
    assert exceeds_drift(1e-04, 1e-03) is False
    assert exceeds_drift(1.1e-03, 1e-04) is True
    assert exceeds_drift(1e-04, 1.1e-03) is True
    assert math.isclose(MAX_DRIFT_FACTOR, 10.0)
