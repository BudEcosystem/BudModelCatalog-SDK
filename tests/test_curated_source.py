"""The bundled hand-curated voice prices.

These rates are transcribed from vendor pricing pages, which means they can be wrong in
ways a fetched feed cannot: a typo, a misread unit, a number nobody has re-checked since
the vendor repriced. The tests below are the guards against each of those.
"""

import textwrap

import pytest
import yaml

from bud_model_catalog.mappings import STRIP_PREFIXES, strip_provider_prefix
from bud_model_catalog.sources.curated import DATA_PATH, CuratedSource
from bud_model_catalog.sources.litellm import LITELLM_TO_TENSORZERO, TENSORZERO_PROVIDERS


@pytest.fixture(scope="module")
def raw():
    return yaml.safe_load(DATA_PATH.read_text())


@pytest.fixture(scope="module")
def loaded():
    return CuratedSource().load().data


def test_the_file_ships_with_the_package():
    """Bundled data that misses the wheel is data that only works in a git checkout."""
    assert DATA_PATH.exists()
    assert DATA_PATH.parent.name == "data"


def test_keys_are_provider_slash_model(loaded):
    assert "revai/reverb" in loaded
    assert "gladia/async" in loaded
    for key, entry in loaded.items():
        provider, _, model = key.partition("/")
        assert entry["litellm_provider"] == provider
        assert entry["metadata"]["original_key"] == model


@pytest.mark.parametrize("key", ["revai/reverb", "gladia/async"])
def test_every_entry_declares_a_mode(loaded, key):
    """bud-connect derives endpoints from `mode` when nothing else says."""
    assert loaded[key]["mode"] in {"audio_transcription", "audio_speech"}


def test_every_entry_carries_a_billing_block(loaded):
    """A curated rate with no provenance cannot be re-checked or aged out."""
    for key, entry in loaded.items():
        billing = entry.get("billing")
        assert billing, f"{key} has no billing block"
        assert billing["confidence"] == "curated", (
            f"{key} claims stronger confidence than a page read"
        )
        assert billing["source"]["url"], f"{key} has no source URL"
        assert billing["source"]["checked_on"], f"{key} has no checked_on date"


def test_every_entry_declares_the_unit_the_rate_is_in(loaded):
    """The whole reason these rates are wrong when copied naively.

    Rev AI quotes Reverb per hour and Whisper Large per minute. Both are stored per second.
    """
    for key, entry in loaded.items():
        assert entry["billing"]["unit"] == "second", key
        assert "input_cost_per_second" in entry, key


def test_a_missing_checked_on_is_rejected(tmp_path):
    """Loudly, at load time -- not silently at bill time."""
    path = tmp_path / "curated.yaml"
    path.write_text(
        textwrap.dedent(
            """
            vendor:
              model:
                mode: audio_transcription
                input_cost_per_second: 0.0001
                billing:
                  unit: second
                  confidence: curated
                  source:
                    url: https://example.invalid/pricing
            """
        )
    )
    with pytest.raises(ValueError, match="checked_on"):
        CuratedSource(path=path).load()


def test_a_missing_file_degrades_instead_of_raising(tmp_path):
    """A packaging mistake should cost the curated prices, not the whole catalog."""
    result = CuratedSource(path=tmp_path / "nope.yaml").load()
    assert result.data == {}


# --------------------------------------------------------------------------------- #
# the arithmetic, checked back against what each page actually says
# --------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("key", "published_per_hour"),
    [
        ("gladia/async", 0.61),
        ("gladia/real-time", 0.75),
        ("revai/reverb", 0.20),
        ("revai/reverb-foreign-language", 0.30),
        ("revai/whisper-large", 0.30),  # published as $0.005/minute
    ],
)
def test_the_stored_rate_converts_back_to_the_published_figure(loaded, key, published_per_hour):
    """Catches a misplaced decimal, which is the failure mode of hand-transcribed rates.

    A rate that is wrong by 60x reads as plausible in scientific notation and is obvious
    the moment it is converted back to the number printed on the page.
    """
    assert loaded[key]["input_cost_per_second"] * 3600 == pytest.approx(published_per_hour)


def test_revai_records_its_minimum_billable_duration(loaded):
    """ "Rounded up to the nearest second, 15 second minimum."

    Without this a 3-second clip bills as 3 seconds instead of 15, and every short request
    is undercharged.
    """
    for key in ("revai/reverb", "revai/reverb-foreign-language", "revai/whisper-large"):
        assert loaded[key]["billing"]["min_billable_units"] == 15
        assert loaded[key]["billing"]["rounding_increment"] == 1


def test_no_rate_is_zero_or_negative(loaded):
    for key, entry in loaded.items():
        assert entry["input_cost_per_second"] > 0, key


# --------------------------------------------------------------------------------- #
# AWS Transcribe, which comes from LiteLLM rather than this file
# --------------------------------------------------------------------------------- #


def test_aws_transcribe_is_mapped():
    """bud-connect already offers `aws_transcribe`; LiteLLM prices it under `transcribe`."""
    assert LITELLM_TO_TENSORZERO["transcribe"] == "aws_transcribe"
    assert "aws_transcribe" in TENSORZERO_PROVIDERS


def test_the_aws_transcribe_prefix_is_stripped_under_its_tensorzero_name():
    """STRIP_PREFIXES is keyed by the TensorZero name, not the LiteLLM one.

    Keyed wrongly it silently does nothing, and the model URI becomes
    `aws_transcribe/transcribe/StartTranscriptionJob`.
    """
    assert STRIP_PREFIXES["aws_transcribe"] == "transcribe/"
    assert (
        strip_provider_prefix("aws_transcribe", "transcribe/StartTranscriptionJob")
        == "StartTranscriptionJob"
    )


def test_curated_vendors_do_not_collide_with_live_feed_providers(raw):
    """A vendor a feed covers should be dropped from this file, not curated in parallel."""
    overlap = set(raw) & set(LITELLM_TO_TENSORZERO.values())
    assert not overlap, f"{overlap} are covered by LiteLLM; delete them from the curated file"


# --------------------------------------------------------------------------------- #
# the overlay
# --------------------------------------------------------------------------------- #


def _result(models):
    from datetime import datetime, timezone

    from bud_model_catalog.models import CatalogResult, MergeStats

    return CatalogResult(
        models=dict(models),
        stats=MergeStats(
            total_litellm=len(models),
            total_output=len(models),
            matched=0,
            unmatched=0,
            deprecated_removed=0,
            cost_fields_updated=0,
        ),
        litellm_fetched_at=datetime.now(timezone.utc),
    )


def test_the_overlay_adds_curated_entries():
    from bud_model_catalog import CatalogClient

    result = CatalogClient()._overlay_curated(
        _result({"openai/gpt-4o": {"litellm_provider": "openai"}})
    )

    assert "revai/reverb" in result.models
    assert "openai/gpt-4o" in result.models
    assert result.stats.total_output == len(result.models)


def test_a_live_feed_entry_wins_over_a_curated_one():
    """The stale source must not overwrite the continuously-refreshed one.

    A collision also means the curated entry is now redundant, which the warning says.
    """
    from bud_model_catalog import CatalogClient

    live = {"revai/reverb": {"litellm_provider": "revai", "input_cost_per_second": 999.0}}
    result = CatalogClient()._overlay_curated(_result(live))

    assert result.models["revai/reverb"]["input_cost_per_second"] == 999.0
