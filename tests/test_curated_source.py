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
    assert "gladia/solaria-1" in loaded
    for key, entry in loaded.items():
        provider, _, model = key.partition("/")
        assert entry["litellm_provider"] == provider
        assert entry["metadata"]["original_key"] == model


@pytest.mark.parametrize("key", ["revai/reverb", "gladia/solaria-1"])
def test_every_entry_declares_a_mode(loaded, key):
    """bud-connect derives endpoints from `mode` when nothing else says."""
    assert loaded[key]["mode"] in {"audio_transcription", "audio_speech"}


def test_every_entry_carries_a_billing_block(loaded):
    """A curated rate with no provenance cannot be re-checked or aged out."""
    for key, entry in loaded.items():
        billing = entry.get("billing")
        assert billing, f"{key} has no billing block"
        assert billing["confidence"] in {"curated", "derived"}, (
            f"{key} claims {billing['confidence']!r}; a page read is CURATED and credit "
            "arithmetic is DERIVED. AUTHORITATIVE is reserved for vendor price APIs."
        )
        assert billing["source"]["url"], f"{key} has no source URL"
        assert billing["source"]["checked_on"], f"{key} has no checked_on date"


#: The catalog cost field each stored unit must be carried in.
UNIT_TO_FIELD = {"second": "input_cost_per_second", "character": "input_cost_per_character"}


def test_every_entry_declares_the_unit_the_rate_is_in(loaded):
    """The whole reason these rates are wrong when copied naively.

    Rev AI quotes Reverb per hour and Whisper Large per minute; Speechmatics quotes STT per
    hour and TTS per 1k characters; Cartesia quotes credits. All are stored in the catalog's
    own units, and `unit` is what says which one.
    """
    for key, entry in loaded.items():
        unit = entry["billing"]["unit"]
        assert unit in UNIT_TO_FIELD, f"{key} declares unrecognised unit {unit!r}"
        field = UNIT_TO_FIELD[unit]
        assert field in entry, f"{key} declares unit {unit!r} but carries no {field}"
        wrong = set(UNIT_TO_FIELD.values()) - {field}
        assert not (wrong & set(entry)), f"{key} carries a rate in a unit it does not declare"


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
        ("gladia/solaria-1", 0.75),  # the "Real-time" (live API) rate
        ("revai/reverb", 0.20),
        ("revai/reverb-foreign-language", 0.30),
        ("speechmatics/standard", 0.45),  # the "Real-time Standard" row
        ("speechmatics/enhanced", 0.80),  # the "Real-time Enhanced" row
        ("google_speech/chirp_3", 0.96),  # published as $0.016/minute
        ("google_speech/chirp_2", 0.96),
        ("google_speech/telephony", 0.96),
        ("google_speech/medical_conversation", 4.68),  # published as $0.078/minute
        ("google_speech/medical_dictation", 4.68),
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
    for key in ("revai/reverb", "revai/reverb-foreign-language"):
        assert loaded[key]["billing"]["min_billable_units"] == 15
        assert loaded[key]["billing"]["rounding_increment"] == 1


def test_no_rate_is_zero_or_negative(loaded):
    """A zero rate is a request billed as free, and it never looks wrong in review."""
    for key, entry in loaded.items():
        field = UNIT_TO_FIELD[entry["billing"]["unit"]]
        assert entry[field] > 0, key


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


#: Models of a feed-covered vendor that the feed does not list. A curated entry for a fed
#: vendor is allowed ONLY here: it fills a gap the feed leaves rather than curating the same
#: thing in parallel. Named model by model, because coverage stopped being all-or-nothing
#: per mode: LiteLLM has ElevenLabs' Multilingual and Scribe v2 but not Flash.
FEED_GAPS = frozenset(
    {
        # LiteLLM carries none of Deepgram's TTS; all its Deepgram entries are STT.
        ("deepgram", "aura"),
        ("deepgram", "aura-2"),
        # Flash is absent from LiteLLM, as is the medical Scribe.
        ("elevenlabs", "eleven_flash_v2_5"),
        ("elevenlabs", "eleven_flash_v2"),
        ("elevenlabs", "scribe_v2_medical"),
        # LiteLLM's AssemblyAI entries are the retired best/nano, which the SDK drops.
        ("assemblyai", "universal-3-5-pro"),
        ("assemblyai", "universal-2"),
    }
)


def test_curated_vendors_do_not_collide_with_live_feed_providers(raw):
    """A feed-covered vendor is curated only for models the feed does not supply.

    This used to be vendor-level -- "a vendor a feed covers should be dropped from this
    file" -- which was right while coverage was all-or-nothing. Deepgram broke that: LiteLLM
    covers its speech-to-text and none of its text-to-speech, so a vendor-level rule would
    have left Aura without a fallback price. The intent is unchanged: nothing the feed
    already supplies is curated in parallel. And if the feed later adds these, the runtime
    overlay lets the feed's key win and logs that the curated one should be deleted.
    """
    covered = set(LITELLM_TO_TENSORZERO.values())
    for vendor, models in raw.items():
        if vendor not in covered:
            continue
        for model in models:
            assert (vendor, model) in FEED_GAPS, (
                f"{vendor}/{model}: {vendor} is covered by LiteLLM, so a curated entry is only "
                "allowed for a model LiteLLM does not list -- add it to FEED_GAPS with the "
                "reason, or delete it from the curated file"
            )


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


# --------------------------------------------------------------------------------- #
# per-character rates, checked back against the published figure
# --------------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("key", "published_per_1k_chars"),
    [
        ("speechmatics/text-to-speech", 0.011),
        ("cartesia/sonic-3.6", 0.065),  # 1 credit/char at $65 per 1M credits
        ("cartesia/sonic-3.5", 0.065),
        ("elevenlabs/eleven_flash_v2_5", 0.05),
        ("elevenlabs/eleven_flash_v2", 0.05),
    ],
)
def test_per_character_rates_convert_back(loaded, key, published_per_1k_chars):
    assert loaded[key]["input_cost_per_character"] * 1000 == pytest.approx(published_per_1k_chars)


def test_cartesias_stt_rate_matches_its_own_credit_arithmetic(loaded):
    """3 credits per second at $65 per 1M credits.

    Cartesia publishes no per-unit price, only the conversions. Recomputing them here means a
    hand-typo in the YAML fails a test rather than becoming a bill.
    """
    for key in ("cartesia/ink-2", "cartesia/ink-whisper"):
        assert loaded[key]["input_cost_per_second"] == pytest.approx(3 * 65e-06)


def test_only_credit_based_vendors_are_marked_derived(loaded):
    """DERIVED means "computed from something that is not a price".

    Keeping it to vendors that genuinely sell credits stops it becoming a soft label for a
    rate nobody checked.
    """
    derived = {
        k.split("/")[0] for k, v in loaded.items() if v["billing"]["confidence"] == "derived"
    }
    assert derived == {"cartesia"}, f"unexpected derived vendors: {sorted(derived - {'cartesia'})}"


def test_a_rate_written_in_scientific_notation_is_rejected(tmp_path):
    """YAML reads `3e-05` as a string, not a float.

    Scientific notation only parses as a number with a decimal point and a signed exponent
    (`3.0e-05`), so the natural way to write a small rate produces a string that looks
    completely fine in the file. This was a real bug: generated entries for google_speech
    and speechify all arrived as strings, and the failure surfaced as a TypeError comparing
    str to int in an unrelated test.
    """
    path = tmp_path / "p.yaml"
    path.write_text(
        "vendor:\n"
        "  model:\n"
        "    mode: audio_speech\n"
        "    input_cost_per_character: 3e-05\n"
        "    billing:\n"
        "      unit: character\n"
        "      source:\n"
        '        checked_on: "2026-09-24"\n'
    )
    with pytest.raises(ValueError, match="plain decimal"):
        CuratedSource(path).load()


def test_a_plain_decimal_rate_loads_as_a_float(tmp_path):
    path = tmp_path / "p.yaml"
    path.write_text(
        "vendor:\n"
        "  model:\n"
        "    mode: audio_speech\n"
        "    input_cost_per_character: 0.00003\n"
        "    billing:\n"
        "      unit: character\n"
        "      source:\n"
        '        checked_on: "2026-09-24"\n'
    )
    loaded = CuratedSource(path).load().data
    assert isinstance(loaded["vendor/model"]["input_cost_per_character"], float)


# --------------------------------------------------------------------------------- #
# keys are model ids the vendor's API accepts, not pricing-page labels
# --------------------------------------------------------------------------------- #

#: Keys this file used to carry that no request can name. Each was a pricing SKU or product
#: label slugged into a key, or an id for an API WaaV does not call. One coming back is a
#: model budapp offers and every request to it fails.
RETIRED_KEYS = {
    "google_speech/instant-custom-voice": "needs a v1beta1 voice_cloning_key; not a model id",
    "google_speech/dynamic-batch-recognition": "a V2 BatchRecognize urgency mode, priced as a SKU",
    "google_speech/recognition": "the V2 standard SKU; the models it prices are chirp_3 etc.",
    "google_speech/speech-recognition-with-data-logging": "a V1 SKU",
    "google_speech/speech-recognition-without-data-logging": "a V1 SKU",
    "google_speech/medical-conversation": "the V2 id is medical_conversation",
    "google_speech/medical-dictation": "the V2 id is medical_dictation",
    "speechmatics/batch-standard": "Batch API; WaaV calls Realtime",
    "speechmatics/batch-enhanced": "Batch API; WaaV calls Realtime",
    "speechmatics/batch-melia-1": "melia-1 is Batch-only",
    "speechmatics/linden-1": "Agent STT (/v2/agent) only",
    "speechmatics/real-time-standard": "a page label; the Realtime id is `standard`",
    "speechmatics/real-time-enhanced": "a page label; the Realtime id is `enhanced`",
    "revai/whisper-large": "no `transcriber` value selects it on either Rev AI API",
    "gladia/async": "a product name, pricing the pre-recorded API WaaV does not call",
    "gladia/real-time": "a product name; the live API's only model is `solaria-1`",
}

#: The keys that replace them, with the rate each carries: the rate of the row it is priced
#: from, which for every renamed key is the rate the old key had.
MODEL_ID_KEYS = {
    "google_speech/chirp_3": 0.016 / 60,  # V2 "Recognition" (Standard) row
    "google_speech/chirp_2": 0.016 / 60,
    "google_speech/telephony": 0.016 / 60,
    "google_speech/medical_conversation": 0.078 / 60,
    "google_speech/medical_dictation": 0.078 / 60,
    "speechmatics/standard": 0.45 / 3600,  # "Real-time Standard"
    "speechmatics/enhanced": 0.80 / 3600,  # "Real-time Enhanced"
    "gladia/solaria-1": 0.75 / 3600,  # Starter "Real-time"
}


@pytest.mark.parametrize("key", sorted(RETIRED_KEYS))
def test_a_retired_label_is_not_published(loaded, key):
    assert key not in loaded, f"{key} is back: {RETIRED_KEYS[key]}"


@pytest.mark.parametrize(("key", "per_second"), sorted(MODEL_ID_KEYS.items()))
def test_each_model_id_is_published_at_its_rows_rate(loaded, key, per_second):
    assert loaded[key]["mode"] == "audio_transcription"
    assert loaded[key]["billing"]["unit"] == "second"
    assert loaded[key]["input_cost_per_second"] == pytest.approx(per_second)


def test_google_speech_to_text_keys_are_v2_model_ids(loaded):
    """V2's `RecognitionConfig.model` takes underscored ids (`chirp_3`, `medical_dictation`).

    The page's row labels slug to hyphens, and a hyphenated id is one Google rejects -- which
    is how `medical-dictation` shipped. Text-to-speech keys are family names WaaV maps to
    voices, so they are not held to this.
    """
    stt = {
        key.split("/", 1)[1]
        for key, entry in loaded.items()
        if key.startswith("google_speech/") and entry["mode"] == "audio_transcription"
    }
    assert stt == {"chirp_3", "chirp_2", "telephony", "medical_conversation", "medical_dictation"}
