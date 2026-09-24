"""Each vendor adapter, against a trimmed capture of the page it reads.

The fixtures are regions of the real pages, kept small enough to review. They are also the
evidence of what the page said when the rate was confirmed, which is the thing a date in a
YAML file cannot give you.

No test here touches the network. An adapter that needed a live page to be tested would be
a test that fails when a vendor has an outage.
"""

import pathlib

import pytest

from bud_model_catalog.scrapers.registry import all_scrapers
from bud_model_catalog.scrapers.validate import validate_batch, validate_model
from bud_model_catalog.scrapers.vendors.cartesia import CartesiaScraper
from bud_model_catalog.scrapers.vendors.gladia import GladiaScraper
from bud_model_catalog.scrapers.vendors.google_speech import (
    GoogleSpeechSttScraper,
    GoogleSpeechTtsScraper,
)
from bud_model_catalog.scrapers.vendors.revai import RevAiScraper
from bud_model_catalog.scrapers.vendors.speechmatics import SpeechmaticsScraper
from bud_model_catalog.sources.curated import CuratedSource

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "scrape"


def fixture(name: str) -> str:
    return (FIXTURES / f"{name}.html").read_text()


# --------------------------------------------------------------------------------- #
# speechmatics: list prices from embedded JSON, promotions from the rendered table
# --------------------------------------------------------------------------------- #

#: What the page published, and the per-second rate it converts to.
SPEECHMATICS_EXPECTED = {
    "batch-melia-1": (0.24, 0.24 / 3600),
    "batch-standard": (0.45, 0.45 / 3600),
    "batch-enhanced": (0.75, 0.75 / 3600),
    "real-time-standard": (0.45, 0.45 / 3600),
    "real-time-enhanced": (0.80, 0.80 / 3600),
    "linden-1": (0.30, 0.30 / 3600),
}


def test_speechmatics_extracts_every_billable_model():
    models = {m.model: m for m in SpeechmaticsScraper().extract(fixture("speechmatics"))}
    assert set(models) == set(SPEECHMATICS_EXPECTED) | {"text-to-speech"}


@pytest.mark.parametrize(
    ("key", "published", "per_second"), [(k, *v) for k, v in SPEECHMATICS_EXPECTED.items()]
)
def test_speechmatics_converts_per_hour_to_per_second(key, published, per_second):
    models = {m.model: m for m in SpeechmaticsScraper().extract(fixture("speechmatics"))}
    m = models[key]
    assert m.unit == "second"
    assert m.mode == "audio_transcription"
    assert m.rate == pytest.approx(per_second)
    assert f"${published:g}/hr" == m.published_as


def test_speechmatics_takes_the_list_price_not_the_promotion():
    """The two disagree by about 46%, and the promotion is the one on screen.

    List is stored because a promotion expires and a cost estimate that silently assumed
    one is wrong the day it ends. The promotion is still captured, so the gap is visible
    rather than lost.
    """
    models = {m.model: m for m in SpeechmaticsScraper().extract(fixture("speechmatics"))}
    melia = models["batch-melia-1"]
    assert melia.rate == pytest.approx(0.24 / 3600)  # list
    assert melia.promotional_rate == pytest.approx(0.129 / 3600)  # displayed
    assert melia.promotional_rate < melia.rate
    assert "promotional $0.129/hr" in melia.note


def test_speechmatics_reads_the_text_to_speech_row_in_characters():
    models = {m.model: m for m in SpeechmaticsScraper().extract(fixture("speechmatics"))}
    tts = models["text-to-speech"]
    assert tts.unit == "character"
    assert tts.mode == "audio_speech"
    assert tts.rate == pytest.approx(0.011 / 1000)


def test_speechmatics_excludes_bolt_ons():
    """Translation, Summaries, Chapters, Sentiment and Topics are surcharges on a
    transcript, not models a user deploys -- the same reason AWS's Call Analytics and
    Redaction SKUs are excluded."""
    models = {m.model for m in SpeechmaticsScraper().extract(fixture("speechmatics"))}
    for bolt_on in ("translation", "summaries", "chapters", "sentiment", "topics"):
        assert bolt_on not in models


@pytest.mark.parametrize("html", ["", "<html><body>Pricing</body></html>", "{}"])
def test_speechmatics_fails_loudly_when_the_page_carries_no_pricing_rows(html):
    """A page that renders its prices in JavaScript, or an error page served with 200.

    Raising is the point: returning [] here would look like a vendor with no models.
    """
    with pytest.raises(ValueError, match="no pricing rows"):
        SpeechmaticsScraper().extract(html)


# --------------------------------------------------------------------------------- #
# cartesia: derived from stated credit conversions
# --------------------------------------------------------------------------------- #


def test_cartesia_derives_both_rates_from_the_page():
    models = {m.model: m for m in CartesiaScraper().extract(fixture("cartesia"))}
    assert models["sonic"].rate == pytest.approx(1 * 65e-06)  # 1 credit/char
    assert models["ink"].rate == pytest.approx(3 * 65e-06)  # 3 credits/second
    assert models["sonic"].unit == "character"
    assert models["ink"].unit == "second"


def test_cartesia_rates_are_marked_derived_not_curated():
    """The arithmetic is an assumption about how credits convert, and it breaks if a tier
    is repriced without repricing credits. A consumer needs to be able to see that."""
    for m in CartesiaScraper().extract(fixture("cartesia")):
        assert m.confidence == "derived"


def test_cartesia_records_the_conversion_it_used():
    models = {m.model: m for m in CartesiaScraper().extract(fixture("cartesia"))}
    assert "per 1M credits" in models["ink"].published_as
    assert "pro tier" in models["ink"].note


def test_cartesia_refuses_to_guess_without_the_credit_price():
    """Every input is parsed, not hardcoded. Losing one has to fail, not fall back.

    A hardcoded $65/1M would keep producing plausible numbers long after Cartesia changed
    it, which is exactly how a derived rate goes quietly wrong.
    """
    html = "<p>1 credit equals 1 character. 3 credits equals 1 second of audio.</p>"
    with pytest.raises(ValueError, match="pro-tier credit price"):
        CartesiaScraper().extract(html)


def test_cartesia_refuses_to_guess_without_the_unit_conversions():
    html = "<p>At pro tier, it's $65 per 1M credits.</p>"
    with pytest.raises(ValueError, match="no credit-to-unit conversions"):
        CartesiaScraper().extract(html)


# --------------------------------------------------------------------------------- #
# properties every adapter has to hold
# --------------------------------------------------------------------------------- #


@pytest.mark.parametrize("scraper", all_scrapers(), ids=lambda s: s.slug())
def test_every_adapter_produces_only_valid_models(scraper):
    """The gates and the adapters have to agree, or the source rejects its own output."""
    extracted = scraper.extract(fixture(scraper.slug()))
    for m in extracted:
        assert validate_model(m) is None, f"{scraper.vendor}/{m.model}: {validate_model(m)}"


@pytest.mark.parametrize("scraper", all_scrapers(), ids=lambda s: s.slug())
def test_every_adapter_clears_its_own_floor(scraper):
    accepted, rejections = validate_batch(
        scraper.vendor, scraper.extract(fixture(scraper.slug())), scraper.min_models
    )
    assert accepted, rejections


@pytest.mark.parametrize("scraper", all_scrapers(), ids=lambda s: s.slug())
def test_every_adapter_agrees_with_the_committed_floor(scraper):
    """The scraper and `curated_voice_pricing.yaml` must not drift apart.

    The YAML is the fallback used when a scrape fails, so a silent divergence means the
    fallback is wrong in exactly the situation it is needed. Any real price change should
    land in both, in the same commit.
    """
    committed = CuratedSource().load().data
    for m in scraper.extract(fixture(scraper.slug())):
        key = f"{scraper.vendor}/{m.model}"
        if key not in committed:
            continue  # a model the page has and the file does not yet
        field = "input_cost_per_second" if m.unit == "second" else "input_cost_per_character"
        assert committed[key][field] == pytest.approx(m.rate), (
            f"{key}: adapter reads {m.rate:g} but the committed floor says "
            f"{committed[key][field]:g}"
        )


# --------------------------------------------------------------------------------- #
# rev ai: a rate per model, plus the minimum that makes short clips expensive
# --------------------------------------------------------------------------------- #


def test_revai_extracts_the_transcription_models():
    models = {m.model: m for m in RevAiScraper().extract(fixture("revai"))}
    assert set(models) == {"reverb", "reverb-foreign-language", "whisper-large"}
    assert models["reverb"].rate == pytest.approx(0.20 / 3600)
    assert models["reverb-foreign-language"].rate == pytest.approx(0.30 / 3600)
    # Quoted per minute, unlike the other two.
    assert models["whisper-large"].rate == pytest.approx(0.005 / 60)


def test_revai_carries_the_fifteen_second_minimum():
    """The field that makes this vendor different, and the one easiest to lose.

    A 2-second clip is billed as 15 seconds. An adapter that reads the rate and drops the
    minimum under-bills nearly every request, because short clips are the common case for
    voice.
    """
    for m in RevAiScraper().extract(fixture("revai")):
        assert m.min_billable_units == 15.0, m.model


def test_revai_model_names_do_not_absorb_surrounding_prose():
    """Regression: the first version of this adapter regexed a tag-stripped blob and
    produced a model called
    "supports-all-popular-media-types-email-and-chat-support-...-reverb",
    because a non-greedy match had no element boundary to stop at."""
    for m in RevAiScraper().extract(fixture("revai")):
        assert len(m.model) < 32, m.model
        assert "get-started" not in m.model


def test_revai_excludes_human_transcription_and_add_ons():
    """Human Transcription is people. Forced Alignment, Language Identification,
    Translation, Sentiment, Summarization and Topic Extraction are applied to a transcript
    rather than deployed."""
    models = {m.model for m in RevAiScraper().extract(fixture("revai"))}
    for excluded in (
        "human",
        "forced-alignment",
        "language-identification",
        "language-translation",
        "sentiment-analysis",
        "summarization",
        "topic-extraction",
    ):
        assert excluded not in models


def test_revai_fails_loudly_when_the_blocks_are_gone():
    with pytest.raises(ValueError, match="payment-offering"):
        RevAiScraper().extract("<html><body>Pricing</body></html>")


# --------------------------------------------------------------------------------- #
# gladia: list price, not the committed-volume price
# --------------------------------------------------------------------------------- #


def test_gladia_extracts_both_rates():
    models = {m.model: m for m in GladiaScraper().extract(fixture("gladia"))}
    assert set(models) == {"async", "real-time"}
    assert models["async"].rate == pytest.approx(0.61 / 3600)
    assert models["real-time"].rate == pytest.approx(0.75 / 3600)


def test_gladia_ignores_the_growth_tier_rate():
    """Growth is "as low as $0.20 /hr" and needs an upfront commitment, which makes it a
    volume discount rather than a list price. The fixture deliberately includes it."""
    text = fixture("gladia")
    assert "as low as" in text, "fixture no longer exercises the Growth tier"
    for m in GladiaScraper().extract(text):
        assert m.rate > 0.20 / 3600, f"{m.model} picked up a committed-volume rate"


def test_gladia_fails_loudly_when_the_starter_rates_are_gone():
    with pytest.raises(ValueError, match="Async at"):
        GladiaScraper().extract("<html><body>Contact us</body></html>")


# --------------------------------------------------------------------------------- #
# google cloud text-to-speech: a genuine per-model table, already per character
# --------------------------------------------------------------------------------- #

#: Model key -> published rate per character, as the page states it.
GOOGLE_EXPECTED = {
    "chirp-3-hd": 0.00003,
    "instant-custom-voice": 0.00006,
    "wavenet": 0.000004,
    "studio": 0.00016,
    "standard": 0.000004,
    "neural2": 0.000016,
    "polyglot": 0.000016,
}


def test_google_extracts_every_character_priced_model():
    models = {m.model: m for m in GoogleSpeechTtsScraper().extract(fixture("google_speech_tts"))}
    assert set(models) == set(GOOGLE_EXPECTED)


@pytest.mark.parametrize(("key", "per_character"), sorted(GOOGLE_EXPECTED.items()))
def test_google_rates_need_no_conversion(key, per_character):
    """The one vendor that quotes the catalog's own unit directly."""
    models = {m.model: m for m in GoogleSpeechTtsScraper().extract(fixture("google_speech_tts"))}
    assert models[key].rate == pytest.approx(per_character)
    assert models[key].unit == "character"
    assert models[key].mode == "audio_speech"


def test_google_skips_the_token_priced_gemini_rows():
    """Gemini TTS is billed per million input text tokens and output audio tokens.

    That is a different basis from per character. The fixture includes one of those rows
    precisely so this stays true -- storing it as a character rate would be wrong by orders
    of magnitude in a way the bounds check would not catch.
    """
    text = fixture("google_speech_tts")
    assert "per 1 million text tokens" in text, "fixture no longer exercises a Gemini row"
    models = {m.model for m in GoogleSpeechTtsScraper().extract(text)}
    assert not any("gemini" in m for m in models)


def test_google_model_keys_drop_the_category_suffix_but_not_the_product_name():
    """ "WaveNet voices" is a category; "Instant custom voice" is a product name."""
    models = {m.model for m in GoogleSpeechTtsScraper().extract(fixture("google_speech_tts"))}
    assert "wavenet" in models
    assert "instant-custom-voice" in models


def test_google_fails_loudly_without_a_table():
    with pytest.raises(ValueError, match="no table rows"):
        GoogleSpeechTtsScraper().extract("<html><body>Pricing</body></html>")


# --------------------------------------------------------------------------------- #
# google speech-to-text: three price columns that look identical in stripped text
# --------------------------------------------------------------------------------- #

#: Model key -> the LIST price per minute. The committed-savings figures for each row are
#: 10% and 20% lower, and reading one of those instead is the failure this table invites.
GOOGLE_STT_EXPECTED = {
    "recognition": 0.016,
    "dynamic-batch-recognition": 0.003,
    "speech-recognition-with-data-logging": 0.016,
    "speech-recognition-without-data-logging": 0.024,
    "medical-dictation": 0.078,
    "medical-conversation": 0.078,
}

#: Every committed-savings rate on the page. None of these may ever be stored.
GOOGLE_STT_SAVINGS = [
    0.0144,
    0.0128,  # recognition
    0.0027,
    0.0024,  # dynamic batch
    0.021,
    0.019,  # without data logging
    0.070,
    0.062,  # medical
]


def test_google_stt_extracts_every_model():
    models = {m.model: m for m in GoogleSpeechSttScraper().extract(fixture("google_speech_stt"))}
    assert set(models) == set(GOOGLE_STT_EXPECTED)


@pytest.mark.parametrize(("key", "per_minute"), sorted(GOOGLE_STT_EXPECTED.items()))
def test_google_stt_takes_the_list_rate(key, per_minute):
    models = {m.model: m for m in GoogleSpeechSttScraper().extract(fixture("google_speech_stt"))}
    assert models[key].rate == pytest.approx(per_minute / 60)
    assert models[key].unit == "second"
    assert models[key].mode == "audio_transcription"


def test_google_stt_never_stores_a_committed_savings_rate():
    """The whole reason this adapter was written separately.

    A 10% error here is invisible: the number is plausible, the units are right, and no
    bounds check or drift check would ever flag it.
    """
    for m in GoogleSpeechSttScraper().extract(fixture("google_speech_stt")):
        for savings in GOOGLE_STT_SAVINGS:
            assert m.rate != pytest.approx(savings / 60), (
                f"{m.model} stored ${savings}/min, which is a committed-savings price"
            )


def test_google_stt_skips_the_free_tier():
    """Cells read "0 minute to 60 minute $0.00 (Free) ... 60 minute and above $0.016".

    Taking the first number in the cell would price these models at zero -- a request
    billed as free, which looks entirely unremarkable in a catalog.
    """
    text = fixture("google_speech_stt")
    assert "(Free)" in text, "fixture no longer exercises a free tier"
    for m in GoogleSpeechSttScraper().extract(text):
        assert m.rate > 0


def test_google_stt_keeps_parentheses_that_change_the_price():
    """ "with data logging" is $0.016/min and "without" is $0.024/min -- a 50% difference
    carried entirely inside a parenthetical, so unlike the TTS names these are not stripped.
    """
    models = {m.model: m for m in GoogleSpeechSttScraper().extract(fixture("google_speech_stt"))}
    with_logging = models["speech-recognition-with-data-logging"]
    without_logging = models["speech-recognition-without-data-logging"]
    assert without_logging.rate > with_logging.rate


def test_google_stt_rejects_a_row_where_list_is_not_the_most_expensive():
    """The invariant that makes a column misread loud instead of silent.

    List is by definition dearer than a rate you commit a year to. If it is not, the columns
    have been read in the wrong order, and every rate from that table is suspect.
    """
    html = """<table>
      <tr><th>Category</th><th>Model</th><th>Price (USD)</th>
          <th>Flexible Savings Plan - 1 Year (USD)</th></tr>
      <tr><td>Recognition (sku:X)</td><td>Standard</td>
          <td>$0.01 / 1 minute, per 1 month / account</td>
          <td>$0.016 / 1 minute, per 1 month / account</td></tr>
    </table>"""
    with pytest.raises(ValueError, match="columns are being misread"):
        GoogleSpeechSttScraper().extract(html)


def test_google_stt_refuses_a_table_whose_price_column_it_cannot_name():
    """There is deliberately no positional fallback.

    Guessing "the third column is the list price" is the entire risk this adapter exists to
    avoid, so an unrecognised header yields nothing and the source's floor check turns that
    into a rejection.
    """
    html = """<table>
      <tr><th>Category</th><th>Model</th><th>Flexible Savings Plan - 1 Year (USD)</th></tr>
      <tr><td>Recognition (sku:X)</td><td>Standard</td>
          <td>$0.0144 / 1 minute, per 1 month / account</td></tr>
    </table>"""
    assert GoogleSpeechSttScraper().extract(html) == []


def test_google_stt_tolerates_a_header_shorter_than_its_rows():
    """One table on the real page declares three columns and then renders five."""
    html = """<table>
      <tr><th>Category</th><th>Model</th><th>Price (USD)</th></tr>
      <tr><td>Dynamic Batch Recognition (sku:X)</td><td>Standard&#185;</td>
          <td>$0.003 / 1 minute, per 1 month / account</td>
          <td>$0.0027 / 1 minute, per 1 month / account</td>
          <td>$0.0024 / 1 minute, per 1 month / account</td></tr>
    </table>"""
    extracted = GoogleSpeechSttScraper().extract(html)
    assert [m.model for m in extracted] == ["dynamic-batch-recognition"]
    assert extracted[0].rate == pytest.approx(0.003 / 60)


def test_google_stt_fails_loudly_without_tables():
    with pytest.raises(ValueError, match="no tables"):
        GoogleSpeechSttScraper().extract("<html><body>Pricing</body></html>")


def test_the_two_google_adapters_do_not_produce_overlapping_keys():
    """They share the `google_speech` vendor, so a clash would collide in the catalog."""
    tts = {m.model for m in GoogleSpeechTtsScraper().extract(fixture("google_speech_tts"))}
    stt = {m.model for m in GoogleSpeechSttScraper().extract(fixture("google_speech_stt"))}
    assert not (tts & stt), f"both adapters claim {sorted(tts & stt)}"
