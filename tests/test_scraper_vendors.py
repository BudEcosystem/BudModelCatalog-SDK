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


@pytest.mark.parametrize("scraper", all_scrapers(), ids=lambda s: s.vendor)
def test_every_adapter_produces_only_valid_models(scraper):
    """The gates and the adapters have to agree, or the source rejects its own output."""
    extracted = scraper.extract(fixture(scraper.vendor))
    for m in extracted:
        assert validate_model(m) is None, f"{scraper.vendor}/{m.model}: {validate_model(m)}"


@pytest.mark.parametrize("scraper", all_scrapers(), ids=lambda s: s.vendor)
def test_every_adapter_clears_its_own_floor(scraper):
    accepted, rejections = validate_batch(
        scraper.vendor, scraper.extract(fixture(scraper.vendor)), scraper.min_models
    )
    assert accepted, rejections


@pytest.mark.parametrize("scraper", all_scrapers(), ids=lambda s: s.vendor)
def test_every_adapter_agrees_with_the_committed_floor(scraper):
    """The scraper and `curated_voice_pricing.yaml` must not drift apart.

    The YAML is the fallback used when a scrape fails, so a silent divergence means the
    fallback is wrong in exactly the situation it is needed. Any real price change should
    land in both, in the same commit.
    """
    committed = CuratedSource().load().data
    for m in scraper.extract(fixture(scraper.vendor)):
        key = f"{scraper.vendor}/{m.model}"
        if key not in committed:
            continue  # a model the page has and the file does not yet
        field = "input_cost_per_second" if m.unit == "second" else "input_cost_per_character"
        assert committed[key][field] == pytest.approx(m.rate), (
            f"{key}: adapter reads {m.rate:g} but the committed floor says "
            f"{committed[key][field]:g}"
        )
