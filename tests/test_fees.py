"""Tests for per-series Kalshi fee terms (slugger/fees.py, bead 81s).

The theme: every failure path must resolve to the *expensive* assumption.
A fee lookup that quietly guesses cheap is indistinguishable from edge.
"""
from __future__ import annotations

import pytest

from slugger.fees import (
    FALLBACK_FEE_TYPE,
    SeriesFeeCache,
    SeriesFees,
    fallback_fees,
    series_ticker,
)
from slugger.models import (
    FEE_TYPE_QUADRATIC,
    FEE_TYPE_QUADRATIC_WITH_MAKER_FEES,
    KALSHI_TAKER_FEE_RATE,
    kalshi_fee_dollars,
)


class _Client:
    def __init__(self, payloads=None, raises=False):
        self.payloads = payloads or {}
        self.raises = raises
        self.calls = []

    def get_series(self, series):
        self.calls.append(series)
        if self.raises:
            raise RuntimeError("boom")
        return self.payloads.get(series)


# ─── Ticker parsing ──────────────────────────────────────────────────────────

def test_series_ticker_extraction():
    assert series_ticker("KXMLBGAME-26AUG181840DETPIT-PIT") == "KXMLBGAME"
    assert series_ticker("KXMLBTOTAL-26AUG181835NYYBAL-8") == "KXMLBTOTAL"
    assert series_ticker("KXMLBGAME") == "KXMLBGAME"
    assert series_ticker("") == ""


# ─── Resolution ──────────────────────────────────────────────────────────────

def test_resolves_fee_type_from_the_series_endpoint():
    c = _Client({"KXMLBGAME": {"fee_type": "quadratic_with_maker_fees",
                               "fee_multiplier": 0.5}})
    f = SeriesFeeCache(c).for_ticker("KXMLBGAME-26AUG181840DETPIT-PIT")
    assert f.fee_type == FEE_TYPE_QUADRATIC_WITH_MAKER_FEES
    assert f.resolved is True
    assert f.reported_multiplier == 0.5


def test_reported_multiplier_is_recorded_but_not_charged():
    """Kalshi reports 0.5 on every MLB series; our fills are charged 0.07.

    Applying it reproduced 24% of 335 taker fills against 86% without it.
    """
    c = _Client({"KXMLBTOTAL": {"fee_type": "quadratic", "fee_multiplier": 0.5}})
    f = SeriesFeeCache(c).for_series("KXMLBTOTAL")
    assert f.reported_multiplier == 0.5
    assert f.taker_rate == KALSHI_TAKER_FEE_RATE
    assert f.fee_dollars(50, 10) == kalshi_fee_dollars(50, 10)


def test_maker_free_on_quadratic_but_not_on_maker_fee_series():
    cache = SeriesFeeCache(_Client({
        "KXMLBTOTAL": {"fee_type": "quadratic", "fee_multiplier": 0.5},
        "KXMLBGAME": {"fee_type": "quadratic_with_maker_fees",
                      "fee_multiplier": 0.5},
    }))
    assert cache.for_series("KXMLBTOTAL").fee_dollars(50, 10, maker=True) == 0.0
    game = cache.for_series("KXMLBGAME")
    assert game.fee_dollars(50, 10, maker=True) == pytest.approx(
        0.25 * game.fee_dollars(50, 10), abs=1e-4)


def test_one_lookup_per_series():
    c = _Client({"KXMLBGAME": {"fee_type": "quadratic"}})
    cache = SeriesFeeCache(c)
    for _ in range(5):
        cache.for_ticker("KXMLBGAME-26AUG181840DETPIT-PIT")
        cache.for_ticker("KXMLBGAME-26AUG181840SFCLE-SF")
    assert c.calls == ["KXMLBGAME"]


# ─── Failure paths all resolve expensive ─────────────────────────────────────

@pytest.mark.parametrize("cache", [
    SeriesFeeCache(None),                                   # no client
    SeriesFeeCache(_Client(raises=True)),                   # endpoint error
    SeriesFeeCache(_Client({})),                            # unknown series
    SeriesFeeCache(object()),                               # client w/o get_series
])
def test_every_failure_path_assumes_the_costly_terms(cache):
    f = cache.for_ticker("KXWHATEVER-1-A")
    assert f.resolved is False
    assert f.fee_type == FALLBACK_FEE_TYPE
    # the costly type: makers are charged, not free
    assert f.fee_dollars(50, 10, maker=True) > 0


def test_failed_lookup_is_cached_not_retried():
    c = _Client({})
    cache = SeriesFeeCache(c)
    for _ in range(4):
        cache.for_series("KXNOPE")
    assert c.calls == ["KXNOPE"]


def test_missing_fee_type_falls_back_rather_than_assuming_free():
    c = _Client({"KXNEW": {"fee_multiplier": 1.0}})
    f = SeriesFeeCache(c).for_series("KXNEW")
    assert f.fee_type == FALLBACK_FEE_TYPE
    assert f.fee_dollars(50, 10, maker=True) > 0


def test_garbage_multiplier_does_not_raise():
    c = _Client({"KXNEW": {"fee_type": "quadratic", "fee_multiplier": "n/a"}})
    f = SeriesFeeCache(c).for_series("KXNEW")
    assert f.reported_multiplier == 1.0
    assert f.fee_type == FEE_TYPE_QUADRATIC


def test_fallback_helper_matches_the_cache_fallback():
    assert fallback_fees("KXMLBGAME-1-A") == SeriesFeeCache(None).for_ticker(
        "KXMLBGAME-1-A")


# ─── The numbers it produces ─────────────────────────────────────────────────

def test_fee_dollars_matches_a_real_charged_fill():
    """KXMLBGAME, taker, 18.28 contracts at 26c, charged $0.2462."""
    f = SeriesFees(series="KXMLBGAME",
                   fee_type=FEE_TYPE_QUADRATIC_WITH_MAKER_FEES, resolved=True)
    assert f.fee_dollars(26, 18.28) == pytest.approx(0.2462)


def test_per_contract_is_the_unrounded_rate():
    f = SeriesFees(series="KXMLBKS", fee_type=FEE_TYPE_QUADRATIC, resolved=True)
    assert f.fee_cents_per_contract(30) == pytest.approx(1.47)
    assert f.fee_cents_per_contract(30, maker=True) == 0.0
