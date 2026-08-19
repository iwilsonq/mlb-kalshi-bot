"""Per-series Kalshi fee terms (mlb-kalshi-bot-81s).

Kalshi's fee formula is uniform but its *terms* are per-series, exposed on
GET /series/{ticker}:

    fee_type        "quadratic"                  -> maker fills are free
                    "quadratic_with_maker_fees"  -> makers pay 25% of taker
    fee_multiplier  a scalar Kalshi reports but, as of 2026-08-19, does not
                    appear to actually charge (see below)

The math itself lives in slugger/models.py, which stays pure. This module is
the I/O half: resolve a market ticker to its series' terms, cache it, and
fall back conservatively when the lookup fails.

Conservative here means *expensive*. Every fallback resolves to
"quadratic_with_maker_fees at the full taker rate", the most costly
combination, because the failure mode of guessing cheap is a bot that takes
trades whose edge does not exist.

On fee_multiplier
-----------------
Every MLB series reports fee_multiplier=0.5. We do not apply it. Checked
against 335 of our own taker fills: the full 0.07 rate reproduces 86% of
them exactly, including every MLB fill on 2026-08-19 — after the field
appeared. Only 2026-08-18's five fills matched 0.035, and the same series
reverted to 0.07 the next day. Applying the multiplier drops the match rate
to 24%. The value is recorded on SeriesFees so a future re-check is cheap;
it is not multiplied in. Re-verify with scripts/analyze_maker_fees.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from slugger.models import (
    FEE_TYPE_QUADRATIC_WITH_MAKER_FEES,
    KALSHI_TAKER_FEE_RATE,
    kalshi_fee_cents_per_contract,
    kalshi_fee_dollars,
)

log = logging.getLogger(__name__)

# Used whenever the real terms are unavailable. The costlier of the two fee
# types, so an unknown series is never cheaper than a known one.
FALLBACK_FEE_TYPE = FEE_TYPE_QUADRATIC_WITH_MAKER_FEES


def series_ticker(market_ticker: str) -> str:
    """Series prefix of a market or event ticker.

    'KXMLBGAME-26AUG181840DETPIT-PIT' -> 'KXMLBGAME'
    """
    return (market_ticker or "").split("-", 1)[0].upper()


@dataclass(frozen=True)
class SeriesFees:
    """Fee terms for one series."""
    series: str
    fee_type: str = FALLBACK_FEE_TYPE
    taker_rate: float = KALSHI_TAKER_FEE_RATE
    # Reported by Kalshi but not applied — see the module docstring.
    reported_multiplier: float = 1.0
    resolved: bool = False   # False => these are the conservative fallback

    def fee_dollars(self, price_cents: float, count: float, *, maker: bool = False) -> float:
        return kalshi_fee_dollars(
            price_cents, count, maker=maker,
            fee_type=self.fee_type, fee_rate=self.taker_rate,
        )

    def fee_cents_per_contract(self, price_cents: float, *, maker: bool = False) -> float:
        return kalshi_fee_cents_per_contract(
            price_cents, maker=maker,
            fee_type=self.fee_type, fee_rate=self.taker_rate,
        )


class SeriesFeeCache:
    """Memoised series -> SeriesFees lookup over a Kalshi client.

    One HTTP call per series per process. Failures are cached too: a series
    whose metadata we could not read should not be retried on every market of
    every game, and the fallback it produces is the safe one anyway.
    """

    def __init__(self, client=None):
        self._client = client
        self._cache: Dict[str, SeriesFees] = {}

    def for_ticker(self, market_ticker: str) -> SeriesFees:
        return self.for_series(series_ticker(market_ticker))

    def for_series(self, series: str) -> SeriesFees:
        if series in self._cache:
            return self._cache[series]
        self._cache[series] = self._fetch(series)
        return self._cache[series]

    def _fetch(self, series: str) -> SeriesFees:
        # MarketClient is a protocol; fixture and backtest clients have no
        # get_series, and that must degrade to the fallback rather than raise.
        getter = getattr(self._client, "get_series", None)
        if getter is None or not series:
            return SeriesFees(series=series)
        try:
            data = getter(series)
        except Exception as exc:
            log.warning("Series %s fee lookup failed (%s) — assuming %s at %.3f",
                        series, exc, FALLBACK_FEE_TYPE, KALSHI_TAKER_FEE_RATE)
            return SeriesFees(series=series)
        if not data:
            return SeriesFees(series=series)

        fee_type = data.get("fee_type") or FALLBACK_FEE_TYPE
        try:
            multiplier = float(data.get("fee_multiplier", 1.0) or 1.0)
        except (TypeError, ValueError):
            multiplier = 1.0
        fees = SeriesFees(
            series=series,
            fee_type=fee_type,
            reported_multiplier=multiplier,
            resolved=True,
        )
        log.debug("Series %s: fee_type=%s reported_multiplier=%.3f "
                  "(multiplier not applied)", series, fee_type, multiplier)
        return fees


def fallback_fees(market_ticker: str = "") -> SeriesFees:
    """Conservative terms for callers with no Kalshi client to hand."""
    return SeriesFees(series=series_ticker(market_ticker))
