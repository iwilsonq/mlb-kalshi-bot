"""Kalshi Trade API client for Slugger MLB bot.

Handles authentication (RSA-PSS signing), market queries,
order placement, and balance/position checks.
"""
from __future__ import annotations
import base64
import datetime
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

log = logging.getLogger(__name__)


# ─── Auth helpers ──────────────────────────────────────────────────────────────

def _load_private_key(key_path: str) -> str:
    """Read PEM private key from file, expanding ~ if present."""
    path = Path(key_path)
    if not path.is_absolute():
        path = path.expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Private key not found: {key_path}")
    return path.read_text().strip()


def _sign_request(
    private_key_pem: str,
    timestamp: str,
    method: str,
    path: str,
    base_url: str,
) -> str:
    """Create RSA-PSS signature for a Kalshi API request.

    Args:
        private_key_pem: PEM-encoded RSA private key
        timestamp: Request timestamp in milliseconds
        method: HTTP method (GET, POST, etc.)
        path: API endpoint path (e.g. /portfolio/balance)
        base_url: The base URL (e.g. https://external-api.kalshi.com/trade-api/v2)

    Returns:
        Base64-encoded signature string
    """
    # Strip query params before signing
    path_no_query = path.split("?")[0]
    # Sign the full URL path from the API root
    full_path = urlparse(base_url + path_no_query).path
    message = f"{timestamp}{method}{full_path}".encode("utf-8")

    private_key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"),
        password=None,
    )
    signature = private_key.sign(
        message,
        asym_padding.PSS(
            mgf=asym_padding.MGF1(hashes.SHA256()),
            salt_length=asym_padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def _auth_headers(
    api_key_id: str,
    private_key_pem: str,
    method: str,
    path: str,
    base_url: str,
) -> Dict[str, str]:
    """Build authenticated request headers."""
    timestamp = str(int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000))
    signature = _sign_request(private_key_pem, timestamp, method, path, base_url)
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }


# ─── Response dataclass ───────────────────────────────────────────────────────

@dataclass
class OrderResult:
    """Result of an order placement."""
    order_id: str
    ticker: str
    action: str           # "buy" or "sell"
    side: str             # "yes" or "no"
    count: int
    price: int            # in cents (1-99)
    status: str
    created_at: str
    client_order_id: str = ""
    error: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


# ─── Client ───────────────────────────────────────────────────────────────────

class KalshiClient:
    """Client for the Kalshi Trade API v2."""

    def __init__(
        self,
        api_key_id: str,
        private_key_path: str,
        use_demo: bool = False,
    ):
        self.api_key_id = api_key_id
        self.private_key_pem = _load_private_key(private_key_path)

        if use_demo:
            self.base_url = "https://external-api.demo.kalshi.co/trade-api/v2"
        else:
            self.base_url = "https://external-api.kalshi.com/trade-api/v2"

        log.info(
            "KalshiClient initialized (demo=%s, key_id=%s)",
            use_demo,
            api_key_id[:8] + "..." if api_key_id else "???",
        )

    # ── Auth request helpers ──────────────────────────────────────────────

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """Make an authenticated GET request."""
        headers = _auth_headers(self.api_key_id, self.private_key_pem, "GET", path, self.base_url)
        url = self.base_url + path
        log.debug("GET %s params=%s", url, params)
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: Dict) -> Dict:
        """Make an authenticated POST request."""
        headers = _auth_headers(self.api_key_id, self.private_key_pem, "POST", path, self.base_url)
        headers["Content-Type"] = "application/json"
        url = self.base_url + path
        log.debug("POST %s data=%s", url, data)
        resp = requests.post(url, headers=headers, json=data, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str) -> Dict:
        """Make an authenticated DELETE request."""
        headers = _auth_headers(self.api_key_id, self.private_key_pem, "DELETE", path, self.base_url)
        url = self.base_url + path
        log.debug("DELETE %s", url)
        resp = requests.delete(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ── Account ──────────────────────────────────────────────────────────

    def get_balance(self) -> float:
        """Return available balance in USD."""
        data = self._get("/portfolio/balance")
        # Balance is in cents
        return data.get("balance", 0) / 100.0

    def get_positions(self) -> List[Dict]:
        """Return all open positions."""
        data = self._get("/portfolio/positions")
        return data.get("positions", [])

    # ── Market queries ───────────────────────────────────────────────────

    def get_markets(
        self,
        limit: int = 100,
        status: str = "open",
        event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        min_liquidity: float = 0.0,
    ) -> List[Dict]:
        """Fetch open markets from Kalshi.

        Args:
            limit: Max number of markets to return (API max ~100)
            status: Market status filter
            event_ticker: Filter by event ticker
            series_ticker: Filter by series ticker
            min_liquidity: Minimum liquidity in dollars

        Returns:
            List of market dicts
        """
        params: Dict[str, Any] = {
            "limit": limit,
            "status": status,
        }
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker

        data = self._get("/markets", params=params)
        markets = data.get("markets", [])

        if min_liquidity > 0:
            markets = [
                m for m in markets
                if m.get("volume", {}).get("bid", 0) + m.get("volume", {}).get("ask", 0) >= min_liquidity
            ]

        return markets

    def get_events(self, sport: str = "baseball", status: str = "open") -> List[Dict]:
        """Fetch Kalshi events (grouped markets) filtered by sport.

        Each event typically has an event_ticker like KXMLBGAME-2605111810LAACLE
        for individual game markets.
        """
        data = self._get("/events", params={"sport": sport, "status": status, "limit": 100})
        return data.get("events", [])

    def get_event_markets(self, event_ticker: str, min_liquidity: float = 0) -> List[Dict]:
        """Fetch markets for a single event by event_ticker."""
        data = self._get("/markets", params={
            "event_ticker": event_ticker,
            "limit": 100,
            "status": "open",
        })
        markets = data.get("markets", [])

        # Kalshi's liquidity_dollars is often "0.0000" (string).
        # Use volume_fp (cumulative volume in contracts) as a better proxy.
        if min_liquidity > 0:
            filtered = []
            for m in markets:
                try:
                    vol = float(m.get("volume_fp", 0))
                except (ValueError, TypeError):
                    vol = 0
                if vol >= min_liquidity:
                    filtered.append(m)
            markets = filtered

        return markets

    def search_markets(self, text: str, limit: int = 100, min_liquidity: float = 0) -> List[Dict]:
        """Search for markets matching text in their title/description.

        Useful for finding specific game or player markets.
        """
        data = self._get("/markets", params={
            "limit": limit,
            "status": "open",
            "cursor": None,
        })
        markets = data.get("markets", [])

        # Filter by text match
        text_lower = text.lower()
        result = []
        for m in markets:
            searchable = f"{m.get('title', '')} {m.get('description', '')} {m.get('ticker', '')}".lower()
            if text_lower in searchable:
                result.append(m)

        # Filter by liquidity
        if min_liquidity > 0:
            result = [m for m in result if float(m.get("liquidity_dollars", 0)) >= min_liquidity]

        return result

    def get_market(self, ticker: str) -> Optional[Dict]:
        """Get a single market by ticker."""
        try:
            data = self._get(f"/markets/{ticker}")
            return data.get("market")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                log.warning("Market %s not found", ticker)
                return None
            raise

    # ── Order placement (V2 API) ──────────────────────────────────────────

    def create_order(
        self,
        ticker: str,
        side: str,
        count: int,
        price_cents: int,
        client_order_id: Optional[str] = None,
        time_in_force: str = "good_till_canceled",
    ) -> OrderResult:
        """Place a limit order using the Kalshi V2 event-markets endpoint.

        V2 API shape (POST /portfolio/events/orders):
          - `side`  : "bid" (buy YES) or "ask" (sell YES / buy NO)
          - `price` : fixed-point dollar string, e.g. "0.07" for 7¢
          - `count` : contract quantity string, e.g. "3"
          No `action` or `type` fields — all V2 orders are limit orders;
          direction is fully encoded in `side`.

        Args:
            ticker:      Market ticker
            side:        "yes" → bid, "no" → ask
            count:       Number of contracts
            price_cents: Limit price in cents (1–99)
            client_order_id: Optional idempotency key
            time_in_force: "good_till_canceled" | "fill_or_kill" | "immediate_or_cancel"
        """
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        book_side = "bid" if side == "yes" else "ask"
        price_str = f"{price_cents / 100:.4f}"   # "0.0700" for 7¢

        payload: Dict[str, Any] = {
            "ticker":                    ticker,
            "client_order_id":           client_order_id,
            "side":                      book_side,
            "count":                     str(count),
            "price":                     price_str,
            "time_in_force":             time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
        }

        log.debug("V2 order payload: %s", payload)

        try:
            result = self._post("/portfolio/events/orders", data=payload)
            # V2 response: {order_id, fill_count, remaining_count, ...}
            order_id    = result.get("order_id", "")
            fill_count  = result.get("fill_count", "0")
            remaining   = result.get("remaining_count", str(count))
            filled      = float(fill_count) if fill_count else 0.0
            status      = "executed" if float(remaining) == 0 else (
                          "resting"  if filled == 0 else "partially_filled")
            return OrderResult(
                order_id=order_id,
                ticker=ticker,
                action="buy" if book_side == "bid" else "sell",
                side=side,
                count=count,
                price=price_cents,
                status=status,
                created_at="",
                client_order_id=result.get("client_order_id", client_order_id),
                detail=result,
            )
        except requests.HTTPError as e:
            error_body = {}
            if e.response is not None:
                try:
                    error_body = e.response.json()
                except Exception:
                    pass
            log.error(
                "Order failed [%s]: %s — full response: %s",
                e.response.status_code if e.response is not None else "?",
                error_body.get("message", str(e)),
                error_body,
            )
            return OrderResult(
                order_id="",
                ticker=ticker,
                action="buy" if side == "yes" else "sell",
                side=side,
                count=count,
                price=0,
                status="failed",
                created_at="",
                client_order_id=client_order_id,
                error=str(e),
                detail=error_body,
            )

    def create_yes_order(
        self,
        ticker: str,
        count: int,
        yes_price: int,
        action: str = "buy",          # kept for call-site compatibility
        order_type: str = "limit",    # kept for call-site compatibility
    ) -> OrderResult:
        """Buy YES contracts at the given limit price (cents)."""
        return self.create_order(
            ticker=ticker,
            side="yes",
            count=count,
            price_cents=yes_price,
        )

    def create_no_order(
        self,
        ticker: str,
        count: int,
        no_price: int,
        action: str = "buy",
        order_type: str = "limit",
    ) -> OrderResult:
        """Buy NO contracts at the given limit price (cents)."""
        return self.create_order(
            ticker=ticker,
            side="no",
            count=count,
            price_cents=no_price,
        )

    # ── Settlement & fill history ─────────────────────────────────────────

    def get_settlements(
        self,
        limit: int = 200,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        ticker: Optional[str] = None,
    ) -> List[Dict]:
        """Return settled market records for this account.

        Each record contains: ticker, market_result ('yes'/'no'/'void'),
        revenue (cents), yes_total_cost_dollars, fee_cost, settled_time.
        """
        params: Dict[str, Any] = {"limit": limit}
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        if ticker:
            params["ticker"] = ticker

        data = self._get("/portfolio/settlements", params=params)
        return data.get("settlements", [])

    def get_fills(
        self,
        limit: int = 200,
        ticker: Optional[str] = None,
        order_id: Optional[str] = None,
        min_ts: Optional[int] = None,
    ) -> List[Dict]:
        """Return fill records (matched order legs) for this account."""
        params: Dict[str, Any] = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if order_id:
            params["order_id"] = order_id
        if min_ts is not None:
            params["min_ts"] = min_ts

        data = self._get("/portfolio/fills", params=params)
        return data.get("fills", [])

    # ── Multivariate (combo / parlay) markets ────────────────────────────

    def get_multivariate_collections(
        self,
        status: str = "open",
        associated_event_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict]:
        """Fetch multivariate event collections (combo templates).

        A collection defines which single-leg events can be combined into
        a combo market.  For example, an MLB same-game-parlay collection
        links K, HR, and game-winner events for a single game.

        Args:
            status: "unopened", "open", or "closed"
            associated_event_ticker: Only collections that include this event
            series_ticker: Only collections for this series
            limit: Max results (API max 200)

        Returns:
            List of MultivariateEventCollection dicts, each containing:
              - collection_ticker, title, description
              - associated_events (list of {ticker, is_yes_only, ...})
              - size_min / size_max (allowed number of legs)
              - functional_description (how legs combine)
        """
        params: Dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if associated_event_ticker:
            params["associated_event_ticker"] = associated_event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker

        data = self._get("/multivariate_event_collections", params=params)
        return data.get("multivariate_contracts", [])

    def get_multivariate_collection(self, collection_ticker: str) -> Optional[Dict]:
        """Fetch a single multivariate event collection by its ticker.

        Returns the full collection dict including associated_events,
        size constraints, and functional_description, or None on 404.
        """
        try:
            data = self._get(f"/multivariate_event_collections/{collection_ticker}")
            return data.get("multivariate_contract")
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                log.debug("Multivariate collection %s not found", collection_ticker)
                return None
            raise

    def get_multivariate_events(
        self,
        collection_ticker: Optional[str] = None,
        series_ticker: Optional[str] = None,
        with_nested_markets: bool = False,
        limit: int = 200,
    ) -> List[Dict]:
        """Fetch multivariate (combo) events.

        These are the dynamically-created events that result from users
        selecting legs within a multivariate event collection.

        Args:
            collection_ticker: Filter to combos from this collection
            series_ticker: Filter to combos from this series
            with_nested_markets: Include full market objects in each event
            limit: Max results (API max 200)

        Returns:
            List of EventData dicts.  When *with_nested_markets* is True
            each event includes a ``markets`` list with full market objects
            (ticker, prices, volume, etc.).
        """
        params: Dict[str, Any] = {"limit": limit}
        if collection_ticker:
            params["collection_ticker"] = collection_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if with_nested_markets:
            params["with_nested_markets"] = "true"

        data = self._get("/events/multivariate", params=params)
        return data.get("events", [])

    def create_combo_market(
        self,
        collection_ticker: str,
        legs: List[Dict[str, str]],
        with_market_payload: bool = True,
    ) -> Optional[Dict]:
        """Create (or retrieve) a combo market from selected legs.

        This must be called at least once before you can trade a particular
        combination.  If the combo already exists, Kalshi returns the
        existing market/event tickers (idempotent).

        Args:
            collection_ticker: The MVE collection that defines which legs
                               are combinable (e.g. "KXMLBSGP")
            legs: List of leg dicts, each with:
                  - market_ticker:  ticker of the single-leg market
                  - event_ticker:   event that market belongs to
                  - side:           "yes" or "no"
            with_market_payload: Include full market data in response

        Returns:
            Dict with keys:
              - event_ticker:  event ticker for the combo
              - market_ticker: market ticker for the combo
              - market:        full market dict (if with_market_payload)
            Returns None on failure.

        Raises:
            requests.HTTPError on 4xx/5xx (except logged gracefully)

        Note:
            Users are limited to 5 000 combo market creations per week.
        """
        payload: Dict[str, Any] = {
            "selected_markets": [
                {
                    "market_ticker": leg["market_ticker"],
                    "event_ticker":  leg["event_ticker"],
                    "side":          leg["side"],
                }
                for leg in legs
            ],
            "with_market_payload": with_market_payload,
        }

        path = f"/multivariate_event_collections/{collection_ticker}"
        log.info(
            "Creating combo market: collection=%s  legs=%d",
            collection_ticker, len(legs),
        )
        for i, leg in enumerate(legs):
            log.debug(
                "  Leg %d: %s (%s) side=%s",
                i + 1, leg["market_ticker"], leg["event_ticker"], leg["side"],
            )

        try:
            result = self._post(path, data=payload)
            market_ticker = result.get("market_ticker", "")
            event_ticker  = result.get("event_ticker", "")
            log.info(
                "Combo market ready: event=%s  market=%s",
                event_ticker, market_ticker,
            )
            return result
        except requests.HTTPError as e:
            error_body = {}
            if e.response is not None:
                try:
                    error_body = e.response.json()
                except Exception:
                    pass
            log.error(
                "Combo market creation failed [%s]: %s — %s",
                e.response.status_code if e.response is not None else "?",
                error_body.get("message", str(e)),
                error_body,
            )
            return None

    # ── Legacy order endpoint (fallback) ──────────────────────────────────

    def create_order_legacy(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        order_type: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
    ) -> OrderResult:
        """Fallback: place order using the legacy /portfolio/orders endpoint."""
        payload: Dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
            "client_order_id": str(uuid.uuid4()),
        }

        if order_type == "limit":
            if side == "yes" and yes_price is not None:
                payload["yes_price"] = yes_price
            elif side == "no" and no_price is not None:
                payload["no_price"] = no_price

        try:
            result = self._post("/portfolio/orders", data=payload)
            order = result.get("order", {})
            return OrderResult(
                order_id=order.get("order_id", ""),
                ticker=ticker,
                action=action,
                side=side,
                count=count,
                price=order.get("price", 0),
                status=order.get("status", ""),
                created_at=order.get("created_at", ""),
                detail=order,
            )
        except requests.HTTPError as e:
            error_body = {}
            if e.response is not None:
                try:
                    error_body = e.response.json()
                except Exception:
                    pass
            log.error("Legacy order failed: %s", error_body.get("message", str(e)))
            return OrderResult(
                order_id="",
                ticker=ticker,
                action=action,
                side=side,
                count=count,
                price=0,
                status="failed",
                created_at="",
                error=str(e),
                detail=error_body,
            )

# ─── Market price extraction ──────────────────────────────────────────────────

def market_price(market: dict) -> int:
    """Extract YES ask price in cents from a Kalshi market dict.

    Handles multiple Kalshi response formats:
      - Flat dollar string: {"yes_ask_dollars": "0.35"} → 35
      - Nested dict:        {"ask": {"price": 35}}      → 35
      - Numeric:            {"ask": 35}                  → 35
    """
    if "yes_ask_dollars" in market:
        try:
            return int(float(market["yes_ask_dollars"]) * 100)
        except (ValueError, TypeError):
            pass
    ask = market.get("ask")
    if isinstance(ask, dict):
        return int(ask.get("price", 0))
    if isinstance(ask, (int, float)):
        return int(ask)
    return 0
