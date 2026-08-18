"""Asyncio Kalshi websocket client for the Phase 0 recorder.

Mirrors the auth/subscribe protocol already proven in
dashboard/src/kalshi-ws.ts, but in Python so the recorder can stamp
receive times and write JSONL alongside the GUMBO feed.

Read-only: subscribes to public market-data channels, never places orders.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Callable, Dict, List, Optional

import websockets

from slugger.kalshi_client import _sign_request

log = logging.getLogger(__name__)

WS_PATH = "/trade-api/ws/v2"

# Public market-data channels we record
CHANNELS = ["ticker", "trade", "orderbook_delta"]

# Kalshi caps market_tickers per subscribe command; batch conservatively.
SUBSCRIBE_BATCH = 40


class KalshiWSRecorder:
    """Connects, subscribes, and forwards every message to a callback.

    on_message receives the parsed message dict plus recv_ts (epoch float
    captured the moment the frame arrived).
    """

    def __init__(
        self,
        api_key_id: str,
        private_key_pem: str,
        market_tickers: List[str],
        on_message: Callable[[dict, float], None],
        use_demo: bool = False,
    ):
        self.api_key_id = api_key_id
        self.private_key_pem = private_key_pem
        self.market_tickers = list(dict.fromkeys(market_tickers))  # dedupe, keep order
        self.on_message = on_message
        if use_demo:
            self.ws_url = f"wss://external-api-ws.demo.kalshi.co{WS_PATH}"
        else:
            self.ws_url = f"wss://external-api-ws.kalshi.com{WS_PATH}"
        self._next_cmd_id = 1
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    def _auth_headers(self) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        signature = _sign_request(
            self.private_key_pem, timestamp, "GET", "", self.ws_url
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    async def run(self) -> None:
        """Connect with auto-reconnect until stop() is called."""
        if not self.market_tickers:
            log.warning("KalshiWSRecorder: no market tickers to record")
            return

        backoff = 1.0
        while not self._stop.is_set():
            try:
                async with websockets.connect(
                    self.ws_url,
                    additional_headers=self._auth_headers(),
                    ping_interval=10,
                    ping_timeout=20,
                    max_size=2**22,
                ) as ws:
                    log.info(
                        "Kalshi WS connected (%d markets)", len(self.market_tickers)
                    )
                    backoff = 1.0
                    await self._subscribe(ws)
                    await self._read_loop(ws)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self._stop.is_set():
                    break
                log.warning("Kalshi WS error: %s — reconnecting in %.0fs", e, backoff)
                self.on_message(
                    {"type": "recorder_disconnect", "error": str(e)}, time.time()
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 30.0)
        log.info("Kalshi WS recorder stopped")

    async def _subscribe(self, ws) -> None:
        for i in range(0, len(self.market_tickers), SUBSCRIBE_BATCH):
            batch = self.market_tickers[i : i + SUBSCRIBE_BATCH]
            cmd_id = self._next_cmd_id
            self._next_cmd_id += 1
            await ws.send(json.dumps({
                "id": cmd_id,
                "cmd": "subscribe",
                "params": {
                    "channels": CHANNELS,
                    "market_tickers": batch,
                    "send_initial_snapshot": True,
                },
            }))

    async def _read_loop(self, ws) -> None:
        while not self._stop.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
            except asyncio.TimeoutError:
                # ping_interval keeps the connection alive; a quiet minute is
                # normal between games. Just keep waiting.
                continue
            recv_ts = time.time()
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                log.debug("Unparseable WS frame: %r", raw[:200])
                continue
            try:
                self.on_message(msg, recv_ts)
            except Exception:
                log.exception("on_message handler failed")
