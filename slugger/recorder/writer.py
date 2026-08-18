"""Append-only JSONL writer with receive timestamps."""
from __future__ import annotations

import datetime
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict


class JsonlWriter:
    """Thread-safe append-only JSONL writer.

    Every record is stamped with:
      recv_ts  - local wall-clock epoch seconds (float) at write time,
                 unless the caller already supplied one (preferred: stamp
                 at receive time, not write time)
      recv_iso - same instant, ISO-8601 UTC, for human eyes
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: Dict[str, Any]) -> None:
        if "recv_ts" not in record:
            record["recv_ts"] = time.time()
        record["recv_iso"] = (
            datetime.datetime.fromtimestamp(
                record["recv_ts"], tz=datetime.timezone.utc
            ).isoformat()
        )
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
