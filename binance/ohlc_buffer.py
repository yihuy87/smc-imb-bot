# binance/ohlc_buffer.py
# Buffer OHLC 5m per symbol dari WebSocket futures.

from collections import deque
from typing import Deque, Dict, List, TypedDict


class Candle(TypedDict):
    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool


class OHLCBufferManager:
    def __init__(self, max_candles: int = 300) -> None:
        self.max_candles = max_candles
        self._buffers: Dict[str, Deque[Candle]] = {}

    def _get_buffer(self, symbol: str) -> Deque[Candle]:
        if symbol not in self._buffers:
            self._buffers[symbol] = deque(maxlen=self.max_candles)
        return self._buffers[symbol]

    def update_from_kline(self, symbol: str, kline: dict) -> None:
        buf = self._get_buffer(symbol)

        open_time = int(kline.get("t", 0))
        close_time = int(kline.get("T", 0))
        try:
            o = float(kline.get("o", "0"))
            h = float(kline.get("h", "0"))
            l = float(kline.get("l", "0"))
            c = float(kline.get("c", "0"))
            v = float(kline.get("v", "0"))
        except ValueError:
            return

        closed = bool(kline.get("x", False))

        candle: Candle = {
            "open_time": open_time,
            "close_time": close_time,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "closed": closed,
        }

        if not buf:
            buf.append(candle)
            return

        last = buf[-1]
        if last["open_time"] == open_time:
            buf[-1] = candle
        else:
            buf.append(candle)

    def get_candles(self, symbol: str) -> List[Candle]:
        return list(self._get_buffer(symbol))

    def preload_candles(self, symbol: str, klines: list[list]) -> None:
        """
        Preload dari REST fapi/v1/klines (list raw Binance array).
        """
        buf = self._get_buffer(symbol)
        buf.clear()
        for row in klines:
            try:
                o = float(row[1])
                h = float(row[2])
                l = float(row[3])
                c = float(row[4])
                v = float(row[5])
            except (ValueError, IndexError):
                continue
            candle: Candle = {
                "open_time": int(row[0]),
                "close_time": int(row[6]),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
                "closed": True,
            }
            buf.append(candle)
