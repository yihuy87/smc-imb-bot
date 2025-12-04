# imb/htf_context.py
# Ambil konteks HTF (15m & 1h) sederhana tanpa indikator:
# - trend UP / DOWN / RANGE di 1h
# - posisi harga di dalam range (DISCOUNT / PREMIUM / MID) untuk 1h & 15m

from typing import Dict, List, Literal, Optional
import time

import numpy as np
import requests

from config import BINANCE_REST_URL


# Cache HTF per symbol (biar tidak spam REST ke Binance)
HTF_CACHE_TTL = 120  # detik
_htf_cache: Dict[str, Dict[str, object]] = {}


def _fetch_klines(symbol: str, interval: str, limit: int = 150) -> Optional[List[dict]]:
    url = f"{BINANCE_REST_URL}/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[{symbol}] ERROR fetch HTF klines ({interval}):", e)
        return None


def _parse_ohlc(data: List[dict]) -> Dict[str, np.ndarray]:
    highs = []
    lows = []
    closes = []

    for row in data:
        try:
            highs.append(float(row[2]))
            lows.append(float(row[3]))
            closes.append(float(row[4]))
        except Exception:
            continue

    return {
        "high": np.asarray(highs, dtype=float),
        "low": np.asarray(lows, dtype=float),
        "close": np.asarray(closes, dtype=float),
    }


def _detect_trend_1h(hlc: Dict[str, np.ndarray]) -> Literal["UP", "DOWN", "RANGE"]:
    highs = hlc["high"]
    lows = hlc["low"]
    n = highs.size

    if n < 20:
        return "RANGE"

    step = max(n // 10, 2)

    swing_highs = highs[::step]
    swing_lows = lows[::step]

    if swing_highs.size < 3 or swing_lows.size < 3:
        return "RANGE"

    first_h = swing_highs[0]
    last_h = swing_highs[-1]
    first_l = swing_lows[0]
    last_l = swing_lows[-1]

    # threshold kecil untuk menghindari noise
    if last_h > first_h * 1.01 and last_l > first_l * 1.005:
        return "UP"
    if last_h < first_h * 0.99 and last_l < first_l * 0.995:
        return "DOWN"

    return "RANGE"


def _discount_premium(
    hlc: Dict[str, np.ndarray],
    window: int = 60,
) -> Dict[str, object]:
    highs = hlc["high"]
    lows = hlc["low"]
    closes = hlc["close"]

    n = highs.size
    if n < 5:
        return {
            "position": "MID",
            "range_high": None,
            "range_low": None,
            "price": float(closes[-1]) if closes.size > 0 else None,
        }

    start = max(0, n - window)

    seg_high = highs[start:]
    seg_low = lows[start:]
    price = float(closes[-1])

    range_high = float(seg_high.max())
    range_low = float(seg_low.min())

    if range_high <= range_low:
        return {
            "position": "MID",
            "range_high": range_high,
            "range_low": range_low,
            "price": price,
        }

    pos = (price - range_low) / (range_high - range_low)

    if pos <= 0.35:
        position = "DISCOUNT"
    elif pos >= 0.65:
        position = "PREMIUM"
    else:
        position = "MID"

    return {
        "position": position,
        "range_high": range_high,
        "range_low": range_low,
        "price": price,
    }


def get_htf_context(symbol: str) -> Dict[str, object]:
    """
    Ambil konteks 1h & 15m untuk symbol (tanpa indikator klasik),
    dengan caching per symbol supaya tidak spam REST.
    """
    symbol_u = symbol.upper()
    now = time.time()

    # Cek cache dulu
    cached = _htf_cache.get(symbol_u)
    if cached:
        ts = cached.get("ts", 0.0)
        ctx_cached = cached.get("ctx")
        if ctx_cached is not None and (now - ts) < HTF_CACHE_TTL:
            return ctx_cached  # langsung pakai cached context

    # Default context (NETRAL)
    ctx_default = {
        "trend_1h": "RANGE",
        "pos_1h": "MID",
        "pos_15m": "MID",
        "htf_ok_long": True,
        "htf_ok_short": True,
    }

    data_1h = _fetch_klines(symbol_u, "1h", 150)
    data_15m = _fetch_klines(symbol_u, "15m", 150)

    if not data_1h or not data_15m:
        # Cache juga default supaya tidak spam request saat error
        _htf_cache[symbol_u] = {"ts": now, "ctx": ctx_default}
        return ctx_default

    hlc_1h = _parse_ohlc(data_1h)
    hlc_15m = _parse_ohlc(data_15m)

    trend_1h = _detect_trend_1h(hlc_1h)
    pos1 = _discount_premium(hlc_1h)
    pos15 = _discount_premium(hlc_15m)

    pos_1h = pos1["position"]
    pos_15m = pos15["position"]

    # aturan sederhana:
    # LONG ideal: 1h bukan DOWN kuat + 1h & 15m bukan PREMIUM
    # SHORT ideal: 1h bukan UP kuat + 1h & 15m bukan DISCOUNT
    htf_ok_long = not (trend_1h == "DOWN" and pos_1h == "PREMIUM")
    if pos_1h == "PREMIUM" and pos_15m == "PREMIUM":
        htf_ok_long = False

    htf_ok_short = not (trend_1h == "UP" and pos_1h == "DISCOUNT")
    if pos_1h == "DISCOUNT" and pos_15m == "DISCOUNT":
        htf_ok_short = False

    ctx_final = {
        "trend_1h": trend_1h,
        "pos_1h": pos_1h,
        "pos_15m": pos_15m,
        "htf_ok_long": htf_ok_long,
        "htf_ok_short": htf_ok_short,
    }

    # Simpan ke cache
    _htf_cache[symbol_u] = {"ts": now, "ctx": ctx_final}
    return ctx_final
