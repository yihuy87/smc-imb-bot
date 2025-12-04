# imb/htf_context.py
# Ambil konteks HTF (15m & 1h) sederhana tanpa indikator:
# - trend UP / DOWN / RANGE di 1h
# - posisi harga di dalam range (DISCOUNT / PREMIUM / MID) untuk 1h & 15m
# Sekarang pakai caching per-symbol & per-timeframe:
#   - 1h  : TTL 3600 detik
#   - 15m : TTL 900 detik

from typing import Dict, List, Literal, Optional
import time

import requests

from config import BINANCE_REST_URL

# Cache HTF: key = "SYMBOL_INTERVAL" (contoh: "BTCUSDT_1h")
# value = {"ts": float, "data": List[dict]}
_HTF_CACHE: Dict[str, Dict[str, object]] = {}


def _fetch_klines(
    symbol: str,
    interval: str,
    limit: int = 150,
    ttl_seconds: Optional[int] = None,
) -> Optional[List[dict]]:
    """
    Fetch klines dari Binance dengan optional cache TTL per symbol+interval.

    ttl_seconds:
      - None atau 0  → selalu fetch baru (tanpa cache)
      - > 0          → pakai cache jika belum lewat TTL
    """
    sym_u = symbol.upper()
    cache_key = f"{sym_u}_{interval}"

    # Cek cache
    if ttl_seconds and ttl_seconds > 0:
        cached = _HTF_CACHE.get(cache_key)
        if cached:
            ts = cached.get("ts", 0.0)
            if time.time() - ts < ttl_seconds:
                data = cached.get("data")
                if isinstance(data, list) and data:
                    return data  # pakai data cache

    url = f"{BINANCE_REST_URL}/fapi/v1/klines"
    params = {"symbol": sym_u, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        # Simpan ke cache kalau TTL dipakai
        if ttl_seconds and ttl_seconds > 0 and isinstance(data, list) and data:
            _HTF_CACHE[cache_key] = {
                "ts": time.time(),
                "data": data,
            }

        return data
    except Exception as e:
        print(f"[{sym_u}] ERROR fetch HTF klines ({interval}):", e)
        return None


def _parse_ohlc(data: List[dict]) -> Dict[str, List[float]]:
    highs: List[float] = []
    lows: List[float] = []
    closes: List[float] = []
    for row in data:
        try:
            h = float(row[2])
            l = float(row[3])
            c = float(row[4])
        except (ValueError, TypeError, IndexError):
            continue
        highs.append(h)
        lows.append(l)
        closes.append(c)
    return {"high": highs, "low": lows, "close": closes}


def _detect_trend_1h(hlc: Dict[str, List[float]]) -> Literal["UP", "DOWN", "RANGE"]:
    highs = hlc["high"]
    lows = hlc["low"]
    n = len(highs)
    if n < 20:
        return "RANGE"

    # ambil beberapa swing kasar: pakai grid sederhana
    step = max(n // 10, 2)
    swing_highs = highs[::step]
    swing_lows = lows[::step]
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "RANGE"

    first_h = swing_highs[0]
    last_h = swing_highs[-1]
    first_l = swing_lows[0]
    last_l = swing_lows[-1]

    # threshold kecil supaya tidak noise
    if last_h > first_h * 1.01 and last_l > first_l * 1.005:
        return "UP"
    if last_h < first_h * 0.99 and last_l < first_l * 0.995:
        return "DOWN"
    return "RANGE"


def _discount_premium(
    hlc: Dict[str, List[float]],
    window: int = 60,
) -> Dict[str, object]:
    highs = hlc["high"]
    lows = hlc["low"]
    closes = hlc["close"]
    n = len(highs)
    if n < 5:
        return {
            "position": "MID",
            "range_high": None,
            "range_low": None,
            "price": closes[-1] if closes else None,
        }

    start = max(0, n - window)
    seg_high = highs[start:]
    seg_low = lows[start:]
    price = closes[-1]

    range_high = max(seg_high)
    range_low = min(seg_low)
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
    Ambil konteks 1h & 15m untuk symbol (tanpa indikator klasik).
    Return dict:

    {
      "trend_1h": "UP"|"DOWN"|"RANGE",
      "pos_1h": "DISCOUNT"|"PREMIUM"|"MID",
      "pos_15m": "DISCOUNT"|"PREMIUM"|"MID",
      "htf_ok_long": bool,
      "htf_ok_short": bool,
    }

    Jika gagal fetch data, semua dianggap NETRAL (htf_ok_long/short = True).
    """

    # default netral
    ctx = {
        "trend_1h": "RANGE",
        "pos_1h": "MID",
        "pos_15m": "MID",
        "htf_ok_long": True,
        "htf_ok_short": True,
    }

    # TTL berbeda:
    # - 1h  : 3600 detik
    # - 15m : 900 detik
    data_1h = _fetch_klines(symbol, "1h", 150, ttl_seconds=3600)
    data_15m = _fetch_klines(symbol, "15m", 150, ttl_seconds=900)

    if not data_1h or not data_15m:
        return ctx  # tetap netral kalau gagal

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

    return {
        "trend_1h": trend_1h,
        "pos_1h": pos_1h,
        "pos_15m": pos_15m,
        "htf_ok_long": htf_ok_long,
        "htf_ok_short": htf_ok_short,
        }
