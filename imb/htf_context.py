# imb/htf_context.py
# Ambil konteks HTF (15m & 1h) dengan cache:
# - trend UP / DOWN / RANGE di 1h
# - posisi harga di dalam range (DISCOUNT / PREMIUM / MID) untuk 1h & 15m

from typing import Dict, List, Literal, Optional
import time

import requests

from config import BINANCE_REST_URL
from core.imb_settings import imb_settings


# ====== CONFIG CACHE ======
TTL_1H = 3600      # 1 jam
TTL_15M = 900      # 15 menit

# Struktur cache per symbol:
# {
#   "ctx": {...},
#   "ts_1h": float (epoch),
#   "ts_15m": float (epoch)
# }
_HTF_CACHE: Dict[str, Dict[str, object]] = {}


def _fetch_klines(symbol: str, interval: str, limit: int = 150) -> Optional[List[dict]]:
    url = f"{BINANCE_REST_URL}/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data
    except Exception as e:
        print(f"[{symbol}] ERROR fetch HTF klines ({interval}):", e)
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


def _neutral_ctx() -> Dict[str, object]:
    return {
        "trend_1h": "RANGE",
        "pos_1h": "MID",
        "pos_15m": "MID",
        "htf_ok_long": True,
        "htf_ok_short": True,
    }


def get_htf_context(symbol: str) -> Dict[str, object]:
    """
    Versi cached:
    - Kalau IMB_USE_HTF_FILTER = false → langsung netral (tidak pakai HTF)
    - Kalau true → pakai cache per symbol, refresh:
        - 1h setiap ≥ 1 jam
        - 15m setiap ≥ 15 menit
    """
    # kalau HTF filter dimatikan di config → selalu netral, tanpa REST call
    if not imb_settings.use_htf_filter:
        return _neutral_ctx()

    now = time.time()
    sym = symbol.upper()

    cache = _HTF_CACHE.get(sym)
    if cache is None:
        cache = {
            "ctx": _neutral_ctx(),
            "ts_1h": 0.0,
            "ts_15m": 0.0,
        }
        _HTF_CACHE[sym] = cache

    ctx = cache["ctx"]
    ts_1h = float(cache.get("ts_1h", 0.0))
    ts_15m = float(cache.get("ts_15m", 0.0))

    trend_1h = ctx.get("trend_1h", "RANGE")
    pos_1h = ctx.get("pos_1h", "MID")
    pos_15m = ctx.get("pos_15m", "MID")

    # --- UPDATE 1H JIKA PERLU ---
    if now - ts_1h >= TTL_1H or ts_1h == 0.0:
        data_1h = _fetch_klines(sym, "1h", 150)
        if data_1h:
            hlc_1h = _parse_ohlc(data_1h)
            trend_1h = _detect_trend_1h(hlc_1h)
            pos1 = _discount_premium(hlc_1h)
            pos_1h = pos1["position"]
            ts_1h = now  # update timestamp hanya kalau berhasil

    # --- UPDATE 15M JIKA PERLU ---
    if now - ts_15m >= TTL_15M or ts_15m == 0.0:
        data_15m = _fetch_klines(sym, "15m", 150)
        if data_15m:
            hlc_15m = _parse_ohlc(data_15m)
            pos15 = _discount_premium(hlc_15m)
            pos_15m = pos15["position"]
            ts_15m = now

    # kalau dua-duanya gagal fetch → pakai nilai lama di ctx (atau netral awal)
    # bangun rule align
    htf_ok_long = not (trend_1h == "DOWN" and pos_1h == "PREMIUM")
    if pos_1h == "PREMIUM" and pos_15m == "PREMIUM":
        htf_ok_long = False

    htf_ok_short = not (trend_1h == "UP" and pos_1h == "DISCOUNT")
    if pos_1h == "DISCOUNT" and pos_15m == "DISCOUNT":
        htf_ok_short = False

    new_ctx = {
        "trend_1h": trend_1h,
        "pos_1h": pos_1h,
        "pos_15m": pos_15m,
        "htf_ok_long": htf_ok_long,
        "htf_ok_short": htf_ok_short,
    }

    cache["ctx"] = new_ctx
    cache["ts_1h"] = ts_1h
    cache["ts_15m"] = ts_15m

    return new_ctx
