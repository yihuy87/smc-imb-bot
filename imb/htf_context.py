# imb/htf_context.py
# HTF context sederhana (1h & 15m) tanpa indikator.

from typing import Dict, List, Optional, Literal
import requests

from config import BINANCE_REST_URL


def _fetch_klines(symbol: str, interval: str, limit: int = 150) -> Optional[List[list]]:
    url = f"{BINANCE_REST_URL}/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[{symbol}] ERROR fetch HTF klines ({interval}):", e)
        return None


def _parse_hlc(data: List[list]) -> Dict[str, List[float]]:
    highs, lows, closes = [], [], []
    for row in data:
        try:
            highs.append(float(row[2]))
            lows.append(float(row[3]))
            closes.append(float(row[4]))
        except (ValueError, IndexError):
            continue
    return {"high": highs, "low": lows, "close": closes}


def _detect_trend_1h(hlc: Dict[str, List[float]]) -> Literal["UP", "DOWN", "RANGE"]:
    highs = hlc["high"]
    lows = hlc["low"]
    n = len(highs)
    if n < 20:
        return "RANGE"
    step = max(n // 10, 2)
    swing_highs = highs[::step]
    swing_lows = lows[::step]
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "RANGE"
    first_h, last_h = swing_highs[0], swing_highs[-1]
    first_l, last_l = swing_lows[0], swing_lows[-1]
    if last_h > first_h * 1.01 and last_l > first_l * 1.005:
        return "UP"
    if last_h < first_h * 0.99 and last_l < first_l * 0.995:
        return "DOWN"
    return "RANGE"


def _discount_premium(hlc: Dict[str, List[float]], window: int = 60) -> Dict[str, object]:
    highs = hlc["high"]
    lows = hlc["low"]
    closes = hlc["close"]
    n = len(highs)
    if n < 5:
        return {"position": "MID", "range_high": None, "range_low": None, "price": closes[-1] if closes else None}

    start = max(0, n - window)
    seg_high = highs[start:]
    seg_low = lows[start:]
    price = closes[-1]
    range_high = max(seg_high)
    range_low = min(seg_low)
    if range_high <= range_low:
        return {"position": "MID", "range_high": range_high, "range_low": range_low, "price": price}

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
    ctx = {
        "trend_1h": "RANGE",
        "pos_1h": "MID",
        "pos_15m": "MID",
        "htf_ok_long": True,
        "htf_ok_short": True,
    }

    data_1h = _fetch_klines(symbol, "1h", 150)
    data_15m = _fetch_klines(symbol, "15m", 150)

    if not data_1h or not data_15m:
        return ctx

    hlc_1h = _parse_hlc(data_1h)
    hlc_15m = _parse_hlc(data_15m)

    trend_1h = _detect_trend_1h(hlc_1h)
    pos1 = _discount_premium(hlc_1h)
    pos15 = _discount_premium(hlc_15m)

    pos_1h = pos1["position"]
    pos_15m = pos15["position"]

    # Aturan sederhana:
    # LONG ideal: 1h bukan DOWN + 1h & 15m bukan PREMIUM bersamaan
    # SHORT ideal: 1h bukan UP + 1h & 15m bukan DISCOUNT bersamaan
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
