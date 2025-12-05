# imb/liquidity_sweep.py
# Deteksi liquidity sweep sederhana untuk IMB STRICT mode.
# Sweep wajib ada → anti manipulasi market maker.

from typing import List, Optional, Dict
from binance.ohlc_buffer import Candle


def detect_liquidity_sweep(
    candles: List[Candle],
    side: str,
    impulse_index: int,
    max_lookback: int = 3,
    wick_factor: float = 1.3,
    range_factor: float = 1.2,
) -> bool:
    """
    Deteksi sweep sederhana:
    - LONG  → sweep low sebelum impuls (stop-hunt)
    - SHORT → sweep high sebelum impuls

    Kriteria sweep:
    - terjadi dalam max_lookback candle sebelum impuls
    - wick dominan (panjang)
    - total range lebih besar dari rata-rata 5 candle sebelumnya
    """

    n = len(candles)
    if impulse_index <= 2 or n < 10:
        return False

    start = max(0, impulse_index - max_lookback)
    end = impulse_index  # sebelum impuls

    # Hitung rata-rata range dari beberapa candle sebelumnya
    prev_start = max(0, start - 5)
    prev = candles[prev_start:start]

    if not prev:
        return False

    avg_range = sum((c["high"] - c["low"]) for c in prev) / len(prev)

    # Sweep detection
    for i in range(end - 1, start - 1, -1):
        c = candles[i]
        high = c["high"]
        low = c["low"]
        open_ = c["open"]
        close = c["close"]

        total_range = high - low
        if total_range <= 0:
            continue

        # wick values
        upper_wick = high - max(open_, close)
        lower_wick = min(open_, close) - low

        # LONG → sweep low
        if side == "long":
            # Low harus "menusuk" lalu close kembali naik
            sweep_condition = (close > low) and (lower_wick > 0)
            wick_ok = lower_wick >= wick_factor * (upper_wick + 1e-9)
        else:
            # SHORT → sweep high
            sweep_condition = (close < high) and (upper_wick > 0)
            wick_ok = upper_wick >= wick_factor * (lower_wick + 1e-9)

        # quality check: wick besar & range besar
        range_ok = total_range >= range_factor * avg_range

        if sweep_condition and wick_ok and range_ok:
            return True

    return False
