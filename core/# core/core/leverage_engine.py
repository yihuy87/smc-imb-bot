# core/leverage_engine.py
# Modul standar untuk rekomendasi leverage berdasarkan SL%.

from typing import Tuple


def recommend_leverage(sl_pct: float) -> Tuple[float, float]:
    """
    Rekomendasi leverage dinamis sesuai gaya SMC.
    Input:
        sl_pct : float  (contoh: 0.38)
    Output:
        (lev_min, lev_max)

    Range resmi SMC:
        SL ≤ 0.40% → 15x–25x
        SL ≤ 0.70% → 8x–15x
        SL ≤ 1.20% → 5x–8x
        > 1.20%    → 3x–5x
    """
    if sl_pct <= 0:
        return 5.0, 10.0

    if sl_pct <= 0.40:
        return 15.0, 25.0

    elif sl_pct <= 0.70:
        return 8.0, 15.0

    elif sl_pct <= 1.20:
        return 5.0, 8.0

    return 3.0, 5.0
