# imb/imb_detector.py
# Deteksi setup IMB (Institutional Mitigation Block) + bangun Entry/SL/TP.

from typing import Dict, List, Optional, Tuple
import numpy as np

from binance.ohlc_buffer import Candle
from core.imb_settings import imb_settings
from imb.htf_context import get_htf_context
from imb.imb_tiers import evaluate_signal_quality


# ==============================
# NumPy helper
# ==============================

def _candle_arrays(candles: List[Candle]):
    """
    Convert list Candle -> NumPy arrays:
    opens, highs, lows, closes
    """
    if not candles:
        return None, None, None, None

    opens = np.fromiter((c["open"] for c in candles), dtype=float)
    highs = np.fromiter((c["high"] for c in candles), dtype=float)
    lows = np.fromiter((c["low"] for c in candles), dtype=float)
    closes = np.fromiter((c["close"] for c in candles), dtype=float)

    return opens, highs, lows, closes


def _avg_body(candles: List[Candle], lookback: int = 30) -> float:
    """
    Rata-rata body |close-open|, pakai NumPy.
    """
    if not candles:
        return 0.0

    opens, _, _, closes = _candle_arrays(candles)
    if opens is None or closes is None or opens.size == 0:
        return 0.0

    if lookback and opens.size > lookback:
        opens = opens[-lookback:]
        closes = closes[-lookback:]

    bodies = np.abs(closes - opens)
    return float(bodies.mean()) if bodies.size > 0 else 0.0


def _find_impulse(candles: List[Candle]) -> Optional[int]:
    """
    Cari candle impuls terakhir (body > factor * rata2 body).
    Versi NumPy.
    """
    n = len(candles)
    if n < 20:
        return None

    avg = _avg_body(candles, lookback=30)
    if avg <= 0:
        return None

    factor = 1.8  # threshold awal; nanti kita ketatkan lagi di analyze_symbol_imb

    opens, _, _, closes = _candle_arrays(candles)
    if opens is None or closes is None:
        return None

    bodies = np.abs(closes - opens)

    start = max(0, n - 20)
    tail = bodies[start:]

    mask = tail > (factor * avg)
    if not mask.any():
        return None

    idx_candidates = np.nonzero(mask)[0] + start
    best_local = bodies[idx_candidates].argmax()
    best_idx = int(idx_candidates[best_local])
    return best_idx


def _find_block(candles: List[Candle], impulse_idx: int) -> Optional[Tuple[float, float, str]]:
    """
    Temukan blok IMB sederhana:
    - untuk impuls naik → blok adalah range dari 1–3 candle bearish sebelum impuls
    - untuk impuls turun → blok adalah range dari 1–3 candle bullish sebelum impuls
    Return (block_low, block_high, side) atau None.
    """
    if impulse_idx is None or impulse_idx <= 0:
        return None

    imp = candles[impulse_idx]
    side = "long" if imp["close"] > imp["open"] else "short"

    start = max(0, impulse_idx - 3)
    end = impulse_idx

    highs: List[float] = []
    lows: List[float] = []

    for i in range(start, end):
        c = candles[i]
        if side == "long":
            if c["close"] < c["open"]:
                highs.append(c["high"])
                lows.append(c["low"])
        else:
            if c["close"] > c["open"]:
                highs.append(c["high"])
                lows.append(c["low"])

    if not highs or not lows:
        return None

    block_high = max(highs)
    block_low = min(lows)

    if block_high <= block_low:
        return None

    return block_low, block_high, side


def _build_levels(
    side: str,
    block_low: float,
    block_high: float,
    last_price: float,
    rr1: float = 1.2,
    rr2: float = 2.0,
    rr3: float = 3.0,
) -> Dict[str, float]:
    """
    Bangun Entry / SL / TP berdasarkan blok.

    Aturan utama:
    - LONG  : Entry di block_low, SL DI BAWAH block_low
    - SHORT : Entry di block_high, SL DI ATAS block_high
    """
    if side == "long":
        entry = block_low
        sl = block_low * 0.997  # ~0.3% di bawah blok
        risk = entry - sl
    else:
        entry = block_high
        sl = block_high * 1.003  # ~0.3% di atas blok
        risk = sl - entry

    # Safety fallback kalau ada kasus aneh
    if risk <= 0:
        risk = abs(entry) * 0.003
        if side == "long":
            sl = entry - risk
        else:
            sl = entry + risk

    if side == "long":
        tp1 = entry + rr1 * risk
        tp2 = entry + rr2 * risk
        tp3 = entry + rr3 * risk
    else:
        tp1 = entry - rr1 * risk
        tp2 = entry - rr2 * risk
        tp3 = entry - rr3 * risk

    sl_pct = abs(risk / entry) * 100.0 if entry != 0 else 0.0
    lev_min, lev_max = recommend_leverage_range(sl_pct)

    return {
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "sl_pct": float(sl_pct),
        "lev_min": float(lev_min),
        "lev_max": float(lev_max),
    }


def recommend_leverage_range(sl_pct: float) -> Tuple[float, float]:
    """
    Rekomendasi leverage rentang berdasarkan SL% (risk per posisi jika 1x).
    Sama gaya dengan bot pertama.
    """
    if sl_pct <= 0:
        return 5.0, 10.0

    if sl_pct <= 0.25:
        return 25.0, 40.0
    elif sl_pct <= 0.50:
        return 15.0, 25.0
    elif sl_pct <= 0.80:
        return 8.0, 15.0
    elif sl_pct <= 1.20:
        return 5.0, 8.0
    else:
        return 3.0, 5.0


def analyze_symbol_imb(symbol: str, candles_5m: List[Candle]) -> Optional[Dict]:
    """
    Analisa IMB untuk satu symbol menggunakan data 5m.
    Flow:
    - cari impuls kuat terakhir
    - temukan blok IMB sebelum impuls
    - bangun Entry/SL/TP berdasarkan blok
    - cek RR & SL%
    - cek konteks HTF
    - skor & tier → hanya kirim jika >= min_tier
    """
    if len(candles_5m) < 40:
        return None

    imp_idx = _find_impulse(candles_5m)
    if imp_idx is None:
        return None

    block = _find_block(candles_5m, imp_idx)
    if not block:
        return None

    block_low, block_high, side = block
    last_price = candles_5m[-1]["close"]

    levels = _build_levels(side, block_low, block_high, last_price)

    entry = levels["entry"]
    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    tp3 = levels["tp3"]
    sl_pct = levels["sl_pct"]

    # ---------- FILTER KETAT DASAR (ANTI SPAM) ----------

    # 1) impuls harus benar-benar kuat: body >= 2.2x rata-rata
    avg_body = _avg_body(candles_5m)
    imp_candle = candles_5m[imp_idx]
    imp_body = abs(imp_candle["close"] - imp_candle["open"])
    if avg_body <= 0 or imp_body < 2.2 * avg_body:
        return None

    # 2) SL% harus di range yang masuk akal (0.25% s/d 0.8%)
    if not (0.25 <= sl_pct <= 0.80):
        return None

    # 3) RR minimal: TP2 >= 2.0R
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    rr_tp2 = abs(tp2 - entry) / risk
    if rr_tp2 < 2.0:
        return None

    # 4) HTF wajib searah, tidak boleh netral / melawan
    htf_ctx = get_htf_context(symbol)
    if side == "long":
        htf_alignment = bool(htf_ctx.get("htf_ok_long", True))
    else:
        htf_alignment = bool(htf_ctx.get("htf_ok_short", True))

    if not htf_alignment:
        return None

    # 5) Entry tidak boleh terlalu jauh dari harga sekarang (>0.35%)
    distance_pct = abs((entry - last_price) / last_price) * 100 if last_price else 999
    if distance_pct > 0.35:
        return None

    # ---------- META & SCORING ----------

    # rr_ok sudah pasti True karena kita paksa rr_tp2 >= 2.0
    meta = {
        "has_block": True,
        "impulse_ok": True,
        "touch_ok": True,
        "reaction_ok": True,
        "rr_ok": True,
        "sl_pct": sl_pct,
        "htf_alignment": htf_alignment,
    }

    q = evaluate_signal_quality(meta)
    if not q["should_send"]:
        return None

    tier = q["tier"]
    score = q["score"]

    direction_label = "LONG" if side == "long" else "SHORT"
    emoji = "🟢" if side == "long" else "🔴"

    lev_min = levels["lev_min"]
    lev_max = levels["lev_max"]
    lev_text = f"{lev_min:.0f}x–{lev_max:.0f}x"
    sl_pct_text = f"{sl_pct:.2f}%"

    max_age_candles = imb_settings.max_entry_age_candles
    approx_minutes = max_age_candles * 5
    valid_text = f"±{approx_minutes} menit" if approx_minutes > 0 else "singkat"

    # Risk calculator mini
    if sl_pct > 0:
        pos_mult = 100.0 / sl_pct
        example_balance = 100.0
        example_pos = pos_mult * example_balance
        risk_calc = (
            f"Risk Calc (contoh risiko 1%):\n"
            f"• SL : {sl_pct_text} → nilai posisi ≈ (1% / SL%) × balance ≈ {pos_mult:.1f}× balance\n"
            f"• Contoh balance 100 USDT → posisi ≈ {example_pos:.0f} USDT\n"
            f"(sesuaikan dengan balance & leverage kamu)"
        )
    else:
        risk_calc = "Risk Calc: SL% tidak valid (0), abaikan kalkulasi ini."

    text = (
        f"{emoji} IMB SIGNAL — {symbol.upper()} ({direction_label})\n"
        f"Entry : `{entry:.6f}`\n"
        f"SL    : `{sl:.6f}`\n"
        f"TP1   : `{tp1:.6f}`\n"
        f"TP2   : `{tp2:.6f}`\n"
        f"TP3   : `{tp3:.6f}`\n"
        f"Model : IMB Mitigation Block\n"
        f"Rekomendasi Leverage : {lev_text} (SL {sl_pct_text})\n"
        f"Validitas Entry : {valid_text}\n"
        f"Tier : {tier} (Score {score})\n"
        f"{risk_calc}"
    )

    return {
        "symbol": symbol.upper(),
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "tp3": tp3,
        "sl_pct": sl_pct,
        "lev_min": lev_min,
        "lev_max": lev_max,
        "tier": tier,
        "score": score,
        "htf_context": htf_ctx,
        "message": text,
    }
