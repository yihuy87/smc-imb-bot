# imb/imb_detector.py
# Deteksi setup IMB (Institutional Mitigation Block) + bangun Entry/SL/TP.
# Versi STRICT / profesional (opsi A): jauh lebih sedikit sinyal.

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


# ==============================
# IMB building blocks
# ==============================

def _find_impulse(
    opens: np.ndarray,
    closes: np.ndarray,
    avg_body: float,
    lookback_tail: int = 20,
    factor: float = 2.5,
) -> Optional[int]:
    """
    Cari candle impuls terakhir:
    - body >= factor * avg_body (STRICT)
    - fokus di lookback_tail candle terakhir.
    Return index candle impuls atau None.
    """
    n = opens.size
    if n < 20 or avg_body <= 0:
        return None

    bodies = np.abs(closes - opens)

    start = max(0, n - lookback_tail)
    tail = bodies[start:]

    mask = tail >= (factor * avg_body)
    if not mask.any():
        return None

    idx_candidates = np.nonzero(mask)[0] + start
    best_local = bodies[idx_candidates].argmax()
    best_idx = int(idx_candidates[best_local])
    return best_idx


def _find_block_and_filters(
    candles: List[Candle],
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    imp_idx: int,
    avg_body: float,
) -> Optional[Tuple[float, float, str, bool, bool]]:
    """
    Temukan blok IMB + filter ketat:
    - blok = 1–3 candle berlawanan sebelum impuls
    - candle blok harus punya body >= 0.75 * avg_body (bukan candle kecil)
    - wajib ada FVG kuat antara blok dan impuls
    - impuls harus break structure (BOS) signifikan.

    Return:
      (block_low, block_high, side, has_fvg, bos_ok)
    atau None kalau tidak valid.
    """
    if imp_idx is None or imp_idx <= 0:
        return None

    imp_open = opens[imp_idx]
    imp_close = closes[imp_idx]
    side = "long" if imp_close > imp_open else "short"

    n = opens.size

    # ---- cari blok: 1–3 candle sebelum impuls, warna berlawanan & body besar ----
    start = max(0, imp_idx - 3)
    end = imp_idx

    block_highs: List[float] = []
    block_lows: List[float] = []

    min_block_body = 0.75 * avg_body  # STRICT: minimal cukup besar

    for i in range(start, end):
        o = opens[i]
        c = closes[i]
        body = abs(c - o)

        if body < min_block_body:
            continue  # candle kecil → bukan blok IMB

        if side == "long":
            # butuh candle merah besar sebelum impuls hijau
            if c < o:
                block_highs.append(highs[i])
                block_lows.append(lows[i])
        else:
            # side short → candle hijau besar sebelum impuls merah
            if c > o:
                block_highs.append(highs[i])
                block_lows.append(lows[i])

    if not block_highs or not block_lows:
        return None

    block_high = max(block_highs)
    block_low = min(block_lows)

    if block_high <= block_low:
        return None

    # ---- check FVG kuat antara blok dan impuls ----
    # Definisi sederhana & ketat:
    # LONG : seluruh candle impuls berada di atas block_high → ada gap dari block ke impuls.
    # SHORT: seluruh candle impuls berada di bawah block_low.
    imp_high = highs[imp_idx]
    imp_low = lows[imp_idx]

    if side == "long":
        has_fvg = imp_low > block_high
    else:
        has_fvg = imp_high < block_low

    if not has_fvg:
        # tanpa FVG jelas → kita anggap bukan IMB setup
        return None

    # ---- BOS (Break of Structure) ----
    # LONG: imp_high harus > high tertinggi 10 candle sebelum blok.
    # SHORT: imp_low harus < low terendah 10 candle sebelum blok.
    pre_start = max(0, start - 10)
    pre_end = start  # sampai sebelum blok

    if pre_end <= pre_start:
        return None

    pre_highs = highs[pre_start:pre_end]
    pre_lows = lows[pre_start:pre_end]

    if side == "long":
        prev_struct_high = float(pre_highs.max())
        bos_ok = imp_high > prev_struct_high * 1.001  # at least ~0.1% break
    else:
        prev_struct_low = float(pre_lows.min())
        bos_ok = imp_low < prev_struct_low * 0.999  # at least ~0.1% break

    if not bos_ok:
        return None

    return block_low, block_high, side, True, True


def _build_levels(
    side: str,
    block_low: float,
    block_high: float,
    last_price: float,
    rr1: float = 1.2,
    rr2: float = 2.2,
    rr3: float = 3.0,
) -> Dict[str, float]:
    """
    Bangun Entry / SL / TP berdasarkan blok (STRICT):

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


# ==============================
# Main analyzer
# ==============================

def analyze_symbol_imb(symbol: str, candles_5m: List[Candle]) -> Optional[Dict]:
    """
    Analisa IMB untuk satu symbol menggunakan data 5m (STRICT mode).
    """
    if len(candles_5m) < 40:
        return None

    opens, highs, lows, closes = _candle_arrays(candles_5m)
    if opens is None or highs is None or lows is None or closes is None:
        return None

    avg_body = _avg_body(candles_5m)
    if avg_body <= 0:
        return None

    imp_idx = _find_impulse(opens, closes, avg_body, lookback_tail=25, factor=2.5)
    if imp_idx is None:
        return None

    block_info = _find_block_and_filters(
        candles_5m, opens, highs, lows, closes, imp_idx, avg_body
    )
    if not block_info:
        return None

    block_low, block_high, side, has_fvg, bos_ok = block_info
    last_price = candles_5m[-1]["close"]

    levels = _build_levels(side, block_low, block_high, last_price)

    entry = levels["entry"]
    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    tp3 = levels["tp3"]
    sl_pct = levels["sl_pct"]

    # ---------- FILTER KETAT LANJUTAN (ANTI SPAM) ----------

    # 1) SL% harus di range yang ketat (0.25% s/d 0.80%)
    if not (0.25 <= sl_pct <= 0.80):
        return None

    # 2) RR minimal: TP2 >= 2.2R
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    rr_tp2 = abs(tp2 - entry) / risk
    if rr_tp2 < 2.2:
        return None

    # 3) HTF wajib searah
    htf_ctx = get_htf_context(symbol)
    if side == "long":
        htf_alignment = bool(htf_ctx.get("htf_ok_long", True))
    else:
        htf_alignment = bool(htf_ctx.get("htf_ok_short", True))

    if not htf_alignment:
        return None

    # 4) Entry tidak boleh terlalu jauh dari harga sekarang (>0.35%)
    distance_pct = abs((entry - last_price) / last_price) * 100 if last_price else 999
    if distance_pct > 0.35:
        return None

    # ---------- META & SCORING ----------

    meta = {
        "has_block": True,
        "impulse_ok": True,
        "touch_ok": has_fvg,       # kita pakai FVG sebagai validasi touch
        "reaction_ok": bos_ok,     # BOS sebagai proxy reaction kuat
        "rr_ok": True,             # sudah dipaksa RR >= 2.2
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
        f"Model : IMB Mitigation Block (STRICT)\n"
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
