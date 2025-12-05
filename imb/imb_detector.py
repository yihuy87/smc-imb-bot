# imb/imb_detector.py
# Deteksi setup IMB (Institutional Mitigation Block) + bangun Entry/SL/TP.
# Versi semi-strict: lebih longgar dari versi sebelumnya tapi tetap pakai filter kualitas.

from typing import Dict, List, Optional, Tuple
import numpy as np

from binance.ohlc_buffer import Candle
from core.imb_settings import imb_settings
from imb.htf_context import get_htf_context
from imb.imb_tiers import evaluate_signal_quality
from imb.liquidity_sweep import detect_liquidity_sweep
from core.leverage_engine import recommend_leverage


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
    lookback_tail: int = 30,
    factor: float = 1.8,
) -> Optional[int]:
    """
    Cari candle impuls:
    - body >= factor * avg_body
    - fokus di lookback_tail candle terakhir.
    Versi lebih longgar dari sebelumnya (factor diturunkan).
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
    Temukan blok IMB + beberapa flag kualitas:
    - blok = 1–3 candle berlawanan sebelum impuls
    - candle blok tidak harus besar sekali, cukup >= 0.5 * avg_body
    - FVG & BOS tidak lagi hard filter, tapi di-return sebagai flag kualitas.

    Return:
      (block_low, block_high, side, has_fvg, bos_ok)
    """
    if imp_idx is None or imp_idx <= 0:
        return None

    imp_open = opens[imp_idx]
    imp_close = closes[imp_idx]
    side = "long" if imp_close > imp_open else "short"

    n = opens.size

    # ---- cari blok: 1–3 candle sebelum impuls, warna berlawanan & body lumayan ----
    start = max(0, imp_idx - 3)
    end = imp_idx

    block_highs: List[float] = []
    block_lows: List[float] = []

    min_block_body = 0.5 * avg_body  # longgar: cukup setengah avg_body

    for i in range(start, end):
        o = opens[i]
        c = closes[i]
        body = abs(c - o)

        if body < min_block_body:
            continue  # candle terlalu kecil → skip

        if side == "long":
            # butuh candle merah sebelum impuls hijau
            if c < o:
                block_highs.append(highs[i])
                block_lows.append(lows[i])
        else:
            # side short → candle hijau sebelum impuls merah
            if c > o:
                block_highs.append(highs[i])
                block_lows.append(lows[i])

    if not block_highs or not block_lows:
        return None

    block_high = max(block_highs)
    block_low = min(block_lows)

    if block_high <= block_low:
        return None

    # ---- flag FVG (tidak wajib, hanya kualitas) ----
    imp_high = highs[imp_idx]
    imp_low = lows[imp_idx]

    if side == "long":
        # cari gap antara block_high dan low impuls
        has_fvg = imp_low > block_high
    else:
        has_fvg = imp_high < block_low

    # ---- flag BOS (juga tidak wajib, tapi kualitas) ----
    pre_start = max(0, start - 10)
    pre_end = start

    bos_ok = False
    if pre_end > pre_start:
        pre_highs = highs[pre_start:pre_end]
        pre_lows = lows[pre_start:pre_end]

        if side == "long":
            prev_struct_high = float(pre_highs.max())
            bos_ok = imp_high > prev_struct_high * 1.0005 or imp_close > prev_struct_high * 1.0005
        else:
            prev_struct_low = float(pre_lows.min())
            bos_ok = imp_low < prev_struct_low * 0.9995 or imp_close < prev_struct_low * 0.9995

    return block_low, block_high, side, has_fvg, bos_ok


# ==============================
# TP Dinamis
# ==============================

def _dynamic_tp_factors(
    candles: List[Candle],
    impulse_idx: int,
    block_low: float,
    block_high: float,
    min_factor: float = 1.2,
    max_factor: float = 3.5,
) -> float:
    """
    Hitung faktor TP dinamis untuk IMB.
    Menggabungkan kekuatan impuls, range IMB, dan volatilitas avg body.
    Semakin kuat impuls & semakin besar block_range dibanding vol,
    semakin besar potensi TP.
    """
    imp = candles[impulse_idx]
    impulse_strength = abs(imp["close"] - imp["open"])

    block_range = abs(block_high - block_low)

    sub = candles[-20:]
    bodies = [abs(c["close"] - c["open"]) for c in sub]
    vol = sum(bodies) / len(bodies) if bodies else impulse_strength

    tp_factor = 1.0
    tp_factor += impulse_strength / max(vol, 1e-9)
    tp_factor += block_range / max(vol, 1e-9)

    # Clamp supaya tidak terlalu ekstrem
    tp_factor = max(min_factor, min(tp_factor, max_factor))
    return tp_factor


# ==============================
# Level builder (Entry / SL / TP / Leverage)
# ==============================

def _build_levels(
    side: str,
    block_low: float,
    block_high: float,
    last_price: float,
    candles_5m: List[Candle],
    imp_idx: int,
) -> Dict[str, float]:
    """
    IMB Dynamic SL + Dynamic TP + Dynamic Leverage:
    - Entry: di area block (anti-FOMO sedikit)
    - SL  : berdasarkan struktur block + buffer dinamis
    - TP  : berdasarkan faktor dinamis (impulse strength + volatility)
    - SL% : dipakai untuk rekomendasi leverage.
    """

    # ---------- ENTRY ----------
    if side == "long":
        entry = min(block_low, last_price)
    else:
        entry = max(block_high, last_price)

    # ---------- BUFFER DINAMIS ----------
    block_range = abs(block_high - block_low)
    base_buffer = max(
        block_range * 0.30,    # 30% dari tinggi block
        abs(entry) * 0.0005    # minimal ~0.05% dari harga
    )

    # ---------- SL DINAMIS ----------
    if side == "long":
        raw_sl = block_low - base_buffer
        # jaga SL selalu di bawah entry
        sl = min(raw_sl, entry * 0.9990)
        risk = entry - sl
    else:
        raw_sl = block_high + base_buffer
        sl = max(raw_sl, entry * 1.0010)
        risk = sl - entry

    # fallback jika ada kasus aneh
    if risk <= 0:
        risk = abs(entry) * 0.003
        if side == "long":
            sl = entry - risk
        else:
            sl = entry + risk

    # ---------- TP DINAMIS ----------
    tp_factor = _dynamic_tp_factors(candles_5m, imp_idx, block_low, block_high)

    if side == "long":
        tp1 = entry + risk * (tp_factor * 0.50)
        tp2 = entry + risk * (tp_factor * 0.90)
        tp3 = entry + risk * tp_factor
    else:
        tp1 = entry - risk * (tp_factor * 0.50)
        tp2 = entry - risk * (tp_factor * 0.90)
        tp3 = entry - risk * tp_factor

    # ---------- SL% & LEVERAGE ----------
    sl_pct = abs(risk / entry) * 100.0 if entry != 0 else 0.0
    lev_min, lev_max = recommend_leverage(sl_pct)

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


# ==============================
# Main analyzer
# ==============================

def analyze_symbol_imb(symbol: str, candles_5m: List[Candle]) -> Optional[Dict]:
    """
    Analisa IMB untuk satu symbol menggunakan data 5m.
    Versi semi-strict: lebih longgar daripada versi sebelumnya, tapi tetap menjaga kualitas.
    """
    if len(candles_5m) < 40:
        return None

    opens, highs, lows, closes = _candle_arrays(candles_5m)
    if opens is None or highs is None or lows is None or closes is None:
        return None

    avg_body = _avg_body(candles_5m)
    if avg_body <= 0:
        return None

    # 1) Cari impuls yang cukup kuat (lebih longgar dari versi strict)
    imp_idx = _find_impulse(opens, closes, avg_body, lookback_tail=30, factor=1.8)
    if imp_idx is None:
        return None

    # 2) Cari blok IMB + flag FVG & BOS (tidak lagi hard filter)
    block_info = _find_block_and_filters(
        candles_5m, opens, highs, lows, closes, imp_idx, avg_body
    )
    if not block_info:
        return None

    block_low, block_high, side, has_fvg, bos_ok = block_info
    last_price = candles_5m[-1]["close"]

    # 3) Deteksi liquidity sweep (tidak wajib, jadi faktor kualitas saja)
    sweep_ok = detect_liquidity_sweep(candles_5m, side, imp_idx)

    # 4) Bangun level (Entry / SL / TP / leverage) dengan model dinamis
    levels = _build_levels(
        side=side,
        block_low=block_low,
        block_high=block_high,
        last_price=last_price,
        candles_5m=candles_5m,
        imp_idx=imp_idx,
    )

    entry = levels["entry"]
    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    tp3 = levels["tp3"]
    sl_pct = levels["sl_pct"]

    # ---------- FILTER LANJUTAN (DILONGGARKAN) ----------

    # SL% harus > 0 dan tidak terlalu besar (longgar sampai 1.5%)
    if sl_pct <= 0 or sl_pct > 1.5:
        return None

    # RR minimal: TP2 >= 1.6R (bukan 2.0R lagi)
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    rr_tp2 = abs(tp2 - entry) / risk
    if rr_tp2 < 1.6:
        return None

    # HTF tetap wajib searah (ini penting, tidak saya longgarkan)
    htf_ctx = get_htf_context(symbol)
    if side == "long":
        htf_alignment = bool(htf_ctx.get("htf_ok_long", True))
    else:
        htf_alignment = bool(htf_ctx.get("htf_ok_short", True))

    if not htf_alignment:
        return None

    # Entry tidak boleh terlalu jauh dari harga sekarang (>0.8% sekarang)
    distance_pct = abs((entry - last_price) / last_price) * 100 if last_price else 999
    if distance_pct > 0.80:
        return None

    # ---------- META & SCORING ----------

    meta = {
        "has_block": True,
        "impulse_ok": True,
        "touch_ok": has_fvg,
        "reaction_ok": bos_ok,
        "liquidity_sweep": sweep_ok,
        "rr_ok": rr_tp2 >= 1.6,
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
        f"Model : IMB Mitigation Block (Semi-Strict)\n"
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
