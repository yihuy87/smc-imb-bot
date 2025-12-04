# imb/imb_detector.py
# Deteksi setup IMB (Institutional Mitigation Block) + bangun Entry/SL/TP.

from typing import Dict, List, Optional, Tuple

from binance.ohlc_buffer import Candle
from core.imb_settings import imb_settings
from imb.htf_context import get_htf_context
from imb.imb_tiers import evaluate_signal_quality


def _avg_body(candles: List[Candle], lookback: int = 30) -> float:
    sub = candles[-lookback:] if len(candles) > lookback else candles
    if not sub:
        return 0.0
    total = 0.0
    for c in sub:
        total += abs(c["close"] - c["open"])
    return total / len(sub)


def _find_impulse(candles: List[Candle]) -> Optional[int]:
    """
    Cari candle impuls terakhir (body > factor * rata2 body).
    Return index candle impuls atau None.
    """
    if len(candles) < 20:
        return None

    avg = _avg_body(candles, lookback=30)
    if avg <= 0:
        return None

    factor = 1.8
    # fokus di ~20 candle terakhir
    start = max(0, len(candles) - 20)
    best_idx = None
    best_body = 0.0

    for i in range(start, len(candles)):
        c = candles[i]
        body = abs(c["close"] - c["open"])
        if body > factor * avg and body > best_body:
            best_body = body
            best_idx = i

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
            # cari candle merah sebelum impuls hijau
            if c["close"] < c["open"]:
                highs.append(c["high"])
                lows.append(c["low"])
        else:
            # side short → cari candle hijau
            if c["close"] > c["open"]:
                highs.append(c["high"])
                lows.append(c["low"])

    if not highs or not lows:
        return None

    block_high = max(highs)
    block_low = min(lows)

    # validasi range tidak aneh
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
    Bangun Entry/SL/TP dengan gaya mirip bot 1:
    - Entry tetap berbasis blok IMB (low/high blok)
    - SL = ekstrem blok ± buffer dinamis (bukan fixed %)
    - TP = kelipatan R (RR1 / RR2 / RR3)
    """
    if side == "long":
        # Entry di dekat low blok, tapi jangan di atas harga terakhir (anti FOMO)
        raw_entry = block_low
        entry = min(raw_entry, last_price)

        sl_base = block_low
        buffer = max(sl_base * 0.0015, abs(entry) * 0.0005)
        sl = sl_base - buffer

        risk = entry - sl
    else:
        # SHORT
        raw_entry = block_high
        entry = max(raw_entry, last_price)

        sl_base = block_high
        buffer = max(sl_base * 0.0015, abs(entry) * 0.0005)
        sl = sl_base + buffer

        risk = sl - entry

    if risk <= 0:
        # fallback kecil
        risk = abs(entry) * 0.003

    # TP berdasarkan RR
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
    1) Cari impuls kuat terakhir (5m)
    2) Temukan blok IMB sebelum impuls
    3) Bangun Entry/SL/TP berbasis blok
    4) Cek RR & SL% → kalau tidak lolos, STOP (tidak panggil HTF)
    5) Baru panggil konteks HTF (15m & 1h) jika diaktifkan
    6) Skor & Tier → hanya kirim jika >= min_tier (state.min_tier)
    """
    # 1) Pastikan cukup candle
    if len(candles_5m) < 40:
        return None

    # 2) Impulse
    imp_idx = _find_impulse(candles_5m)
    if imp_idx is None:
        return None

    # 3) Blok IMB
    block = _find_block(candles_5m, imp_idx)
    if not block:
        return None

    block_low, block_high, side = block
    last_price = candles_5m[-1]["close"]

    # 4) Bangun level + cek RR / SL dulu (tanpa HTF)
    levels = _build_levels(side, block_low, block_high, last_price)

    entry = levels["entry"]
    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    tp3 = levels["tp3"]
    sl_pct = levels["sl_pct"]

    # validasi RR (TP2 minimal RR tertentu dari config)
    risk = abs(entry - sl)
    if risk <= 0:
        return None

    rr_tp2 = abs(tp2 - entry) / risk
    min_rr = imb_settings.min_rr_tp2
    rr_ok = rr_tp2 >= min_rr

    if not rr_ok:
        # RR jelek → tidak usah lanjut HTF
        return None

    # 5) Konteks HTF (15m & 1h) — hanya kalau diaktifkan
    if imb_settings.use_htf_filter:
        htf_ctx = get_htf_context(symbol)
        if side == "long":
            htf_alignment = bool(htf_ctx.get("htf_ok_long", True))
        else:
            htf_alignment = bool(htf_ctx.get("htf_ok_short", True))
    else:
        htf_ctx = {
            "trend_1h": "RANGE",
            "pos_1h": "MID",
            "pos_15m": "MID",
            "htf_ok_long": True,
            "htf_ok_short": True,
        }
        htf_alignment = True

    # 6) Meta untuk skoring
    meta = {
        "has_block": True,
        "impulse_ok": True,
        "touch_ok": True,        # versi awal: dianggap valid saat blok terbentuk
        "reaction_ok": True,     # bisa di-refine nanti
        "rr_ok": rr_ok,
        "sl_pct": sl_pct,
        "htf_alignment": htf_alignment,
    }

    q = evaluate_signal_quality(meta)
    if not q["should_send"]:
        return None

    tier = q["tier"]
    score = q["score"]

    # 7) Build pesan Telegram
    direction_label = "LONG" if side == "long" else "SHORT"
    emoji = "🟢" if side == "long" else "🔴"

    lev_min = levels["lev_min"]
    lev_max = levels["lev_max"]
    lev_text = f"{lev_min:.0f}x–{lev_max:.0f}x"
    sl_pct_text = f"{sl_pct:.2f}%"

    # validitas sinyal (pakai IMB_MAX_ENTRY_AGE_CANDLES dari config/env)
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
