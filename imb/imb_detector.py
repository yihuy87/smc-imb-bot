# imb/imb_detector.py
# Deteksi setup IMB (Institutional Mitigation Block) + bangun Entry/SL/TP (STRICT).

from typing import Dict, List, Optional, Tuple

from binance.ohlc_buffer import Candle
from core.imb_settings import imb_settings
from imb.htf_context import get_htf_context
from imb.imb_tiers import evaluate_signal_quality


def _avg_body(candles: List[Candle], lookback: int = 40) -> float:
    sub = candles[-lookback:] if len(candles) > lookback else candles
    if not sub:
        return 0.0
    total = 0.0
    for c in sub:
        total += abs(c["close"] - c["open"])
    return total / len(sub)


def _find_impulse(candles: List[Candle]) -> Optional[int]:
    """
    Cari candle impuls terakhir (STRICT):
    - body > 2.2 × rata2 body
    - dan break range 10–15 candle terakhir (bukan noise kecil).
    """
    n = len(candles)
    if n < 30:
        return None

    avg = _avg_body(candles, lookback=40)
    if avg <= 0:
        return None

    factor = 2.2
    start = max(0, n - 25)
    best_idx = None
    best_body = 0.0

    for i in range(start, n):
        c = candles[i]
        body = abs(c["close"] - c["open"])
        if body <= factor * avg:
            continue

        # cek break struktur lokal (10 candle sebelumnya)
        pre_start = max(0, i - 10)
        prev_closes = [cc["close"] for cc in candles[pre_start:i]]
        if not prev_closes:
            continue

        if c["close"] > c["open"]:
            # impuls naik → close harus di atas high range sebelumnya
            prev_high = max(prev_closes)
            if c["close"] <= prev_high * 1.001:
                continue
        else:
            # impuls turun → close harus di bawah low range sebelumnya
            prev_low = min(prev_closes)
            if c["close"] >= prev_low * 0.999:
                continue

        if body > best_body:
            best_body = body
            best_idx = i

    return best_idx


def _find_block(
    candles: List[Candle],
    impulse_idx: int,
) -> Optional[Tuple[float, float, str]]:
    """
    Temukan blok IMB sederhana:
    - impuls naik → blok dari 1–3 candle bearish sebelum impuls
    - impuls turun → blok dari 1–3 candle bullish sebelum impuls
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


def _price_near_block(
    side: str,
    block_low: float,
    block_high: float,
    last_price: float,
    max_dist_pct: float = 0.5,
) -> bool:
    """
    Pastikan harga sekarang tidak terlalu jauh dari area blok.
    max_dist_pct dalam % (default 0.5%).
    """
    if last_price <= 0 or block_low <= 0 or block_high <= 0:
        return False

    # entry mentah = ekstrem blok
    raw_entry = block_low if side == "long" else block_high
    dist_pct = abs(last_price - raw_entry) / last_price * 100.0
    return dist_pct <= max_dist_pct


def _check_touch(
    candles: List[Candle],
    side: str,
    block_low: float,
    block_high: float,
    lookback: int = 4,
) -> bool:
    """
    Cek apakah harga sudah 'menyentuh' blok dalam beberapa candle terakhir.
    """
    recent = candles[-lookback:] if len(candles) > lookback else candles
    if not recent:
        return False

    for c in recent:
        if side == "long":
            # low menyentuh area blok (sedikit toleransi di bawah)
            if block_low * 0.997 <= c["low"] <= block_high * 1.001:
                return True
        else:
            # high menyentuh area blok (sedikit toleransi di atas)
            if block_low * 0.999 <= c["high"] <= block_high * 1.003:
                return True

    return False


def _check_reaction(
    candles: List[Candle],
    side: str,
    block_low: float,
    block_high: float,
) -> bool:
    """
    Cek reaksi awal setelah touch blok:
    - LONG: candle terakhir bullish + low di dekat blok
    - SHORT: candle terakhir bearish + high di dekat blok
    """
    if not candles:
        return False
    c = candles[-1]

    if side == "long":
        if c["close"] <= c["open"]:
            return False
        if not (block_low * 0.997 <= c["low"] <= block_high * 1.005):
            return False
        return True
    else:
        if c["close"] >= c["open"]:
            return False
        if not (block_low * 0.995 <= c["high"] <= block_high * 1.003):
            return False
        return True


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
    Bangun Entry/SL/TP berdasarkan blok IMB + sedikit buffer.
    """
    if side == "long":
        raw_entry = block_low
        entry = min(raw_entry, last_price)
        sl = block_low * 0.997
        risk = entry - sl
    else:
        raw_entry = block_high
        entry = max(raw_entry, last_price)
        sl = block_high * 1.003
        risk = sl - entry

    if risk <= 0:
        risk = abs(entry) * 0.003

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


def analyze_symbol_imb(symbol: str, candles_5m: List[Candle]) -> Optional[Dict]:
    """
    Analisa IMB STRICT:
    - cari impuls kuat terakhir
    - cek usia impuls (max_entry_age_candles)
    - temukan blok IMB sebelum impuls
    - pastikan harga masih dekat blok
    - pastikan sudah ada touch + reaction
    - bangun Entry/SL/TP
    - cek RR & SL%
    - cek konteks HTF (opsional)
    - skor & tier → hanya kirim jika >= min_tier
    """
    n = len(candles_5m)
    if n < 40:
        return None

    # 1) Impuls kuat
    imp_idx = _find_impulse(candles_5m)
    if imp_idx is None:
        return None

    # 2) Usia setup IMB (berapa candle sejak impuls)
    last_idx = n - 1
    age = last_idx - imp_idx
    if age < 0 or age > imb_settings.max_entry_age_candles:
        return None

    # 3) Blok IMB
    block = _find_block(candles_5m, imp_idx)
    if not block:
        return None

    block_low, block_high, side = block
    last_price = candles_5m[-1]["close"]

    # 4) Harga harus masih dekat blok (bukan jauh lari)
    if not _price_near_block(side, block_low, block_high, last_price):
        return None

    # 5) Touch & Reaction di area blok
    touch_ok = _check_touch(candles_5m, side, block_low, block_high, lookback=4)
    reaction_ok = _check_reaction(candles_5m, side, block_low, block_high)

    if not (touch_ok and reaction_ok):
        return None

    # 6) Bangun level
    levels = _build_levels(side, block_low, block_high, last_price)
    entry = levels["entry"]
    sl = levels["sl"]
    tp1 = levels["tp1"]
    tp2 = levels["tp2"]
    tp3 = levels["tp3"]
    sl_pct = levels["sl_pct"]

    # 7) Validasi RR (TP2 >= min RR)
    risk = abs(entry - sl)
    if risk <= 0:
        return None
    rr_tp2 = abs(tp2 - entry) / risk
    rr_ok = rr_tp2 >= imb_settings.min_rr_tp2

    # 8) Konteks HTF (opsional)
    if imb_settings.use_htf_filter:
        htf_ctx = get_htf_context(symbol)
        if side == "long":
            htf_alignment = bool(htf_ctx.get("htf_ok_long", True))
        else:
            htf_alignment = bool(htf_ctx.get("htf_ok_short", True))
    else:
        htf_ctx = {}
        htf_alignment = True

    # 9) Meta untuk skoring
    meta = {
        "has_block": True,
        "impulse_ok": True,
        "touch_ok": touch_ok,
        "reaction_ok": reaction_ok,
        "rr_ok": rr_ok,
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
