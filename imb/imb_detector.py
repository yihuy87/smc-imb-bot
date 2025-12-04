# imb/imb_detector.py
# Deteksi IMB (Institutional Mitigation Block) + scoring + build signal text.

from typing import List, Dict, Optional, Literal

from binance.ohlc_buffer import Candle
from imb.imb_settings import imb_settings
from imb.htf_context import get_htf_context
from imb.imb_tiers import evaluate_imb_quality


def _avg_body(candles: List[Candle], count: int) -> float:
    n = min(len(candles), count)
    if n <= 0:
        return 0.0
    total = 0.0
    for c in candles[-n:]:
        total += abs(c["close"] - c["open"])
    return total / n


def _detect_impulse(candles: List[Candle]) -> Optional[Dict]:
    """
    Cari impuls terakhir (bullish atau bearish) di beberapa candle terakhir.
    Impuls = body besar & close dekat ekstrem.
    """
    if len(candles) < 15:
        return None

    # cek beberapa candle terakhir
    lookback = min(20, len(candles))
    segment = candles[-lookback:]
    avg_body10 = _avg_body(segment[:-1], 10)

    best_idx = None
    best_strength = 0.0
    direction: Optional[Literal["long", "short"]] = None

    for i in range(len(segment)):
        c = segment[i]
        body = abs(c["close"] - c["open"])
        if body <= 0 or avg_body10 <= 0:
            continue

        strength = body / avg_body10
        if strength < 1.5:
            continue

        high = c["high"]
        low = c["low"]
        close = c["close"]
        op = c["open"]

        # close dekat high → impuls bullish
        if close > op and high > low:
            pos = (close - low) / (high - low)
            if pos >= 0.7 and strength > best_strength:
                best_strength = strength
                direction = "long"
                best_idx = len(candles) - lookback + i

        # close dekat low → impuls bearish
        if close < op and high > low:
            pos = (close - low) / (high - low)
            if pos <= 0.3 and strength > best_strength:
                best_strength = strength
                direction = "short"
                best_idx = len(candles) - lookback + i

    if best_idx is None or direction is None:
        return None

    return {"index": best_idx, "side": direction, "strength": best_strength}


def _find_imb_block(candles: List[Candle], impulse_index: int, side: str) -> Optional[Dict]:
    """
    Cari candle block sebelum impuls:
    - long: cari candle bearish terakhir sebelum impuls
    - short: cari candle bullish terakhir sebelum impuls
    """
    if impulse_index <= 0:
        return None

    block_idx = None
    for i in range(impulse_index - 1, max(impulse_index - 8, -1), -1):
        c = candles[i]
        if side == "long" and c["close"] < c["open"]:  # bearish
            block_idx = i
            break
        if side == "short" and c["close"] > c["open"]:  # bullish
            block_idx = i
            break

    if block_idx is None:
        return None

    block = candles[block_idx]
    block_high = block["high"]
    block_low = block["low"]
    if block_high <= block_low:
        return None

    # range block tidak boleh terlalu besar
    mid_price = candles[impulse_index]["close"]
    range_pct = (block_high - block_low) / mid_price if mid_price != 0 else 0
    if range_pct > 0.008:  # max ~0.8% block
        return None

    return {
        "index": block_idx,
        "high": block_high,
        "low": block_low,
        "range_pct": range_pct,
    }


def _build_levels(side: str, block: Dict, candles: List[Candle], impulse_idx: int) -> Dict:
    """
    Entry di mid block.
    SL di luar block dengan buffer kecil.
    TP pakai RR ke risk (RR 1.2, 2, 3).
    """
    block_high = block["high"]
    block_low = block["low"]
    mid = (block_high + block_low) / 2.0

    # gunakan close impuls sebagai referensi untuk buffer
    ref_price = candles[impulse_idx]["close"]
    if ref_price <= 0:
        ref_price = mid

    buffer = ref_price * 0.0005  # 0.05% buffer

    if side == "long":
        entry = mid
        sl = block_low - buffer
        risk = entry - sl
        tp1 = entry + 1.2 * risk
        tp2 = entry + 2.0 * risk
        tp3 = entry + 3.0 * risk
    else:
        entry = mid
        sl = block_high + buffer
        risk = sl - entry
        tp1 = entry - 1.2 * risk
        tp2 = entry - 2.0 * risk
        tp3 = entry - 3.0 * risk

    if risk <= 0:
        risk = abs(entry) * 0.003

    sl_pct = abs(risk / entry) * 100.0 if entry != 0 else 0.0
    return {
        "entry": float(entry),
        "sl": float(sl),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": float(tp3),
        "risk": float(risk),
        "sl_pct": float(sl_pct),
    }


def analyze_symbol_imb(symbol: str, candles_5m: List[Candle]) -> Optional[Dict]:
    """
    IMB murni:
    - deteksi impuls 5m
    - cari mitigation block sebelum impuls
    - bangun Entry/SL/TP
    - filter HTF (optional)
    - scoring & tier
    """
    if len(candles_5m) < 30:
        return None

    # 1) deteksi impuls
    impulse = _detect_impulse(candles_5m)
    if not impulse:
        return None

    side = impulse["side"]
    impulse_idx = impulse["index"]

    # validitas umur setup: impuls harus termasuk dalam max_entry_age
    last_idx = len(candles_5m) - 1
    age_candles = last_idx - impulse_idx
    if age_candles > imb_settings.max_entry_age_candles:
        return None

    # 2) cari IMB block
    block = _find_imb_block(candles_5m, impulse_idx, side)
    if not block:
        return None

    # 3) build level
    levels = _build_levels(side, block, candles_5m, impulse_idx)

    entry = levels["entry"]
    sl = levels["sl"]
    tp2 = levels["tp2"]
    risk = levels["risk"]

    if risk <= 0 or entry == 0:
        return None

    rr_tp2 = abs(tp2 - entry) / risk
    if rr_tp2 < imb_settings.min_rr_tp2:
        return None

    # 4) HTF context
    htf_ctx = get_htf_context(symbol) if imb_settings.use_htf_filter else {
        "htf_ok_long": True,
        "htf_ok_short": True,
        "trend_1h": "RANGE",
        "pos_1h": "MID",
        "pos_15m": "MID",
    }

    if side == "long" and not htf_ctx.get("htf_ok_long", True):
        return None
    if side == "short" and not htf_ctx.get("htf_ok_short", True):
        return None

    # 5) scoring & tier
    meta = {
        "side": side,
        "sl_pct": levels["sl_pct"],
        "rr_tp2": rr_tp2,
        "impulse_strength": float(impulse["strength"]),
        "block_range_pct": float(block["range_pct"]),
        "htf_alignment": bool(
            htf_ctx.get("htf_ok_long") if side == "long" else htf_ctx.get("htf_ok_short")
        ),
    }
    q = evaluate_imb_quality(meta)
    if not q["should_send"]:
        return None

    tier = q["tier"]
    score = q["score"]

    # 6) build message (format mirip bot 1)
    direction_label = "LONG" if side == "long" else "SHORT"
    emoji = "🟢" if side == "long" else "🔴"

    sl_pct_text = f"{levels['sl_pct']:.2f}%"
    # rekomendasi leverage mirip mapping SL%
    lev_min, lev_max = _recommend_leverage(levels["sl_pct"])
    lev_text = f"{lev_min:.0f}x–{lev_max:.0f}x"

    approx_minutes = imb_settings.max_entry_age_candles * 5
    valid_text = f"±{approx_minutes} menit" if approx_minutes > 0 else "singkat"

    # Risk calc mini 1% balance
    sl_pct = levels["sl_pct"]
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
        f"Entry : `{levels['entry']:.6f}`\n"
        f"SL    : `{levels['sl']:.6f}`\n"
        f"TP1   : `{levels['tp1']:.6f}`\n"
        f"TP2   : `{levels['tp2']:.6f}`\n"
        f"TP3   : `{levels['tp3']:.6f}`\n"
        f"Model : Institutional Mitigation Block (IMB)\n"
        f"Rekomendasi Leverage : {lev_text} (SL {sl_pct_text})\n"
        f"Validitas Entry : {valid_text}\n"
        f"Tier : {tier} (Score {score})\n"
        f"{risk_calc}"
    )

    return {
        "symbol": symbol.upper(),
        "side": side,
        "entry": levels["entry"],
        "sl": levels["sl"],
        "tp1": levels["tp1"],
        "tp2": levels["tp2"],
        "tp3": levels["tp3"],
        "sl_pct": levels["sl_pct"],
        "tier": tier,
        "score": score,
        "htf_context": htf_ctx,
        "message": text,
    }


def _recommend_leverage(sl_pct: float) -> tuple[float, float]:
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
