# imb/imb_tiers.py
# Evaluasi kualitas sinyal IMB dan tentukan Tier (A+, A, B, NONE).

from typing import Dict

from core.bot_state import state


def score_signal(meta: Dict) -> int:
    """
    Skoring berdasarkan kualitas IMB:
    - ada blok IMB yang jelas
    - impuls kuat menjauh dari blok
    - retest / touch ke blok
    - reaksi awal yang rapi (reject)
    - RR sehat
    - SL% sehat
    - align dengan konteks HTF
    """
    score = 0

    has_block = bool(meta.get("has_block"))
    impulse_ok = bool(meta.get("impulse_ok"))
    touch_ok = bool(meta.get("touch_ok"))
    reaction_ok = bool(meta.get("reaction_ok"))
    rr_ok = bool(meta.get("rr_ok"))
    htf_alignment = bool(meta.get("htf_alignment"))

    sl_pct = float(meta.get("sl_pct", 0.0))

    if has_block:
        score += 25
    if impulse_ok:
        score += 25
    if touch_ok:
        score += 15
    if reaction_ok:
        score += 15
    if rr_ok:
        score += 10

    # SL% sehat (kecil tapi tidak ekstrem)
    if 0.20 <= sl_pct <= 0.90:
        score += 10

    if htf_alignment:
        score += 20

    return int(min(score, 150))


def tier_from_score(score: int) -> str:
    """
    Tier:
    - A+ : >= 120
    - A  : 100–119
    - B  : 80–99
    - NONE : < 80
    """
    if score >= 120:
        return "A+"
    elif score >= 100:
        return "A"
    elif score >= 80:
        return "B"
    else:
        return "NONE"


def should_send_tier(tier: str) -> bool:
    """
    Urutan: NONE < B < A < A+
    Bandingkan terhadap state.min_tier (diatur via Telegram /mode).
    """
    order = {"NONE": 0, "B": 1, "A": 2, "A+": 3}
    min_tier = state.min_tier or "A"
    return order.get(tier, 0) >= order.get(min_tier, 2)


def evaluate_signal_quality(meta: Dict) -> Dict:
    score = score_signal(meta)
    tier = tier_from_score(score)
    send = should_send_tier(tier)
    return {
        "score": score,
        "tier": tier,
        "should_send": send,
    }
