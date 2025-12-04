# imb/imb_tiers.py
# Skoring kualitas sinyal IMB & tier.

from typing import Dict

from core.bot_state import state


def score_imb(meta: Dict) -> int:
    score = 0

    sl_pct = float(meta.get("sl_pct", 0.0))
    rr_tp2 = float(meta.get("rr_tp2", 0.0))
    impulse_strength = float(meta.get("impulse_strength", 0.0))
    block_range_pct = float(meta.get("block_range_pct", 0.0))
    htf_alignment = bool(meta.get("htf_alignment", True))

    # impuls kuat
    if impulse_strength >= 1.5:
        score += 20
    if impulse_strength >= 2.0:
        score += 10

    # block kecil (rapi)
    if 0.0005 <= block_range_pct <= 0.006:
        score += 20
    elif block_range_pct < 0.0005:
        score += 10  # terlalu tipis tapi masih ok
    elif block_range_pct <= 0.010:
        score += 5

    # RR ke TP2
    if rr_tp2 >= 1.8:
        score += 25
    if rr_tp2 >= 2.2:
        score += 10

    # SL sehat
    if 0.10 <= sl_pct <= 0.80:
        score += 20
    elif 0.80 < sl_pct <= 1.50:
        score += 10

    # HTF align
    if htf_alignment:
        score += 15

    return int(min(score, 150))


def tier_from_score(score: int) -> str:
    if score >= 120:
        return "A+"
    elif score >= 100:
        return "A"
    elif score >= 80:
        return "B"
    else:
        return "NONE"


def should_send_tier(tier: str) -> bool:
    order = {"NONE": 0, "B": 1, "A": 2, "A+": 3}
    min_tier = state.min_tier or "A"
    return order.get(tier, 0) >= order.get(min_tier, 2)


def evaluate_imb_quality(meta: Dict) -> Dict:
    score = score_imb(meta)
    tier = tier_from_score(score)
    send = should_send_tier(tier)
    return {
        "score": score,
        "tier": tier,
        "should_send": send,
    }
