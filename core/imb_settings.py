# core/imb_settings.py
from dataclasses import dataclass

from config import (
    IMB_ENTRY_TF,
    IMB_USE_HTF_FILTER,
    IMB_MAX_ENTRY_AGE_CANDLES,
    IMB_MIN_RR_TP2,
    MIN_TIER_TO_SEND,
)


@dataclass
class IMBSettings:
    entry_tf: str = IMB_ENTRY_TF        # "5m"
    use_htf_filter: bool = IMB_USE_HTF_FILTER
    max_entry_age_candles: int = IMB_MAX_ENTRY_AGE_CANDLES
    min_rr_tp2: float = IMB_MIN_RR_TP2
    min_tier_to_send: str = MIN_TIER_TO_SEND


imb_settings = IMBSettings()
