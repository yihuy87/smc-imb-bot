# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# === TELEGRAM ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_ADMIN_ID = os.getenv("TELEGRAM_ADMIN_ID", "")
TELEGRAM_ADMIN_USERNAME = os.getenv("TELEGRAM_ADMIN_USERNAME", "")

# === BINANCE FUTURES (USDT PERP) ===
BINANCE_REST_URL = "https://fapi.binance.com"
BINANCE_STREAM_URL = "wss://fstream.binance.com/stream"

# Filtering volume minimum (dalam USDT) untuk pilih pair
MIN_VOLUME_USDT = float(os.getenv("MIN_VOLUME_USDT", "1000000"))

# Berapa banyak pair USDT yang discan
MAX_USDT_PAIRS = int(os.getenv("MAX_USDT_PAIRS", "300"))

# Tier minimum sinyal yang dikirim: "A+", "A", "B"
MIN_TIER_TO_SEND = os.getenv("MIN_TIER_TO_SEND", "A")

# Cooldown default antar sinyal per pair (detik)
SIGNAL_COOLDOWN_SECONDS = int(os.getenv("SIGNAL_COOLDOWN_SECONDS", "600"))

# Refresh interval untuk daftar pair (jam)
REFRESH_PAIR_INTERVAL_HOURS = int(os.getenv("REFRESH_PAIR_INTERVAL_HOURS", "24"))

# ========== IMB SETTINGS ==========
# TF entry IMB
IMB_ENTRY_TF = "5m"

# HTF filter (15m dan 1h) aktif atau tidak
IMB_USE_HTF_FILTER = os.getenv("IMB_USE_HTF_FILTER", "true").lower() == "true"

# Usia maksimal setup IMB (berapa candle 5m sejak impuls)
IMB_MAX_ENTRY_AGE_CANDLES = int(os.getenv("IMB_MAX_ENTRY_AGE_CANDLES", "8"))

# Minimal RR ke TP2 (misal 1.8 berarti TP2 minimal 1:1.8)
IMB_MIN_RR_TP2 = float(os.getenv("IMB_MIN_RR_TP2", "1.8"))

# Strict mode IMB
IMB_STRICT_MODE = os.getenv("IMB_STRICT_MODE", "false").lower() == "true"
