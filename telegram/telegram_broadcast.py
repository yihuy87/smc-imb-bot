# telegram/telegram_broadcast.py
# broadcast_signal: kirim teks sinyal ke admin + subscribers

import time

from config import TELEGRAM_ADMIN_ID
from core.bot_state import state, is_vip, cleanup_expired_vip
from telegram.telegram_common import send_telegram


def broadcast_signal(text: str) -> None:
    today = time.strftime("%Y-%m-%d")
    if state.daily_date != today:
        state.daily_date = today
        state.daily_counts = {}
        cleanup_expired_vip()
        print("Reset daily_counts & cleanup VIP untuk hari baru:", today)

    # admin
    if TELEGRAM_ADMIN_ID:
        try:
            send_telegram(text, chat_id=int(TELEGRAM_ADMIN_ID))
        except Exception as e:
            print("Gagal kirim ke admin:", e)
    else:
        print("⚠️ TELEGRAM_ADMIN_ID belum di-set. Admin tidak menerima sinyal.")

    # user
    if not state.subscribers:
        print("Belum ada subscriber. Hanya admin yang menerima sinyal.")
        return

    for cid in list(state.subscribers):
        if TELEGRAM_ADMIN_ID and str(cid) == str(TELEGRAM_ADMIN_ID):
            continue

        if is_vip(cid):
            send_telegram(text, chat_id=cid)
            continue

        count = state.daily_counts.get(cid, 0)
        if count >= 2:
            continue

        send_telegram(text, chat_id=cid)
        state.daily_counts[cid] = count + 1
