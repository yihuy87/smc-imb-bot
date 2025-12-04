# main.py
# Entry point: start Telegram command loop + Binance IMB stream loop.

import asyncio
import threading

from core.bot_state import state
from telegram.telegram_core import telegram_command_loop
from binance.binance_stream import run_imb_bot


if __name__ == "__main__":
    # Jalankan loop command Telegram di thread terpisah
    cmd_thread = threading.Thread(target=telegram_command_loop, daemon=True)
    cmd_thread.start()

    try:
        asyncio.run(run_imb_bot())
    except KeyboardInterrupt:
        state.running = False
        print("Bot dihentikan oleh user (CTRL+C).")
