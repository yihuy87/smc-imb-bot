# telegram/telegram_core.py
# Polling loop Telegram: getUpdates, dispatch ke command/callback.

import time
import requests

from config import TELEGRAM_TOKEN, TELEGRAM_ADMIN_USERNAME
from core.bot_state import state, is_admin
from telegram.telegram_common import send_telegram
from telegram.telegram_commands import handle_command, handle_callback
from telegram.telegram_keyboards import get_admin_reply_keyboard


def telegram_command_loop():
    if not TELEGRAM_TOKEN:
        print("Tidak ada TELEGRAM_TOKEN, command loop tidak dijalankan.")
        return

    print("Telegram command loop start...")
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"

    # sync awal: skip pesan lama
    try:
        r = requests.get(url, timeout=20)
        if r.ok:
            data = r.json()
            results = data.get("result", [])
            if results:
                state.last_update_id = results[-1]["update_id"]
                print(f"Sync Telegram: skip {len(results)} pesan lama.")
    except Exception as e:
        print("Error sync awal Telegram:", e)

    while state.running:
        try:
            params: dict = {}
            if state.last_update_id is not None:
                params["offset"] = state.last_update_id + 1

            r = requests.get(url, params=params, timeout=20)
            if not r.ok:
                print("Error getUpdates:", r.text)
                time.sleep(2)
                continue

            data = r.json()
            for upd in data.get("result", []):
                state.last_update_id = upd["update_id"]

                msg = upd.get("message")
                if msg:
                    chat = msg.get("chat", {})
                    chat_id = chat.get("id")
                    text = msg.get("text", "")

                    if not text:
                        continue

                    if text == "🏠 Home":
                        handle_command("/start", [], chat_id)
                        continue

                    if text == "🔔 Aktifkan Sinyal":
                        handle_command("/activate", [], chat_id)
                        continue
                    if text == "🔕 Nonaktifkan Sinyal":
                        handle_command("/deactivate", [], chat_id)
                        continue
                    if text == "📊 Status Saya":
                        handle_command("/mystatus", [], chat_id)
                        continue
                    if text == "⭐ Upgrade VIP" and not is_admin(chat_id):
                        send_telegram(
                            "⭐ *UPGRADE KE VIP*\n\n"
                            "Paket VIP memberikan:\n"
                            "• Sinyal *unlimited* setiap hari\n"
                            "• Akses penuh IMB signal\n\n"
                            "Hubungi admin untuk upgrade:\n"
                            f"`{TELEGRAM_ADMIN_USERNAME}`.",
                            chat_id,
                        )
                        continue
                    if text == "❓ Bantuan" and not is_admin(chat_id):
                        send_telegram(
                            "📖 *BANTUAN PENGGUNA*\n\n"
                            "🔔 Aktifkan Sinyal — hidupkan sinyal.\n"
                            "🔕 Nonaktifkan Sinyal — matikan sinyal.\n"
                            "📊 Status Saya — lihat paket & limit.\n"
                            "⭐ Upgrade VIP — info upgrade.\n",
                            chat_id,
                        )
                        continue

                    # Tombol ADMIN
                    if is_admin(chat_id):
                        if text == "▶️ Start Scan":
                            handle_command("/startscan", [], chat_id)
                            continue
                        if text == "⏸️ Pause Scan":
                            handle_command("/pausescan", [], chat_id)
                            continue
                        if text == "⛔ Stop Scan":
                            handle_command("/stopscan", [], chat_id)
                            continue
                        if text == "📊 Status Bot":
                            handle_command("/status", [], chat_id)
                            continue
                        if text == "⚙️ Mode Tier":
                            send_telegram(
                                "⚙️ *Mode Tier*\n\n"
                                "Gunakan command:\n"
                                "`/mode aplus` — hanya Tier A+\n"
                                "`/mode a`     — Tier A & A+\n"
                                "`/mode b`     — Tier B, A, A+",
                                chat_id,
                            )
                            continue
                        if text == "⏲️ Cooldown":
                            send_telegram(
                                "⏲️ *Cooldown Sinyal*\n\n"
                                "Atur jarak minimal antar sinyal per pair.\n"
                                "Contoh:\n"
                                "`/cooldown 300`  (5 menit)\n"
                                "`/cooldown 900`  (15 menit)\n"
                                "`/cooldown 1800` (30 menit)",
                                chat_id,
                            )
                            continue
                        if text == "📈 Min Volume":
                            send_telegram(
                                "📈 *MINIMUM VOLUME USDT*\n\n"
                                f"Sekarang: `{state.min_volume_usdt:,.0f}` USDT\n\n"
                                "Atur dengan command:\n"
                                "`/minvol 100000000`  (contoh 100 juta USDT)\n",
                                chat_id,
                            )
                            continue
                        if text == "📌 Max Pair":
                            send_telegram(
                                "📌 *MAXIMUM PAIR YANG DI-SCAN*\n\n"
                                f"Sekarang: `{state.max_pairs}` pair\n\n"
                                "Atur dengan command:\n"
                                "`/maxpairs 30`  (scan 30 pair teratas)\n",
                                chat_id,
                            )
                            continue
                        if text == "⭐ VIP Control":
                            send_telegram(
                                "⭐ *VIP CONTROL*\n\n"
                                "Gunakan:\n"
                                "`/addvip <user_id> [hari]` — aktifkan VIP\n"
                                "`/removevip <user_id>` — hapus VIP user\n",
                                chat_id,
                            )
                            continue
                        if text == "🔄 Restart Bot":
                            send_telegram(
                                "Pilih metode restart:",
                                chat_id,
                                reply_markup={
                                    "inline_keyboard": [
                                        [
                                            {
                                                "text": "♻ Soft Restart",
                                                "callback_data": "admin_soft_restart",
                                            },
                                            {
                                                "text": "🔄 Hard Restart",
                                                "callback_data": "admin_hard_restart",
                                            },
                                        ],
                                        [
                                            {
                                                "text": "❌ Batal",
                                                "callback_data": "admin_restart_cancel",
                                            }
                                        ],
                                    ]
                                },
                            )
                            continue
                        if text == "❓ Help Admin":
                            send_telegram(
                                "📖 *BANTUAN ADMIN*\n\n"
                                "▶️ Start Scan / ⏸️ Pause Scan / ⛔ Stop Scan — kontrol scanning.\n"
                                "📊 Status Bot — lihat status.\n"
                                "⚙️ Mode Tier — atur kualitas sinyal.\n"
                                "⏲️ Cooldown — atur jarak antar sinyal.\n"
                                "📈 Min Volume — filter volume minimum USDT.\n"
                                "📌 Max Pair — atur jumlah pair yang discan.\n"
                                "⭐ VIP Control — kelola VIP.\n"
                                "🔄 Restart Bot — Soft/Hard restart bot.\n",
                                chat_id,
                            )
                            continue

                    if not text.startswith("/"):
                        continue

                    parts = text.strip().split()
                    cmd_text = parts[0]
                    args_text = parts[1:]

                    print(f"[TELEGRAM CMD] {chat_id} {cmd_text} {args_text}")
                    handle_command(cmd_text, args_text, chat_id)
                    continue

                cq = upd.get("callback_query")
                if cq:
                    callback_id = cq.get("id")
                    from_id = cq.get("from", {}).get("id")
                    data_cb = cq.get("data")
                    msg_cq = cq.get("message", {})
                    chat_cq = msg_cq.get("chat", {})
                    chat_id_cq = chat_cq.get("id")

                    print(f"[TELEGRAM CB] {from_id} {data_cb}")

                    try:
                        answer_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery"
                        requests.post(
                            answer_url,
                            data={"callback_query_id": callback_id},
                            timeout=10,
                        )
                    except Exception as e:
                        print("Error answerCallbackQuery:", e)

                    if data_cb:
                        handle_callback(data_cb, from_id, chat_id_cq)

        except Exception as e:
            print("Error di telegram_command_loop:", e)
            time.sleep(2)
