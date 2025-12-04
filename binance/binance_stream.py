# binance/binance_stream.py
# WebSocket scanner Binance Futures 5m + IMB analyzer.
# Versi multi-WebSocket:
# - Symbol dibagi beberapa batch
# - Tiap batch pakai koneksi WebSocket sendiri (paralel)
# - Analisa & kirim sinyal tetap per-candle-close 5m

import asyncio
import json
import time
from typing import Dict, List

import requests
import websockets

from config import BINANCE_STREAM_URL, BINANCE_REST_URL, REFRESH_PAIR_INTERVAL_HOURS
from binance.binance_pairs import get_usdt_pairs
from binance.ohlc_buffer import OHLCBufferManager
from core.bot_state import (
    state,
    load_subscribers,
    load_vip_users,
    cleanup_expired_vip,
    load_bot_state,
)
from imb.imb_detector import analyze_symbol_imb
from telegram.telegram_broadcast import broadcast_signal

# setting buffer & preload
MAX_5M_CANDLES = 120
PRELOAD_LIMIT_5M = 60

# ukuran batch symbol per WebSocket (misal 70–100)
WS_BATCH_SIZE = 80


def _fetch_klines(symbol: str, interval: str, limit: int) -> List[list]:
    url = f"{BINANCE_REST_URL}/fapi/v1/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()


def _chunk_list(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


async def _ws_worker(
    name: str,
    symbols: List[str],
    ohlc_mgr: OHLCBufferManager,
    refresh_deadline: float,
):
    """
    Worker WebSocket untuk 1 batch symbol.
    - Connect ke stream kline_5m untuk symbols
    - Update buffer OHLC
    - Setiap candle close + scan ON → analisa IMB + kirim sinyal
    Loop terus sampai:
      - state.running = False, atau
      - state.request_soft_restart = True, atau
      - state.force_pairs_refresh = True, atau
      - time > refresh_deadline
    """
    if not symbols:
        return

    streams = "/".join(f"{s}@kline_5m" for s in symbols)
    ws_url = f"{BINANCE_STREAM_URL}?streams={streams}"

    while state.running:
        # stop kalau sudah saatnya refresh pair / soft restart
        if time.time() > refresh_deadline or state.request_soft_restart or state.force_pairs_refresh:
            break

        try:
            print(f"[{name}] Connect WS: {ws_url}")
            async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                print(f"[{name}] WebSocket connected ({len(symbols)} symbols).")

                while state.running:
                    if (
                        time.time() > refresh_deadline
                        or state.request_soft_restart
                        or state.force_pairs_refresh
                    ):
                        print(f"[{name}] Stop worker (refresh/soft-restart/force_pairs).")
                        return

                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=60)
                    except asyncio.TimeoutError:
                        if state.debug:
                            print(f"[{name}] WS timeout, continue...")
                        continue

                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        if state.debug:
                            print(f"[{name}] Gagal decode JSON.")
                        continue

                    kline = data.get("data", {}).get("k")
                    if not kline:
                        continue

                    symbol = kline.get("s", "").upper()
                    if not symbol:
                        continue

                    # update buffer
                    ohlc_mgr.update_from_kline(symbol, kline)
                    candle_closed = bool(kline.get("x", False))

                    if state.debug and candle_closed:
                        buf_len = len(ohlc_mgr.get_candles(symbol))
                        print(
                            f"[{time.strftime('%H:%M:%S')}] [{name}] 5m close: "
                            f"{symbol} — total candle: {buf_len}"
                        )

                    if not candle_closed:
                        continue
                    if not state.scanning:
                        continue

                    candles = ohlc_mgr.get_candles(symbol)
                    if len(candles) < 40:
                        continue

                    now_ts = time.time()
                    if state.cooldown_seconds > 0:
                        last_ts = state.last_signal_time.get(symbol)
                        if last_ts and now_ts - last_ts < state.cooldown_seconds:
                            if state.debug:
                                print(
                                    f"[{name}] [{symbol}] Skip cooldown "
                                    f"({int(now_ts - last_ts)}s/{state.cooldown_seconds}s)"
                                )
                            continue

                    # ANALISA IMB
                    result = analyze_symbol_imb(symbol, candles)
                    if not result:
                        continue

                    text = result["message"]
                    broadcast_signal(text)

                    state.last_signal_time[symbol] = now_ts
                    print(
                        f"[{name}] [{symbol}] IMB sinyal dikirim: "
                        f"Tier {result['tier']} (Score {result['score']}) "
                        f"Entry {result['entry']:.6f} SL {result['sl']:.6f}"
                    )

        except websockets.ConnectionClosed:
            print(f"[{name}] WebSocket terputus. Reconnect 3s...")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"[{name}] Error worker:", e)
            await asyncio.sleep(3)

    print(f"[{name}] Worker selesai (state.running={state.running}).")


async def run_imb_bot():
    """
    Main loop SMC IMB Bot:
    - Load subscribers/VIP/state
    - Ambil daftar pair USDT perpetual (filter volume)
    - Preload 5m history via REST
    - Bagi symbol ke beberapa batch, tiap batch 1 WebSocket worker
    - Worker jalan paralel (async), analisa per candle close
    - Refresh pair periodik sesuai REFRESH_PAIR_INTERVAL_HOURS
    """

    # Load state persistent
    state.subscribers = load_subscribers()
    state.vip_users = load_vip_users()
    state.daily_date = time.strftime("%Y-%m-%d")
    cleanup_expired_vip()
    load_bot_state()

    print(f"Loaded {len(state.subscribers)} subscribers, {len(state.vip_users)} VIP users.")

    symbols: List[str] = []
    last_pairs_refresh: float = 0.0
    refresh_interval = REFRESH_PAIR_INTERVAL_HOURS * 3600

    ohlc_mgr = OHLCBufferManager(max_candles=MAX_5M_CANDLES)

    while state.running:
        try:
            now = time.time()
            need_refresh_pairs = (
                not symbols
                or (now - last_pairs_refresh) > refresh_interval
                or state.force_pairs_refresh
            )

            if need_refresh_pairs:
                print("Refresh daftar pair USDT perpetual berdasarkan volume...")
                symbols = get_usdt_pairs(state.max_pairs, state.min_volume_usdt)
                last_pairs_refresh = time.time()
                state.force_pairs_refresh = False

                print(f"Scan {len(symbols)} pair:", ", ".join(s.upper() for s in symbols))

                # preload 60 candle 5m untuk tiap symbol
                print(
                    f"Mulai preload history 5m untuk {len(symbols)} symbol "
                    f"(limit={PRELOAD_LIMIT_5M})..."
                )
                for sym in symbols:
                    try:
                        kl = _fetch_klines(sym.upper(), "5m", PRELOAD_LIMIT_5M)
                        ohlc_mgr.preload_candles(sym.upper(), kl)
                    except Exception as e:
                        print(f"[{sym}] Gagal preload 5m:", e)
                        continue
                print("Preload selesai.")

            if not symbols:
                print("Tidak ada symbol untuk discan. Tidur sebentar...")
                await asyncio.sleep(5)
                continue

            # buat batch symbol untuk beberapa WebSocket
            batches = _chunk_list(symbols, WS_BATCH_SIZE)
            refresh_deadline = last_pairs_refresh + refresh_interval

            print(
                f"Mulai WS workers: {len(batches)} batch, "
                f"{len(symbols)} symbol total, refresh_deadline={time.strftime('%H:%M:%S', time.localtime(refresh_deadline))}"
            )

            # jalankan semua worker paralel
            tasks = []
            for idx, batch in enumerate(batches):
                name = f"WS-{idx+1}"
                t = asyncio.create_task(
                    _ws_worker(name, batch, ohlc_mgr, refresh_deadline)
                )
                tasks.append(t)

            # tunggu sampai:
            # - semua worker selesai, atau
            # - ada worker yang selesai duluan, lalu kita lanjut loop dan cek kondisi
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            # kalau ada yang masih pending, cancel semua & tunggu selesai
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            # reset soft restart flag (kalau ada)
            if state.request_soft_restart:
                print("Soft restart selesai: worker dihentikan & loop utama lanjut.")
                state.request_soft_restart = False

            # loop akan lanjut, cek lagi need_refresh_pairs / running / dsb.

        except Exception as e:
            print("Error di run_imb_bot (luar):", e)
            print("Coba ulang loop utama dalam 5 detik...")
            await asyncio.sleep(5)

    print("run_imb_bot selesai karena state.running = False")
