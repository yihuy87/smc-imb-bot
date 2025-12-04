# binance/binance_pairs.py
# Ambil dan filter pair USDT perpetual futures berdasarkan volume.

from typing import List, Dict
import requests
import pandas as pd

from config import BINANCE_REST_URL


def get_usdt_pairs(max_pairs: int, min_volume_usdt: float) -> List[str]:
    """
    Ambil semua pair USDT PERPETUAL yang statusnya TRADING,
    lalu filter hanya yang 24h quote volume >= min_volume_usdt USDT.
    Return: list symbol lower-case (ethusdt, btcusdt, ...)
    """
    # 1) Ambil info exchange untuk filter symbol yang valid
    info_url = f"{BINANCE_REST_URL}/fapi/v1/exchangeInfo"
    r = requests.get(info_url, timeout=10)
    r.raise_for_status()
    info = r.json()

    usdt_symbols: List[str] = []
    for s in info.get("symbols", []):
        if (
            s.get("status") == "TRADING"
            and s.get("quoteAsset") == "USDT"
            and s.get("contractType") == "PERPETUAL"
        ):
            usdt_symbols.append(s["symbol"])

    if not usdt_symbols:
        print("Tidak ada USDT perpetual symbols yang ditemukan.")
        return []

    # 2) Ambil data ticker 24h untuk semua symbol
    ticker_url = f"{BINANCE_REST_URL}/fapi/v1/ticker/24hr"
    r2 = requests.get(ticker_url, timeout=10)
    r2.raise_for_status()
    tickers = r2.json()

    # 3) Pakai Pandas untuk memproses volume
    df = pd.DataFrame(tickers)

    # Pastikan kolom yang dibutuhkan ada
    if "symbol" not in df.columns or "quoteVolume" not in df.columns:
        print("Response ticker tidak mengandung kolom symbol/quoteVolume.")
        return []

    # Filter hanya symbol yang ada di usdt_symbols
    df = df[df["symbol"].isin(usdt_symbols)]

    # Convert quoteVolume -> float, handle error jadi 0
    df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce").fillna(0.0)

    min_vol = float(min_volume_usdt)

    # Filter berdasarkan minimum volume
    df = df[df["quoteVolume"] >= min_vol]

    # Urutkan desc berdasarkan volume
    df = df.sort_values("quoteVolume", ascending=False)

    # Ambil list symbol lower-case
    symbols_lower: List[str] = df["symbol"].str.lower().tolist()

    # Batasi jumlah pair jika max_pairs > 0
    if max_pairs > 0:
        symbols_lower = symbols_lower[:max_pairs]

    print(f"Filter volume >= {min_vol:,.0f} USDT → {len(symbols_lower)} pair.")
    return symbols_lower
