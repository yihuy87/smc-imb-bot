# SMC IMB Bot (IMB Engine)

Bot Telegram untuk sinyal trading berbasis Institutional Mitigation Block (IMB).
Strategi ini membaca jejak institusi setelah displacement kuat, kemudian mencari mitigation retrace yang menjadi entry presisi & tidak mudah dimanipulasi market maker.

Bot otomatis scan banyak pair USDT di Binance Futures, lalu kirim sinyal ketika ada IMB valid.

Contoh format sinyal:


🟢 SMC SIGNAL — BTCUSDT (LONG)  
Entry : 67350  
SL    : 67080  
TP1   : 67500  
TP2   : 67720  
TP3   : 68100  
Model : Displacement → IMB Mitigation Entry  
Rekomendasi Leverage : 15x–25x (SL 0.40%)

---

## Setup

### 1. Buat bot Telegram

- Chat ke **@BotFather**
- `/newbot` → ambil **BOT TOKEN**

### 2. Ambil chat ID admin

- Chat ke **@userinfobot**
- Catat `Your user ID` → itu **TELEGRAM_ADMIN_ID**

### 3. Clone / download repo ini

```bash
git clone https://github.com/yihuy87/smc-imb-bot.git
cd smc-imb-bot
