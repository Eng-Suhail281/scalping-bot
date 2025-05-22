import asyncio
import time
import pandas as pd
import yfinance as yf
from binance import Client
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
from telegram import Bot
from datetime import datetime
import os
import numpy as np

# ====== CONFIG ======
TELEGRAM_TOKEN = '7552938323:AAE9iIT4h5TIT7BZ_fCh8vGMY-OIHzZvDco'
CHANNEL_ID = '@ScalpingMasterz'
BINANCE_API_KEY = 'Q7Kq48n6NyXCn6jQBNcZdz9Igdaeh6QNjVQKdKryF5UFKsGWM4JE3yiQIkeAKC8Q'
BINANCE_SECRET = 'HZVCHqQB6pslysKiUgoMRmgAQKHjRhVvCLTzwIHqmJR7km5ThQOSCvVgp7BJSVDv'
SIGNALS_FILE = "signals.csv"
BALANCE = 1000  # حساب تجريبي
RISK_PERCENT = 1  # نسبة المخاطرة لكل صفقة

# Binance client
binance = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_SECRET)
bot = Bot(token=TELEGRAM_TOKEN)

symbols_yahoo = {'XAU/USD':'GC=F','US100':'^NDX'}
symbols_binance = {'BTC/USDT':'BTCUSDT'}

# Changed timeframe to 5 minutes
yahoo_intervals = '5m'
binance_interval = Client.KLINE_INTERVAL_5MINUTE

# Track open positions to avoid duplicates
open_positions = {}

# Initialize signals file
if not os.path.exists(SIGNALS_FILE):
    pd.DataFrame(columns=["time","symbol","type","entry","tp","sl","status","confidence","size"]).to_csv(SIGNALS_FILE,index=False)

# Position sizing function
def calculate_position_size(balance, risk_percent, entry, sl):
    risk_amount = balance * risk_percent / 100
    risk_per_trade = abs(entry - sl)
    if risk_per_trade == 0:
        return 0
    return round(risk_amount / risk_per_trade, 4)

# Stub for news filter
def has_important_news(symbol):
    # Integrate with an economic news API to return True if upcoming news
    return False


def save_signal(symbol, sig, entry, tp, sl, conf, size):
    df = pd.read_csv(SIGNALS_FILE)
    df.loc[len(df)] = {
        "time": datetime.utcnow(),
        "symbol": symbol,
        "type": sig,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "status": "pending",
        "confidence": conf,
        "size": size
    }
    df.to_csv(SIGNALS_FILE, index=False)
    open_positions[symbol] = sig


def get_yahoo_data(symbol):
    df = yf.download(tickers=symbol, interval=yahoo_intervals, period="5d")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    df.rename(columns=str.lower, inplace=True)
    return df


def get_binance_data(symbol):
    kl = binance.get_klines(symbol=symbol, interval=binance_interval, limit=200)
    df = pd.DataFrame(kl, columns=["time","open","high","low","close","volume","close_time","qav","num_trades","taker_base_vol","taker_quote_vol","ignore"])
    for c in ["close","high","low","volume"]:
        df[c] = df[c].astype(float)
    return df


def calculate_pivot_points(h, l, c):
    pp = (h + l + c) / 3
    r1 = 2*pp - l
    s1 = 2*pp - h
    r2 = pp + (h - l)
    s2 = pp - (h - l)
    return pp, r1, r2, s1, s2


def analyze(df, name=None):
    if df.empty or len(df) < 50:
        return None
    close = df['close']; high = df['high']; low = df['low']; vol = df['volume']
    # news filter
    if has_important_news(name):
        return None
    # current price
    if name == 'BTC/USDT':
        try:
            cp = float(binance.get_symbol_ticker(symbol='BTCUSDT')['price'])
        except:
            cp = close.iloc[-1]
    else:
        cp = close.iloc[-1]
    # indicators
    ema9 = EMAIndicator(close=close, window=9).ema_indicator().iloc[-1]
    ema21 = EMAIndicator(close=close, window=21).ema_indicator().iloc[-1]
    ema50 = EMAIndicator(close=close, window=50).ema_indicator().iloc[-1]
    ema200 = EMAIndicator(close=close, window=200).ema_indicator().iloc[-1]
    trend_up = cp > ema200
    rsi = RSIIndicator(close=close, window=7).rsi().iloc[-1]
    st = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    sk = st.stoch().iloc[-1]; sd = st.stoch_signal().iloc[-1]
    adx = ADXIndicator(high=high, low=low, close=close, window=14).adx().iloc[-1]
    macd_obj = MACD(close=close)
    macd = macd_obj.macd().iloc[-1]; ms = macd_obj.macd_signal().iloc[-1]
    atr = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range().iloc[-1]
    vwap = VolumeWeightedAveragePrice(high=high, low=low, close=close, volume=vol, window=20).vwap.iloc[-1]
    bb = BollingerBands(close=close, window=20, window_dev=2)
    bbu = bb.bollinger_hband().iloc[-1]; bbl = bb.bollinger_lband().iloc[-1]
    pp, r1, r2, s1, s2 = calculate_pivot_points(high.iloc[-2], low.iloc[-2], close.iloc[-2])
    sup = min(low.rolling(20).min().iloc[-1], s1, bbl)
    res = max(high.rolling(20).max().iloc[-1], r1, bbu)
    # dynamic TP/SL
    base_tp = 1.5 * atr / cp; base_sl = 0.8 * atr / cp
    buy = [trend_up, (rsi < 30 or (rsi < 35 and adx > 25)), (sk < 20 and sk > sd), ema9 > ema21 > ema50,
           ((cp - sup) / cp < 0.003), macd > ms, cp > vwap, (vol.iloc[-1] > np.mean(vol.iloc[-5:]))]
    sell = [(not trend_up), (rsi > 70 or (rsi > 65 and adx > 25)), (sk > 80 and sk < sd), ema9 < ema21 < ema50,
            ((res - cp) / cp < 0.003), macd < ms, cp < vwap, (vol.iloc[-1] > np.mean(vol.iloc[-5:]))]
    bs = sum(buy); ss = sum(sell); conf = max(bs, ss) / len(buy)
    if bs >= 6:
        tp = cp * (1 + base_tp * (1 + conf * 0.5))
        sl = cp * (1 - base_sl * (1 - conf * 0.3))
        return 'buy', cp, tp, sl, rsi, sk, sd, ema9, ema21, conf
    if ss >= 6:
        tp = cp * (1 - base_tp * (1 + conf * 0.5))
        sl = cp * (1 + base_sl * (1 - conf * 0.3))
        return 'sell', cp, tp, sl, rsi, sk, sd, ema9, ema21, conf
    return None

async def main():
    cool = 20; last = 0
    while True:
        now = time.time(); msgs = []
        # Yahoo
        for name, sym in symbols_yahoo.items():
            if name in open_positions: continue
            try:
                df = get_yahoo_data(sym); res = analyze(df, name)
                if res:
                    sig, pr, tp, sl, r, sk, sd, e9, e21, cf = res
                    prev = open_positions.get(name)
                    size = calculate_position_size(BALANCE, RISK_PERCENT, pr, sl)
                    if now - last > cool and (not prev or prev[0] != sig or abs(pr - prev[1]) > 0.1):
                        txt = (f"{'🟢 شراء قوي' if sig=='buy' else '🔴 بيع قوي'} {name}\n"
                               f"السعر: {pr:.2f}\nRSI: {r:.1f} | Stoch: {sk:.1f}/{sd:.1f}\n"
                               f"EMA9={e9:.2f},21={e21:.2f}\n🎯{tp:.2f} | 🛑{sl:.2f}\n"
                               f"ثقة: {cf*100:.1f}% | حجم: {size}")
                        msgs.append(txt)
                        save_signal(name, sig, pr, tp, sl, cf, size); last = name_signal = last = time.time()
            except Exception as e: print(f"[Yahoo] {name} error: {e}")
        # Binance
        for name, sym in symbols_binance.items():
            if name in open_positions: continue
            try:
                df = get_binance_data(sym); res = analyze(df, name)
                if res:
                    sig, pr, tp, sl, r, sk, sd, e9, e21, cf = res
                    prev = open_positions.get(name)
                    size = calculate_position_size(BALANCE, RISK_PERCENT, pr, sl)
                    if now - last > cool and (not prev or prev[0] != sig or abs(pr - prev[1]) > 0.1):
                        txt = (f"{'🟢 شراء قوي' if sig=='buy' else '🔴 بيع قوي'} {name}\n"
                               f"السعر: {pr:.2f}\nRSI: {r:.1f} | Stoch: {sk:.1f}/{sd:.1f}\n"
                               f"EMA9={e9:.2f},21={e21:.2f}\n🎯{tp:.2f} | 🛑{sl:.2f}\n"
                               f"ثقة: {cf*100:.1f}% | حجم: {size}")
                        msgs.append(txt)
                        save_signal(name, sig, pr, tp, sl, cf, size); last = name_signal = last = time.time()
            except Exception as e: print(f"[Binance] {name} error: {e}")
        if msgs:
            for m in msgs:
                try: await bot.send_message(chat_id=CHANNEL_ID, text=m)
                except: pass
        else:
            print("Searching🔍")
        await asyncio.sleep(8)

if __name__ == "__main__": asyncio.run(main())
