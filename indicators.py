import pandas as pd
import numpy as np

cache = {}

def calculate_apo(data, fast_period, slow_period):
    key = ('apo', fast_period, slow_period)
    if key not in cache:
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        cache[key] = fast_ema - slow_ema
    return cache[key]

def calculate_bollinger_bands(data, period, deviation_lower, deviation_upper):
    key = ('bollinger_bands', period, deviation_lower, deviation_upper)
    if key not in cache:
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * deviation_upper)
        lower_band = sma - (std * deviation_lower)
        cache[key] = (upper_band, lower_band)
    return cache[key]

def calculate_cci(data, period):
    key = ('cci', period)
    if key not in cache:
        TP = (data['high'] + data['low'] + data['close']) / 3
        TP_mean = TP.rolling(window=period).mean()
        TP_std = TP.rolling(window=period).std()
        CCI = (TP - TP_mean) / (0.015 * TP_std)
        CCI = CCI.replace([np.inf, -np.inf], np.nan)
        cache[key] = CCI
    return cache[key]

def calculate_ema(data, fast_period, slow_period):
    key = ('ema', fast_period, slow_period)
    if key not in cache:
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        cache[key] = (fast_ema, slow_ema)
    return cache[key]

def calculate_macd(data, fast_period, slow_period, signal_period):
    key = ('macd', fast_period, slow_period, signal_period)
    if key not in cache:
        exp1 = data['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = data['close'].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        cache[key] = (macd, signal_line)
    return cache[key]

def calculate_obv(data, ema_period):
    key = ('obv', ema_period)
    if key not in cache:
        obv = np.where(data['close'] > data['close'].shift(), data['volume'], -data['volume']).cumsum()
        obv_ema = pd.Series(obv, index=data.index).ewm(span=ema_period, adjust=False).mean()
        cache[key] = obv_ema
    return cache[key]

def calculate_rsi(data, rsi_period):
    key = ('rsi', rsi_period)
    if key not in cache:
        delta = data['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        cache[key] = rsi
    return cache[key]

def calculate_stochastic_oscillator(data, k_period, slow_k_period, slow_d_period):
    key = ('stochastic_oscillator', k_period, slow_k_period, slow_d_period)
    if key not in cache:
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        fast_k = ((data['close'] - low_min) / (high_max - low_min)) * 100
        slow_k = fast_k.rolling(window=slow_k_period).mean()
        slow_d = slow_k.rolling(window=slow_d_period).mean()
        stoch_avg = (slow_k + slow_d) / 2
        cache[key] = stoch_avg
    return cache[key]