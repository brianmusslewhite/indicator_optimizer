import pandas as pd
import numpy as np

def calculate_apo(data, fast_period, slow_period):
    fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
    return fast_ema - slow_ema

def calculate_bollinger_bands(data, period, deviation_lower, deviation_upper):
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper_band = sma + (std * deviation_upper)
    lower_band = sma - (std * deviation_lower)
    return upper_band, lower_band

def calculate_cci(data, period):
    TP = (data['high'] + data['low'] + data['close']) / 3
    TP_mean = TP.rolling(window=period).mean()
    TP_std = TP.rolling(window=period).std()
    CCI = (TP - TP_mean) / (0.015 * TP_std)
    CCI = CCI.replace([np.inf, -np.inf], np.nan)
    return CCI

def calculate_ema(data, fast_period, slow_period):
    fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
    return fast_ema, slow_ema

def calculate_obv(data, ema_period):
    obv = np.where(data['close'] > data['close'].shift(), data['volume'], -data['volume']).cumsum()
    obv_ema = pd.Series(obv, index=data.index).ewm(span=ema_period, adjust=False).mean()
    return obv_ema

def calculate_rsi(data, rsi_period):
    delta = data['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic_oscillator(data, k_period, slow_k_period, slow_d_period):
    low_min = data['low'].rolling(window=k_period).min()
    high_max = data['high'].rolling(window=k_period).max()
    fast_k = ((data['close'] - low_min) / (high_max - low_min)) * 100

    slow_k = fast_k.rolling(window=slow_k_period).mean()
    slow_d = slow_k.rolling(window=slow_d_period).mean()
    stoch_avg = (slow_k + slow_d) / 2
    return stoch_avg
