import subprocess
import multiprocessing

configs = [
    ("BCHXBT_15min_Kraken.csv", {  # Assuming you'll adjust the filename to reflect 15-minute intervals if necessary
        'macd_fast_min': 5, 'macd_fast_max': 12,  # Exploration range for MACD fast line
        'macd_slow_min': 12, 'macd_slow_max': 26,  # Exploration range for MACD slow line
        'macd_signal_min': 9, 'macd_signal_max': 18,  # Range for MACD signal line smoothing
        'rsi_period_min': 10, 'rsi_period_max': 14,  # Standard short-range for RSI to capture quicker changes
        'rsi_threshold_min': 30, 'rsi_threshold_max': 70,  # Classic overbought/oversold thresholds, adjust based on strategy preference
        'obv_ema_period_min': 15, 'obv_ema_period_max': 20,  # Slightly adjusted for quicker volume trend detection on 15-min data
        'bb_period_min': 15, 'bb_period_max': 20,  # Adjusted for medium-term trends on 15-min charts
        'bb_dev_lower_min': 2, 'bb_dev_lower_max': 2.5,  # Narrower bands for tighter volatility capture
        'bb_dev_upper_min': 2, 'bb_dev_upper_max': 2.5,  # Matching lower bands for consistency
        'arming_pct_min': 0.5, 'arming_pct_max': 1.0,  # Tighter range for quicker stop activation
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.2  # Tightened range to manage risk effectively
    }),

    ("ETHXBT_15min_Kraken.csv", {
        'macd_fast_min': 5, 'macd_fast_max': 12,  # Exploration range for MACD fast line
        'macd_slow_min': 12, 'macd_slow_max': 26,  # Exploration range for MACD slow line
        'macd_signal_min': 9, 'macd_signal_max': 18,  # Range for MACD signal line smoothing
        'rsi_period_min': 10, 'rsi_period_max': 14,  # Standard short-range for RSI to capture quicker changes
        'rsi_threshold_min': 30, 'rsi_threshold_max': 70,  # Classic overbought/oversold thresholds, adjust based on strategy preference
        'obv_ema_period_min': 15, 'obv_ema_period_max': 20,  # Slightly adjusted for quicker volume trend detection on 15-min data
        'bb_period_min': 15, 'bb_period_max': 20,  # Adjusted for medium-term trends on 15-min charts
        'bb_dev_lower_min': 2, 'bb_dev_lower_max': 2.5,  # Narrower bands for tighter volatility capture
        'bb_dev_upper_min': 2, 'bb_dev_upper_max': 2.5,  # Matching lower bands for consistency
        'arming_pct_min': 0.5, 'arming_pct_max': 1.0,  # Tighter range for quicker stop activation
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.2  # Tightened range to manage risk effectively
    }),

    ("MATICXBT_15min_Kraken.csv", {
        'macd_fast_min': 5, 'macd_fast_max': 12,  # Exploration range for MACD fast line
        'macd_slow_min': 12, 'macd_slow_max': 26,  # Exploration range for MACD slow line
        'macd_signal_min': 9, 'macd_signal_max': 18,  # Range for MACD signal line smoothing
        'rsi_period_min': 10, 'rsi_period_max': 14,  # Standard short-range for RSI to capture quicker changes
        'rsi_threshold_min': 30, 'rsi_threshold_max': 70,  # Classic overbought/oversold thresholds, adjust based on strategy preference
        'obv_ema_period_min': 15, 'obv_ema_period_max': 20,  # Slightly adjusted for quicker volume trend detection on 15-min data
        'bb_period_min': 15, 'bb_period_max': 20,  # Adjusted for medium-term trends on 15-min charts
        'bb_dev_lower_min': 2, 'bb_dev_lower_max': 2.5,  # Narrower bands for tighter volatility capture
        'bb_dev_upper_min': 2, 'bb_dev_upper_max': 2.5,  # Matching lower bands for consistency
        'arming_pct_min': 0.5, 'arming_pct_max': 1.0,  # Tighter range for quicker stop activation
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.2  # Tightened range to manage risk effectively
    }),
]

num_cpu_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cpu_cores // len(configs))


for filename, pbounds in configs:
    command = ' '.join([
        'python', 'IO_MACD_RSI_BB_OBV.py',
        '--filename', filename,
        '--macd_fast_min', str(pbounds['macd_fast_min']),
        '--macd_fast_max', str(pbounds['macd_fast_max']),
        '--macd_slow_min', str(pbounds['macd_slow_min']),
        '--macd_slow_max', str(pbounds['macd_slow_max']),
        '--macd_signal_min', str(pbounds['macd_signal_min']),
        '--macd_signal_max', str(pbounds['macd_signal_max']),
        '--rsi_period_min', str(pbounds['rsi_period_min']),
        '--rsi_period_max', str(pbounds['rsi_period_max']),
        '--rsi_threshold_min', str(pbounds['rsi_threshold_min']),
        '--rsi_threshold_max', str(pbounds['rsi_threshold_max']),
        '--obv_ema_period_min', str(pbounds['obv_ema_period_min']),
        '--obv_ema_period_max', str(pbounds['obv_ema_period_max']),
        '--bb_period_min', str(pbounds['bb_period_min']),
        '--bb_period_max', str(pbounds['bb_period_max']),
        '--bb_dev_lower_min', str(pbounds['bb_dev_lower_min']),
        '--bb_dev_lower_max', str(pbounds['bb_dev_lower_max']),
        '--bb_dev_upper_min', str(pbounds['bb_dev_upper_min']),
        '--bb_dev_upper_max', str(pbounds['bb_dev_upper_max']),
        '--arming_pct_min', str(pbounds['arming_pct_min']),
        '--arming_pct_max', str(pbounds['arming_pct_max']),
        '--stop_loss_pct_min', str(pbounds['stop_loss_pct_min']),
        '--stop_loss_pct_max', str(pbounds['stop_loss_pct_max']),
        '--number_of_cores', str(n_jobs),
    ])

    # Open a new terminal window for each optimization run
    # subprocess.Popen(f'gnome-terminal -- bash -c "{command}; exec bash"', shell=True)
    subprocess.Popen(f'cmd /c start cmd /k "{command}"', shell=True)