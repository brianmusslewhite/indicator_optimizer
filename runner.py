import subprocess
import multiprocessing
import platform

os_name = platform.system()
if os_name == "Linux" or os_name == "Darwin":
    python_version = 'python3'
elif os_name == "Windows":
    python_version = 'python'

configs = [
    ("BCHXBT_15min_Kraken.csv", {
        'macd_fast_min': 3, 'macd_fast_max': 15,
        'macd_slow_min': 5, 'macd_slow_max': 30,
        'macd_signal_min': 3, 'macd_signal_max': 20,
        'bb_period_min': 5, 'bb_period_max': 40,
        'bb_dev_lower_min': 1.5, 'bb_dev_lower_max': 3.0,
        'bb_dev_upper_min': 1.5, 'bb_dev_upper_max': 3.0,
        'sar_af_min': 0.01, 'sar_af_max': 0.05,
        'sar_af_max_min': 0.2, 'sar_af_max_max': 0.6,
        'arming_pct_min': 0.25, 'arming_pct_max': 1.0,
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.3
    }),

    # ("ETHXBT_15min_Kraken.csv", {
    #     'macd_fast_min': 5, 'macd_fast_max': 12,  # Exploration range for MACD fast line
    #     'macd_slow_min': 12, 'macd_slow_max': 26,  # Exploration range for MACD slow line
    #     'macd_signal_min': 9, 'macd_signal_max': 18,  # Range for MACD signal line smoothing
    #     'rsi_period_min': 10, 'rsi_period_max': 14,  # Standard short-range for RSI to capture quicker changes
    #     'rsi_threshold_min': 30, 'rsi_threshold_max': 70,  # Classic overbought/oversold thresholds, adjust based on strategy preference
    #     'obv_ema_period_min': 15, 'obv_ema_period_max': 20,  # Slightly adjusted for quicker volume trend detection on 15-min data
    #     'bb_period_min': 15, 'bb_period_max': 20,  # Adjusted for medium-term trends on 15-min charts
    #     'bb_dev_lower_min': 2, 'bb_dev_lower_max': 2.5,  # Narrower bands for tighter volatility capture
    #     'bb_dev_upper_min': 2, 'bb_dev_upper_max': 2.5,  # Matching lower bands for consistency
    #     'arming_pct_min': 0.5, 'arming_pct_max': 1.0,  # Tighter range for quicker stop activation
    #     'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.2  # Tightened range to manage risk effectively
    # }),

    # ("MATICXBT_15min_Kraken.csv", {
    #     'macd_fast_min': 5, 'macd_fast_max': 12,  # Exploration range for MACD fast line
    #     'macd_slow_min': 12, 'macd_slow_max': 26,  # Exploration range for MACD slow line
    #     'macd_signal_min': 9, 'macd_signal_max': 18,  # Range for MACD signal line smoothing
    #     'rsi_period_min': 10, 'rsi_period_max': 14,  # Standard short-range for RSI to capture quicker changes
    #     'rsi_threshold_min': 30, 'rsi_threshold_max': 70,  # Classic overbought/oversold thresholds, adjust based on strategy preference
    #     'obv_ema_period_min': 15, 'obv_ema_period_max': 20,  # Slightly adjusted for quicker volume trend detection on 15-min data
    #     'bb_period_min': 15, 'bb_period_max': 20,  # Adjusted for medium-term trends on 15-min charts
    #     'bb_dev_lower_min': 2, 'bb_dev_lower_max': 2.5,  # Narrower bands for tighter volatility capture
    #     'bb_dev_upper_min': 2, 'bb_dev_upper_max': 2.5,  # Matching lower bands for consistency
    #     'arming_pct_min': 0.5, 'arming_pct_max': 1.0,  # Tighter range for quicker stop activation
    #     'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.2  # Tightened range to manage risk effectively
    # }),
]

num_cpu_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cpu_cores // len(configs))


for filename, pbounds in configs:
    command = ' '.join([
        python_version, 'IO_MACD_BB_SAR.py',
        '--filename', filename,
        '--macd_fast_min', str(pbounds['macd_fast_min']),
        '--macd_fast_max', str(pbounds['macd_fast_max']),
        '--macd_slow_min', str(pbounds['macd_slow_min']),
        '--macd_slow_max', str(pbounds['macd_slow_max']),
        '--macd_signal_min', str(pbounds['macd_signal_min']),
        '--macd_signal_max', str(pbounds['macd_signal_max']),
        '--bb_period_min', str(pbounds['bb_period_min']),
        '--bb_period_max', str(pbounds['bb_period_max']),
        '--bb_dev_lower_min', str(pbounds['bb_dev_lower_min']),
        '--bb_dev_lower_max', str(pbounds['bb_dev_lower_max']),
        '--bb_dev_upper_min', str(pbounds['bb_dev_upper_min']),
        '--bb_dev_upper_max', str(pbounds['bb_dev_upper_max']),
        '--sar_af_min', str(pbounds['sar_af_min']),
        '--sar_af_max', str(pbounds['sar_af_max']),
        '--sar_af_max_min', str(pbounds['sar_af_max_min']),
        '--sar_af_max_max', str(pbounds['sar_af_max_max']),
        '--arming_pct_min', str(pbounds['arming_pct_min']),
        '--arming_pct_max', str(pbounds['arming_pct_max']),
        '--stop_loss_pct_min', str(pbounds['stop_loss_pct_min']),
        '--stop_loss_pct_max', str(pbounds['stop_loss_pct_max']),
        '--number_of_cores', str(n_jobs),
    ])

    if os_name == "Linux" or os_name == "Darwin":
        subprocess.Popen(f'gnome-terminal -- bash -c "{command}; exec bash"', shell=True)
    elif os_name == "Windows":
        subprocess.Popen(f'cmd /c start cmd /k "{command}"', shell=True)
    else:
        print(f"Unsupported operating system: {os_name}")