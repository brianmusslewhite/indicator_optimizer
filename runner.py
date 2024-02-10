import subprocess

configs = [
    ("BCHXBT_1min_Kraken.csv", {
        'apo_fast_min': 10, 'apo_fast_max': 30,  # Wider range for APO fast period
        'apo_slow_min': 20, 'apo_slow_max': 60,  # Wider range for APO slow period
        'stoch_k_period_min': 10, 'stoch_k_period_max': 25,  # Wider range for Stochastic K period
        'slow_k_period_min': 3, 'slow_k_period_max': 8,  # Wider range for Stochastic Slow K period
        'slow_d_period_min': 3, 'slow_d_period_max': 8,  # Wider range for Stochastic Slow D period
        'obv_ema_period_min': 5, 'obv_ema_period_max': 30,  # Wider range for OBV EMA period
        'bb_period_min': 5, 'bb_period_max': 30,  # Wider range for Bollinger Bands period
        'bb_dev_lower_min': 1, 'bb_dev_lower_max': 3,  # Wider range for Bollinger Bands lower deviation
        'bb_dev_upper_min': 1, 'bb_dev_upper_max': 3,  # Wider range for Bollinger Bands upper deviation
        'arming_pct_min': 0.3, 'arming_pct_max': 1.4,  # Wider range for arming percentage
        'stop_loss_pct_min': 0.03, 'stop_loss_pct_max': 0.3  # Wider range for stop loss percentage
    }),

    ("ETHXBT_1min_Kraken.csv", {
        'apo_fast_min': 10, 'apo_fast_max': 30,  # Wider range for APO fast period
        'apo_slow_min': 20, 'apo_slow_max': 60,  # Wider range for APO slow period
        'stoch_k_period_min': 10, 'stoch_k_period_max': 25,  # Wider range for Stochastic K period
        'slow_k_period_min': 3, 'slow_k_period_max': 8,  # Wider range for Stochastic Slow K period
        'slow_d_period_min': 3, 'slow_d_period_max': 8,  # Wider range for Stochastic Slow D period
        'obv_ema_period_min': 5, 'obv_ema_period_max': 30,  # Wider range for OBV EMA period
        'bb_period_min': 5, 'bb_period_max': 30,  # Wider range for Bollinger Bands period
        'bb_dev_lower_min': 1, 'bb_dev_lower_max': 3,  # Wider range for Bollinger Bands lower deviation
        'bb_dev_upper_min': 1, 'bb_dev_upper_max': 3,  # Wider range for Bollinger Bands upper deviation
        'arming_pct_min': 0.3, 'arming_pct_max': 1.4,  # Wider range for arming percentage
        'stop_loss_pct_min': 0.03, 'stop_loss_pct_max': 0.3  # Wider range for stop loss percentage
    }),

    ("MATICXBT_1min_Kraken.csv", {
        'apo_fast_min': 10, 'apo_fast_max': 30,  # Wider range for APO fast period
        'apo_slow_min': 20, 'apo_slow_max': 60,  # Wider range for APO slow period
        'stoch_k_period_min': 10, 'stoch_k_period_max': 25,  # Wider range for Stochastic K period
        'slow_k_period_min': 3, 'slow_k_period_max': 8,  # Wider range for Stochastic Slow K period
        'slow_d_period_min': 3, 'slow_d_period_max': 8,  # Wider range for Stochastic Slow D period
        'obv_ema_period_min': 5, 'obv_ema_period_max': 30,  # Wider range for OBV EMA period
        'bb_period_min': 5, 'bb_period_max': 30,  # Wider range for Bollinger Bands period
        'bb_dev_lower_min': 1, 'bb_dev_lower_max': 3,  # Wider range for Bollinger Bands lower deviation
        'bb_dev_upper_min': 1, 'bb_dev_upper_max': 3,  # Wider range for Bollinger Bands upper deviation
        'arming_pct_min': 0.3, 'arming_pct_max': 1.4,  # Wider range for arming percentage
        'stop_loss_pct_min': 0.03, 'stop_loss_pct_max': 0.3  # Wider range for stop loss percentage
    }),
]


for filename, pbounds in configs:
    command = ' '.join([
        'python3', 'indicator_optimizer_APO_BB_STOCH_OBV.py',
        '--filename', filename,
        '--apo_fast_min', str(pbounds['apo_fast_min']),
        '--apo_fast_max', str(pbounds['apo_fast_max']),
        '--apo_slow_min', str(pbounds['apo_slow_min']),
        '--apo_slow_max', str(pbounds['apo_slow_max']),
        '--stoch_k_period_min', str(pbounds['stoch_k_period_min']),
        '--stoch_k_period_max', str(pbounds['stoch_k_period_max']),
        '--slow_k_period_min', str(pbounds['slow_k_period_min']),
        '--slow_k_period_max', str(pbounds['slow_k_period_max']),
        '--slow_d_period_min', str(pbounds['slow_d_period_min']),
        '--slow_d_period_max', str(pbounds['slow_d_period_max']),
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
    ])

    # Open a new terminal window for each optimization run
    subprocess.Popen(f'gnome-terminal -- bash -c "{command}; exec bash"', shell=True)
