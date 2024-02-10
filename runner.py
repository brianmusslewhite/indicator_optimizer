import subprocess

configs = [
    ("BCHXBT_1min_Kraken.csv", {
        'apo_fast_min': 12, 'apo_fast_max': 26,
        'apo_slow_min': 26, 'apo_slow_max': 52,
        'atr_ema_period_min': 10, 'atr_ema_period_max': 20,  # EMA Period for ATR
        'natr_period_min': 14, 'natr_period_max': 21,  # NATR Period
        'stoch_k_period_min': 14, 'stoch_k_period_max': 20,
        'slow_k_period_min': 3, 'slow_k_period_max': 5,
        'slow_d_period_min': 3, 'slow_d_period_max': 5,
        'obv_ema_period_min': 10, 'obv_ema_period_max': 20,
        'arming_pct_min': 0.5, 'arming_pct_max': 1.5,
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.4
    }),

    # Add more configurations for other CSV files as needed
]

for filename, pbounds in configs:
    command = ' '.join([
        'python3', 'indicator_optimizerAPO_ATR_STOCH_OBV.py',
        '--filename', filename,
        '--apo_fast_min', str(pbounds['apo_fast_min']),
        '--apo_fast_max', str(pbounds['apo_fast_max']),
        '--apo_slow_min', str(pbounds['apo_slow_min']),
        '--apo_slow_max', str(pbounds['apo_slow_max']),
        '--atr_ema_period_min', str(pbounds['atr_ema_period_min']),
        '--atr_ema_period_max', str(pbounds['atr_ema_period_max']),
        '--natr_period_min', str(pbounds['natr_period_min']),
        '--natr_period_max', str(pbounds['natr_period_max']),
        '--stoch_k_period_min', str(pbounds['stoch_k_period_min']),
        '--stoch_k_period_max', str(pbounds['stoch_k_period_max']),
        '--slow_k_period_min', str(pbounds['slow_k_period_min']),
        '--slow_k_period_max', str(pbounds['slow_k_period_max']),
        '--slow_d_period_min', str(pbounds['slow_d_period_min']),
        '--slow_d_period_max', str(pbounds['slow_d_period_max']),
        '--obv_ema_period_min', str(pbounds['obv_ema_period_min']),
        '--obv_ema_period_max', str(pbounds['obv_ema_period_max']),
        '--arming_pct_min', str(pbounds['arming_pct_min']),
        '--arming_pct_max', str(pbounds['arming_pct_max']),
        '--stop_loss_pct_min', str(pbounds['stop_loss_pct_min']),
        '--stop_loss_pct_max', str(pbounds['stop_loss_pct_max']),
    ])

    subprocess.Popen(f'gnome-terminal -- bash -c "{command}; exec bash"', shell=True)
