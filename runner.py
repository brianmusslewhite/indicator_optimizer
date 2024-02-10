import subprocess
import multiprocessing

configs = [
    ("BCHXBT_1min_Kraken.csv", {
        'apo_fast_min': 10, 'apo_fast_max': 15,  # Shorter for responsiveness
        'apo_slow_min': 20, 'apo_slow_max': 30,  # Longer to capture trends
        'stoch_k_period_min': 14, 'stoch_k_period_max': 21,  # Standard range for capturing momentum
        'stoch_slow_k_period_min': 3, 'stoch_slow_k_period_max': 5,  # Smoothing the %K line
        'stoch_slow_d_period_min': 3, 'stoch_slow_d_period_max': 5,  # Smoothing the %D line
        'obv_ema_period_min': 15, 'obv_ema_period_max': 20,  # Adjusted for quicker volume trend detection
        'bb_period_min': 15, 'bb_period_max': 20,  # Adjusted to capture medium-term volatility
        'bb_dev_lower_min': 2, 'bb_dev_lower_max': 2.5,  # Narrower bands for trend confirmation
        'bb_dev_upper_min': 2, 'bb_dev_upper_max': 2.5,  # Consistent with lower band for symmetry
        'arming_pct_min': 0.5, 'arming_pct_max': 1.0,  # Tightened range for arming percentage
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.2  # Tightened range to manage risk
    }),

    ("ETHXBT_1min_Kraken.csv", {
        'apo_fast_min': 10, 'apo_fast_max': 15,  # Shorter for responsiveness
        'apo_slow_min': 20, 'apo_slow_max': 30,  # Longer to capture trends
        'stoch_k_period_min': 14, 'stoch_k_period_max': 21,  # Standard range for capturing momentum
        'stoch_slow_k_period_min': 3, 'stoch_slow_k_period_max': 5,  # Smoothing the %K line
        'stoch_slow_d_period_min': 3, 'stoch_slow_d_period_max': 5,  # Smoothing the %D line
        'obv_ema_period_min': 15, 'obv_ema_period_max': 20,  # Adjusted for quicker volume trend detection
        'bb_period_min': 15, 'bb_period_max': 20,  # Adjusted to capture medium-term volatility
        'bb_dev_lower_min': 2, 'bb_dev_lower_max': 2.5,  # Narrower bands for trend confirmation
        'bb_dev_upper_min': 2, 'bb_dev_upper_max': 2.5,  # Consistent with lower band for symmetry
        'arming_pct_min': 0.5, 'arming_pct_max': 1.0,  # Tightened range for arming percentage
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.2  # Tightened range to manage risk
    }),

    ("MATICXBT_1min_Kraken.csv", {
        'apo_fast_min': 10, 'apo_fast_max': 15,  # Shorter for responsiveness
        'apo_slow_min': 20, 'apo_slow_max': 30,  # Longer to capture trends
        'stoch_k_period_min': 14, 'stoch_k_period_max': 21,  # Standard range for capturing momentum
        'stoch_slow_k_period_min': 3, 'stoch_slow_k_period_max': 5,  # Smoothing the %K line
        'stoch_slow_d_period_min': 3, 'stoch_slow_d_period_max': 5,  # Smoothing the %D line
        'obv_ema_period_min': 15, 'obv_ema_period_max': 20,  # Adjusted for quicker volume trend detection
        'bb_period_min': 15, 'bb_period_max': 20,  # Adjusted to capture medium-term volatility
        'bb_dev_lower_min': 2, 'bb_dev_lower_max': 2.5,  # Narrower bands for trend confirmation
        'bb_dev_upper_min': 2, 'bb_dev_upper_max': 2.5,  # Consistent with lower band for symmetry
        'arming_pct_min': 0.5, 'arming_pct_max': 1.0,  # Tightened range for arming percentage
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.2  # Tightened range to manage risk
    }),
]

num_cpu_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cpu_cores // len(configs))


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
        '--stoch_slow_k_period_min', str(pbounds['stoch_slow_k_period_min']),
        '--stoch_slow_k_period_max', str(pbounds['stoch_slow_k_period_max']),
        '--stoch_slow_d_period_min', str(pbounds['stoch_slow_d_period_min']),
        '--stoch_slow_d_period_max', str(pbounds['stoch_slow_d_period_max']),
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
    subprocess.Popen(f'gnome-terminal -- bash -c "{command}; exec bash"', shell=True)
