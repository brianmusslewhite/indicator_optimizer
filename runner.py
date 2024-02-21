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
        'macd_slow_min': 10, 'macd_slow_max': 50,
        'macd_signal_min': 5, 'macd_signal_max': 20,
        'bb_period_min': 5, 'bb_period_max': 40,
        'bb_dev_min': 0.5, 'bb_dev_max': 3.0,
        'sar_af_min': 0.01, 'sar_af_max': 0.1,
        'sar_af_max_min': 0.1, 'sar_af_max_max': 0.3,
        'arming_pct_min': 0.25, 'arming_pct_max': 1.0,
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.25
    }),

    ("ETHXBT_15min_Kraken.csv", {
        'macd_fast_min': 3, 'macd_fast_max': 15,
        'macd_slow_min': 10, 'macd_slow_max': 50,
        'macd_signal_min': 5, 'macd_signal_max': 20,
        'bb_period_min': 5, 'bb_period_max': 40,
        'bb_dev_min': 0.5, 'bb_dev_max': 3.0,
        'sar_af_min': 0.01, 'sar_af_max': 0.1,
        'sar_af_max_min': 0.1, 'sar_af_max_max': 0.3,
        'arming_pct_min': 0.25, 'arming_pct_max': 1.0,
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.25
    }),

    ("MATICXBT_15min_Kraken.csv", {
        'macd_fast_min': 3, 'macd_fast_max': 15,
        'macd_slow_min': 10, 'macd_slow_max': 50,
        'macd_signal_min': 5, 'macd_signal_max': 20,
        'bb_period_min': 5, 'bb_period_max': 40,
        'bb_dev_min': 0.5, 'bb_dev_max': 3.0,
        'sar_af_min': 0.01, 'sar_af_max': 0.1,
        'sar_af_max_min': 0.1, 'sar_af_max_max': 0.3,
        'arming_pct_min': 0.25, 'arming_pct_max': 1.0,
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.25
    }),
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
        '--bb_dev_min', str(pbounds['bb_dev_min']),
        '--bb_dev_max', str(pbounds['bb_dev_max']),
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