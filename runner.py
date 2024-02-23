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
        'start_date': '2023-10-30', 'end_date': '2023-11-29',
        'init_points': 100, 'iter_points':3000,
        'pair_points': 500,
        'macd_fast_min': 3, 'macd_fast_max': 8,
        'macd_slow_min': 10, 'macd_slow_max': 50,
        'macd_signal_min': 5, 'macd_signal_max': 15,
        'bb_period_min': 33, 'bb_period_max': 40,
        'bb_dev_min': 0.7, 'bb_dev_max': 1.3,
        'sar_af_min': 0.06, 'sar_af_max': 0.1,
        'sar_af_max_min': 0.1, 'sar_af_max_max': 0.3,
        'arming_pct_min': 0.25, 'arming_pct_max': 1.0,
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.25
    }),

    ("ETHXBT_15min_Kraken.csv", {
        'start_date': '2023-10-25', 'end_date': '2023-12-31',
        'init_points': 100, 'iter_points':3000,
        'pair_points': 500,
        'macd_fast_min': 3, 'macd_fast_max': 12,
        'macd_slow_min': 10, 'macd_slow_max': 50,
        'macd_signal_min': 5, 'macd_signal_max': 20,
        'bb_period_min': 10, 'bb_period_max': 35,
        'bb_dev_min': 0.5, 'bb_dev_max': 1.5,
        'sar_af_min': 0.005, 'sar_af_max': 0.03,
        'sar_af_max_min': 0.1, 'sar_af_max_max': 0.3,
        'arming_pct_min': 0.25, 'arming_pct_max': 1.0,
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.25
    }),

    ("MATICXBT_15min_Kraken.csv", {
        'start_date': '2023-8-1', 'end_date': '2023-11-1',
        'init_points': 100, 'iter_points':3000,
        'pair_points': 500,
        'macd_fast_min': 3, 'macd_fast_max': 15,
        'macd_slow_min': 10, 'macd_slow_max': 50,
        'macd_signal_min': 5, 'macd_signal_max': 15,
        'bb_period_min': 15, 'bb_period_max': 40,
        'bb_dev_min': 0.5, 'bb_dev_max': 1.4,
        'sar_af_min': 0.01, 'sar_af_max': 0.1,
        'sar_af_max_min': 0.1, 'sar_af_max_max': 0.3,
        'arming_pct_min': 0.25, 'arming_pct_max': 1.0,
        'stop_loss_pct_min': 0.05, 'stop_loss_pct_max': 0.25
    }),
]

num_cpu_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cpu_cores // len(configs))


for filename, inputs in configs:
    command = ' '.join([
        python_version, 'IO_MACD_BB_SAR.py',
        '--filename', filename,
        '--macd_fast_min', str(inputs['macd_fast_min']),
        '--macd_fast_max', str(inputs['macd_fast_max']),
        '--macd_slow_min', str(inputs['macd_slow_min']),
        '--macd_slow_max', str(inputs['macd_slow_max']),
        '--macd_signal_min', str(inputs['macd_signal_min']),
        '--macd_signal_max', str(inputs['macd_signal_max']),
        '--bb_period_min', str(inputs['bb_period_min']),
        '--bb_period_max', str(inputs['bb_period_max']),
        '--bb_dev_min', str(inputs['bb_dev_min']),
        '--bb_dev_max', str(inputs['bb_dev_max']),
        '--sar_af_min', str(inputs['sar_af_min']),
        '--sar_af_max', str(inputs['sar_af_max']),
        '--sar_af_max_min', str(inputs['sar_af_max_min']),
        '--sar_af_max_max', str(inputs['sar_af_max_max']),
        '--arming_pct_min', str(inputs['arming_pct_min']),
        '--arming_pct_max', str(inputs['arming_pct_max']),
        '--stop_loss_pct_min', str(inputs['stop_loss_pct_min']),
        '--stop_loss_pct_max', str(inputs['stop_loss_pct_max']),
        '--number_of_cores', str(n_jobs),
        '--start_date', str(inputs['start_date']),
        '--end_date', str(inputs['end_date']),
        '--init_points', str(inputs['init_points']),
        '--iter_points', str(inputs['iter_points']),
        '--pair_points', str(inputs['pair_points']),
    ])

    if os_name == "Linux" or os_name == "Darwin":
        subprocess.Popen(f'gnome-terminal -- bash -c "{command}; exec bash"', shell=True)
    elif os_name == "Windows":
        subprocess.Popen(f'cmd /c start cmd /k "{command}"', shell=True)
    else:
        print(f"Unsupported operating system: {os_name}")