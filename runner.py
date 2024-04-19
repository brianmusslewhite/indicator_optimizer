import subprocess
import multiprocessing
import platform

os_name = platform.system()
if os_name == "Linux" or os_name == "Darwin":
    python_version = 'gamemoderun python3'
elif os_name == "Windows":
    python_version = 'python'

configs = [
    ("ETHXBT_30min_Kraken.csv", {
        'start_date': '2023-9-1', 'end_date': '2023-12-31',
        'ideal_trade_frequency_hours': (24*10),
        'init_points': 200, 'iter_points': 1500,
        'pair_points': 500,
        'stoch_k_period_min': 5, 'stoch_k_period_max': 14,
        'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
        'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
        'stoch_threshold_min': 20, 'stoch_threshold_max': 80,
        'bb_p_min': 1, 'bb_p_max': 40,
        'bb_dev_low_min': 1, 'bb_dev_low_max': 3,
        'bb_dev_up_min': 1, 'bb_dev_up_max': 3,
        'stop_loss_pct_min': 0.5, 'stop_loss_pct_max': 3
    }),

    ("LTCXBT_30min_Kraken.csv", {
        'start_date': '2023-9-1', 'end_date': '2023-12-31',
        'ideal_trade_frequency_hours': (24*10),
        'init_points': 200, 'iter_points': 1500,
        'pair_points': 500,
        'stoch_k_period_min': 5, 'stoch_k_period_max': 14,
        'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
        'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
        'stoch_threshold_min': 20, 'stoch_threshold_max': 80,
        'bb_p_min': 1, 'bb_p_max': 40,
        'bb_dev_low_min': 1, 'bb_dev_low_max': 3,
        'bb_dev_up_min': 1, 'bb_dev_up_max': 3,
        'stop_loss_pct_min': 0.5, 'stop_loss_pct_max': 3
    }),
]

num_cpu_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cpu_cores // len(configs))


for filename, inputs in configs:
    command = ' '.join([
        python_version, 'IO_BB_Stoch.py',
        '--filename', filename,
        '--stoch_k_period_min', str(inputs['stoch_k_period_min']),
        '--stoch_k_period_max', str(inputs['stoch_k_period_max']),
        '--stoch_d_period_min', str(inputs['stoch_d_period_min']),
        '--stoch_d_period_max', str(inputs['stoch_d_period_max']),
        '--stoch_slowing_min', str(inputs['stoch_slowing_min']),
        '--stoch_slowing_max', str(inputs['stoch_slowing_max']),
        '--stoch_threshold_min', str(inputs['stoch_threshold_min']),
        '--stoch_threshold_max', str(inputs['stoch_threshold_max']),
        '--bb_period_min', str(inputs['bb_p_min']),
        '--bb_period_max', str(inputs['bb_p_max']),
        '--bb_dev_low_min', str(inputs['bb_dev_low_min']),
        '--bb_dev_low_max', str(inputs['bb_dev_low_max']),
        '--bb_dev_up_min', str(inputs['bb_dev_up_min']),
        '--bb_dev_up_max', str(inputs['bb_dev_up_max']),
        '--stop_loss_pct_min', str(inputs['stop_loss_pct_min']),
        '--stop_loss_pct_max', str(inputs['stop_loss_pct_max']),
        '--number_of_cores', str(n_jobs),
        '--start_date', str(inputs['start_date']),
        '--end_date', str(inputs['end_date']),
        '--ideal_trade_frequency_hours', str(inputs['ideal_trade_frequency_hours']),
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
