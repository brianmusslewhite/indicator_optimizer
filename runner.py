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
        'start_date': '2023-6-1', 'end_date': '2023-12-31',
        'init_points': 300, 'iter_points': 500,
        'pair_points': 500,
        'stoch_k_period_min': 5, 'stoch_k_period_max': 14,
        'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
        'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
        'stoch_threshold_min': 20, 'stoch_threshold_max': 80,
        'bb_p_min': 1, 'bb_p_max': 40,
        'bb_dev_min': 1, 'bb_dev_max': 3,
        'arming_pct_min': 0.4, 'arming_pct_max': 0.4,
        'arm_stop_loss_pct_min': 0.1, 'arm_stop_loss_pct_max': 0.1,
        'stop_loss_pct_min': 2, 'stop_loss_pct_max': 2
    }),

    ("LTCXBT_30min_Kraken.csv", {
        'start_date': '2023-7-10', 'end_date': '2023-12-31',
        'init_points': 300, 'iter_points': 500,
        'pair_points': 500,
        'stoch_k_period_min': 5, 'stoch_k_period_max': 14,
        'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
        'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
        'stoch_threshold_min': 20, 'stoch_threshold_max': 80,
        'bb_p_min': 1, 'bb_p_max': 40,
        'bb_dev_min': 1, 'bb_dev_max': 3,
        'arming_pct_min': 0.4, 'arming_pct_max': 0.4,
        'arm_stop_loss_pct_min': 0.1, 'arm_stop_loss_pct_max': 0.1,
        'stop_loss_pct_min': 2, 'stop_loss_pct_max': 2
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
        '--bb_dev_min', str(inputs['bb_dev_min']),
        '--bb_dev_max', str(inputs['bb_dev_max']),
        '--arming_pct_min', str(inputs['arming_pct_min']),
        '--arming_pct_max', str(inputs['arming_pct_max']),
        '--arm_stop_loss_pct_min', str(inputs['arm_stop_loss_pct_min']),
        '--arm_stop_loss_pct_max', str(inputs['arm_stop_loss_pct_max']),
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
