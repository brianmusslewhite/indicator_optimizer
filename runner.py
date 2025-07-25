import subprocess
import multiprocessing
import platform

os_name = platform.system()
if os_name == "Linux" or os_name == "Darwin":
    python_version = 'gamemoderun python3'
elif os_name == "Windows":
    python_version = 'python'

configs = [
    ("ETHXBT_15min_Kraken.csv", {
        'start_date': '2023-12-1', 'end_date': '2023-12-31',
        'init_points': 1000, 'iter_points': 20000,
        'pair_points': 500,
        'rsi_period_min': 2, 'rsi_period_max': 15,
        'rsi_threshold_min': 20, 'rsi_threshold_max': 80,
        'ema_short_period_min': 2, 'ema_short_period_max': 10,
        'ema_long_period_min': 11, 'ema_long_period_max': 30,
        'ema_persistence_min': 1, 'ema_persistence_max': 10,
        'stoch_k_period_min': 5, 'stoch_k_period_max': 14,
        'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
        'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
        'stoch_threshold_min': 20, 'stoch_threshold_max': 80,
        'obv_ema_period_min': 1, 'obv_ema_period_max': 10,
        'obv_persistence_min': 1, 'obv_persistence_max': 10,
        'arming_pct_min': 0.7, 'arming_pct_max': 1.4,
        'arm_stop_loss_pct_min': 0.1, 'arm_stop_loss_pct_max': 0.3,
        'stop_loss_pct_min': 1, 'stop_loss_pct_max': 3
    }),

    ("LTCXBT_15min_Kraken.csv", {
        'start_date': '2023-12-1', 'end_date': '2023-12-31',
        'init_points': 1000, 'iter_points': 20000,
        'pair_points': 500,
        'rsi_period_min': 2, 'rsi_period_max': 15,
        'rsi_threshold_min': 20, 'rsi_threshold_max': 80,
        'ema_short_period_min': 2, 'ema_short_period_max': 10,
        'ema_long_period_min': 11, 'ema_long_period_max': 30,
        'ema_persistence_min': 1, 'ema_persistence_max': 10,
        'stoch_k_period_min': 5, 'stoch_k_period_max': 14,
        'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
        'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
        'stoch_threshold_min': 20, 'stoch_threshold_max': 80,
        'obv_ema_period_min': 1, 'obv_ema_period_max': 10,
        'obv_persistence_min': 1, 'obv_persistence_max': 10,
        'arming_pct_min': 0.7, 'arming_pct_max': 1.4,
        'arm_stop_loss_pct_min': 0.1, 'arm_stop_loss_pct_max': 0.3,
        'stop_loss_pct_min': 1, 'stop_loss_pct_max': 3
    }),

    # ("LTCXBT_15min_Kraken.csv", {
    #     'start_date': '2023-9-1', 'end_date': '2023-12-31',
    #     'init_points': 1000, 'iter_points': 2000,
    #     'pair_points': 500,
    #     'rsi_period_min': 2, 'rsi_period_max': 30,
    #     'rsi_threshold_min': 5, 'rsi_threshold_max': 50,
    #     'ema_short_period_min': 2, 'ema_short_period_max': 20,
    #     'ema_long_period_min': 5, 'ema_long_period_max': 50,
    #     'ema_persistence_min': 1, 'ema_persistence_max': 8,
    #     'stoch_k_period_min': 4, 'stoch_k_period_max': 20,
    #     'stoch_d_period_min': 2, 'stoch_d_period_max': 10,
    #     'stoch_slowing_min': 2, 'stoch_slowing_max': 8,
    #     'stoch_threshold_min': 5, 'stoch_threshold_max': 50,
    #     'obv_ema_period_min': 1, 'obv_ema_period_max': 20,
    #     'obv_persistence_min': 1, 'obv_persistence_max': 8,
    #     'arming_pct_min': 0.7, 'arming_pct_max': 1.4,
    #     'arm_stop_loss_pct_min': 0.1, 'arm_stop_loss_pct_max': 0.3,
    #     'stop_loss_pct_min': 1, 'stop_loss_pct_max': 3
    # }),

]

num_cpu_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cpu_cores // len(configs))


for filename, inputs in configs:
    command = ' '.join([
        python_version, 'IO_RSI_STOCH_EMA_OBV.py',
        '--filename', filename,
        '--rsi_period_min', str(inputs['rsi_period_min']),
        '--rsi_period_max', str(inputs['rsi_period_max']),
        '--rsi_threshold_min', str(inputs['rsi_threshold_min']),
        '--rsi_threshold_max', str(inputs['rsi_threshold_max']),
        '--ema_short_period_min', str(inputs['ema_short_period_min']),
        '--ema_short_period_max', str(inputs['ema_short_period_max']),
        '--ema_long_period_min', str(inputs['ema_long_period_min']),
        '--ema_long_period_max', str(inputs['ema_long_period_max']),
        '--ema_persistence_min', str(inputs['ema_persistence_min']),
        '--ema_persistence_max', str(inputs['ema_persistence_max']),
        '--stoch_k_period_min', str(inputs['stoch_k_period_min']),
        '--stoch_k_period_max', str(inputs['stoch_k_period_max']),
        '--stoch_d_period_min', str(inputs['stoch_d_period_min']),
        '--stoch_d_period_max', str(inputs['stoch_d_period_max']),
        '--stoch_slowing_min', str(inputs['stoch_slowing_min']),
        '--stoch_slowing_max', str(inputs['stoch_slowing_max']),
        '--stoch_threshold_min', str(inputs['stoch_threshold_min']),
        '--stoch_threshold_max', str(inputs['stoch_threshold_max']),
        '--obv_ema_period_min', str(inputs['obv_ema_period_min']),
        '--obv_ema_period_max', str(inputs['obv_ema_period_max']),
        '--obv_persistence_min', str(inputs['obv_persistence_min']),
        '--obv_persistence_max', str(inputs['obv_persistence_max']),
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
