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
        'start_date': '2023-11-1', 'end_date': '2023-12-31',
        'ideal_trade_frequency_hours_min': (24*1),
        'ideal_trade_frequency_hours_max': (24*7),
        'init_points': 500, 'iter_points': 10000,
        'pair_points': 1000,
        'stoch_k_period_min': 14, 'stoch_k_period_max': 24,
        'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
        'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
        'stoch_threshold_min': 30, 'stoch_threshold_max': 80,
        'bb_p_min': 12, 'bb_p_max': 25,
        'bb_dev_low_min': 1.0, 'bb_dev_low_max': 2.5,
        'bb_dev_up_min': 1.5, 'bb_dev_up_max': 2.5,
        'cci_p_min': 10, 'cci_p_max': 60,
        'stop_loss_pct_min': 0.75, 'stop_loss_pct_max': 2
    }),

    # ("LTCXBT_30min_Kraken.csv", {
    #     'start_date': '2023-11-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (24*1),
    #     'ideal_trade_frequency_hours_max': (24*7),
    #     'init_points': 500, 'iter_points': 10000,
    #     'pair_points': 1000,
    #     'stoch_k_period_min': 14, 'stoch_k_period_max': 24,
    #     'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
    #     'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
    #     'stoch_threshold_min': 30, 'stoch_threshold_max': 80,
    #     'bb_p_min': 12, 'bb_p_max': 25,
    #     'bb_dev_low_min': 1.0, 'bb_dev_low_max': 2.5,
    #     'bb_dev_up_min': 1.5, 'bb_dev_up_max': 2.5,
    #     'cci_p_min': 10, 'cci_p_max': 60,
    #     'stop_loss_pct_min': 0.75, 'stop_loss_pct_max': 2
    # }),

    # ("MATICXBT_30min_Kraken.csv", {
    #     'start_date': '2023-11-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (24*1),
    #     'ideal_trade_frequency_hours_max': (24*7),
    #     'init_points': 500, 'iter_points': 10000,
    #     'pair_points': 1000,
    #     'stoch_k_period_min': 14, 'stoch_k_period_max': 24,
    #     'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
    #     'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
    #     'stoch_threshold_min': 30, 'stoch_threshold_max': 80,
    #     'bb_p_min': 12, 'bb_p_max': 25,
    #     'bb_dev_low_min': 1.0, 'bb_dev_low_max': 2.5,
    #     'bb_dev_up_min': 1.5, 'bb_dev_up_max': 2.5,
    #     'cci_p_min': 10, 'cci_p_max': 60,
    #     'stop_loss_pct_min': 0.75, 'stop_loss_pct_max': 2
    # }),

    # ("XDGXBT_30min_Kraken.csv", {
    #     'start_date': '2023-11-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (24*1),
    #     'ideal_trade_frequency_hours_max': (24*7),
    #     'init_points': 500, 'iter_points': 10000,
    #     'pair_points': 1000,
    #     'stoch_k_period_min': 14, 'stoch_k_period_max': 24,
    #     'stoch_d_period_min': 3, 'stoch_d_period_max': 8,
    #     'stoch_slowing_min': 1, 'stoch_slowing_max': 5,
    #     'stoch_threshold_min': 30, 'stoch_threshold_max': 80,
    #     'bb_p_min': 12, 'bb_p_max': 25,
    #     'bb_dev_low_min': 1.0, 'bb_dev_low_max': 2.5,
    #     'bb_dev_up_min': 1.5, 'bb_dev_up_max': 2.5,
    #     'cci_p_min': 10, 'cci_p_max': 60,
    #     'stop_loss_pct_min': 0.75, 'stop_loss_pct_max': 2
    # }),
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
        '--cci_p_min', str(inputs['cci_p_min']),
        '--cci_p_max', str(inputs['cci_p_max']),
        '--stop_loss_pct_min', str(inputs['stop_loss_pct_min']),
        '--stop_loss_pct_max', str(inputs['stop_loss_pct_max']),
        '--number_of_cores', str(n_jobs),
        '--start_date', str(inputs['start_date']),
        '--end_date', str(inputs['end_date']),
        '--ideal_trade_frequency_hours_min', str(inputs['ideal_trade_frequency_hours_min']),
        '--ideal_trade_frequency_hours_max', str(inputs['ideal_trade_frequency_hours_max']),
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
