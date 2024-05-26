import subprocess
import multiprocessing
import platform

os_name = platform.system()
if os_name == "Linux" or os_name == "Darwin":
    python_version = 'python3'
elif os_name == "Windows":
    python_version = 'python'

configs = [
    ("ETHXBT_15min_Kraken.csv", {
        'start_date': '2023-06-1', 'end_date': '2023-12-31',
        'ideal_trade_frequency_hours_min': (24*2),
        'ideal_trade_frequency_hours_max': (24*2),
        'init_points': 800, 'iter_points': 2000,
        'macd_fast_min': 3, 'macd_fast_max': 25,
        'macd_slow_min': 12, 'macd_slow_max': 50,
        'macd_signal_min': 2, 'macd_signal_max': 30,
        'macd_persist_min': 1, 'macd_persist_max': 10,
        'bb_p_min': 3, 'bb_p_max': 40,
        'bb_dev_low_min': 0.25, 'bb_dev_low_max': 3.0,
        'bb_dev_up_min': 0.25, 'bb_dev_up_max': 2.5,
        'rsi_period_min': 3, 'rsi_period_max': 25,
        'rsi_threshold_buy_min': 20, 'rsi_threshold_buy_max': 80,
        'rsi_threshold_sell_min': 20, 'rsi_threshold_sell_max': 80,
        'obv_p_min': 2, 'obv_p_max': 30,
        'obv_persist_min': 1, 'obv_persist_max': 10,
        'stop_loss_pct_min': 1.5, 'stop_loss_pct_max': 1.5
    }),

    # ("LTCXBT_15min_Kraken.csv", {
    #     'start_date': '2023-06-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (24*2),
    #     'ideal_trade_frequency_hours_max': (24*2),
    #     'init_points': 800, 'iter_points': 2000,
    #     'macd_fast_min': 3, 'macd_fast_max': 25,
    #     'macd_slow_min': 12, 'macd_slow_max': 50,
    #     'macd_signal_min': 2, 'macd_signal_max': 30,
    #     'macd_persist_min': 1, 'macd_persist_max': 10,
    #     'bb_p_min': 3, 'bb_p_max': 40,
    #     'bb_dev_low_min': 0.25, 'bb_dev_low_max': 3.0,
    #     'bb_dev_up_min': 0.25, 'bb_dev_up_max': 2.5,
    #     'rsi_period_min': 3, 'rsi_period_max': 25,
    #     'rsi_threshold_buy_min': 20, 'rsi_threshold_buy_max': 80,
    #     'rsi_threshold_sell_min': 20, 'rsi_threshold_sell_max': 80,
    #     'obv_p_min': 2, 'obv_p_max': 30,
    #     'obv_persist_min': 1, 'obv_persist_max': 10,
    #     'stop_loss_pct_min': 1.5, 'stop_loss_pct_max': 1.5
    # }),

    # ("XRPXBT_15min_Kraken.csv", {
    #     'start_date': '2023-06-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (24*2),
    #     'ideal_trade_frequency_hours_max': (24*2),
    #     'init_points': 800, 'iter_points': 2000,
    #     'macd_fast_min': 3, 'macd_fast_max': 25,
    #     'macd_slow_min': 12, 'macd_slow_max': 50,
    #     'macd_signal_min': 2, 'macd_signal_max': 30,
    #     'macd_persist_min': 1, 'macd_persist_max': 10,
    #     'bb_p_min': 3, 'bb_p_max': 40,
    #     'bb_dev_low_min': 0.25, 'bb_dev_low_max': 3.0,
    #     'bb_dev_up_min': 0.25, 'bb_dev_up_max': 2.5,
    #     'rsi_period_min': 3, 'rsi_period_max': 25,
    #     'rsi_threshold_buy_min': 20, 'rsi_threshold_buy_max': 80,
    #     'rsi_threshold_sell_min': 20, 'rsi_threshold_sell_max': 80,
    #     'obv_p_min': 2, 'obv_p_max': 30,
    #     'obv_persist_min': 1, 'obv_persist_max': 10,
    #     'stop_loss_pct_min': 1.5, 'stop_loss_pct_max': 1.5
    # }),

    # ("XRPXBT_15min_Kraken.csv", {
    #     'start_date': '2023-09-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (12),
    #     'ideal_trade_frequency_hours_max': (24*7),
    #     'init_points': 50000, 'iter_points': 0,
    #     'stoch_k_period_min': 10, 'stoch_k_period_max': 30,
    #     'stoch_d_period_min': 3, 'stoch_d_period_max': 10,
    #     'stoch_slowing_min': 1, 'stoch_slowing_max': 8,
    #     'stoch_threshold_min': 20, 'stoch_threshold_max': 90,
    #     'bb_p_min': 10, 'bb_p_max': 40,
    #     'bb_dev_low_min': 0.5, 'bb_dev_low_max': 3.0,
    #     'bb_dev_up_min': 0.5, 'bb_dev_up_max': 3.0,
    #     'cci_p_min': 5, 'cci_p_max': 100,
    #     'obv_p_min': 3, 'obv_p_max': 30,
    #     'obv_persist_min': 1, 'obv_persist_max': 10,
    #     'stop_loss_pct_min': 0.5, 'stop_loss_pct_max': 3
    # }),

    # ("ETHXBT_30min_Kraken.csv", {
    #     'start_date': '2023-09-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (12),
    #     'ideal_trade_frequency_hours_max': (24*7),
    #     'init_points': 50000, 'iter_points': 0,
    #     'stoch_k_period_min': 10, 'stoch_k_period_max': 30,
    #     'stoch_d_period_min': 3, 'stoch_d_period_max': 10,
    #     'stoch_slowing_min': 1, 'stoch_slowing_max': 8,
    #     'stoch_threshold_min': 20, 'stoch_threshold_max': 90,
    #     'bb_p_min': 10, 'bb_p_max': 40,
    #     'bb_dev_low_min': 0.5, 'bb_dev_low_max': 3.0,
    #     'bb_dev_up_min': 0.5, 'bb_dev_up_max': 3.0,
    #     'cci_p_min': 5, 'cci_p_max': 100,
    #     'obv_p_min': 3, 'obv_p_max': 30,
    #     'obv_persist_min': 1, 'obv_persist_max': 10,
    #     'stop_loss_pct_min': 0.5, 'stop_loss_pct_max': 3
    # }),

    # ("LTCXBT_30min_Kraken.csv", {
    #     'start_date': '2023-09-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (12),
    #     'ideal_trade_frequency_hours_max': (24*7),
    #     'init_points': 50000, 'iter_points': 0,
    #     'stoch_k_period_min': 10, 'stoch_k_period_max': 30,
    #     'stoch_d_period_min': 3, 'stoch_d_period_max': 10,
    #     'stoch_slowing_min': 1, 'stoch_slowing_max': 8,
    #     'stoch_threshold_min': 20, 'stoch_threshold_max': 90,
    #     'bb_p_min': 10, 'bb_p_max': 40,
    #     'bb_dev_low_min': 0.5, 'bb_dev_low_max': 3.0,
    #     'bb_dev_up_min': 0.5, 'bb_dev_up_max': 3.0,
    #     'cci_p_min': 5, 'cci_p_max': 100,
    #     'obv_p_min': 3, 'obv_p_max': 30,
    #     'obv_persist_min': 1, 'obv_persist_max': 10,
    #     'stop_loss_pct_min': 0.5, 'stop_loss_pct_max': 3
    # }),

    # ("XDGXBT_30min_Kraken.csv", {
    #     'start_date': '2023-09-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (12),
    #     'ideal_trade_frequency_hours_max': (24*7),
    #     'init_points': 50000, 'iter_points': 0,
    #     'stoch_k_period_min': 10, 'stoch_k_period_max': 30,
    #     'stoch_d_period_min': 3, 'stoch_d_period_max': 10,
    #     'stoch_slowing_min': 1, 'stoch_slowing_max': 8,
    #     'stoch_threshold_min': 20, 'stoch_threshold_max': 90,
    #     'bb_p_min': 10, 'bb_p_max': 40,
    #     'bb_dev_low_min': 0.5, 'bb_dev_low_max': 3.0,
    #     'bb_dev_up_min': 0.5, 'bb_dev_up_max': 3.0,
    #     'cci_p_min': 5, 'cci_p_max': 100,
    #     'obv_p_min': 3, 'obv_p_max': 30,
    #     'obv_persist_min': 1, 'obv_persist_max': 10,
    #     'stop_loss_pct_min': 0.5, 'stop_loss_pct_max': 3
    # }),

    # ("XRPXBT_30min_Kraken.csv", {
    #     'start_date': '2023-09-1', 'end_date': '2023-12-31',
    #     'ideal_trade_frequency_hours_min': (12),
    #     'ideal_trade_frequency_hours_max': (24*7),
    #     'init_points': 50000, 'iter_points': 0,
    #     'stoch_k_period_min': 10, 'stoch_k_period_max': 30,
    #     'stoch_d_period_min': 3, 'stoch_d_period_max': 10,
    #     'stoch_slowing_min': 1, 'stoch_slowing_max': 8,
    #     'stoch_threshold_min': 20, 'stoch_threshold_max': 90,
    #     'bb_p_min': 10, 'bb_p_max': 40,
    #     'bb_dev_low_min': 0.5, 'bb_dev_low_max': 3.0,
    #     'bb_dev_up_min': 0.5, 'bb_dev_up_max': 3.0,
    #     'cci_p_min': 5, 'cci_p_max': 100,
    #     'obv_p_min': 3, 'obv_p_max': 30,
    #     'obv_persist_min': 1, 'obv_persist_max': 10,
    #     'stop_loss_pct_min': 0.5, 'stop_loss_pct_max': 3
    # }),
]

num_cpu_cores = multiprocessing.cpu_count()
n_jobs = max(1, num_cpu_cores // len(configs))


for filename, inputs in configs:
    command = ' '.join([
        python_version, 'optimizer.py',
        '--filename', filename,
        '--macd_fast_min', str(inputs['macd_fast_min']),
        '--macd_fast_max', str(inputs['macd_fast_max']),
        '--macd_slow_min', str(inputs['macd_slow_min']),
        '--macd_slow_max', str(inputs['macd_slow_max']),
        '--macd_signal_min', str(inputs['macd_signal_min']),
        '--macd_signal_max', str(inputs['macd_signal_max']),
        '--macd_persist_min', str(inputs['macd_persist_min']),
        '--macd_persist_max', str(inputs['macd_persist_max']),
        '--bb_period_min', str(inputs['bb_p_min']),
        '--bb_period_max', str(inputs['bb_p_max']),
        '--bb_dev_low_min', str(inputs['bb_dev_low_min']),
        '--bb_dev_low_max', str(inputs['bb_dev_low_max']),
        '--bb_dev_up_min', str(inputs['bb_dev_up_min']),
        '--bb_dev_up_max', str(inputs['bb_dev_up_max']),
        '--obv_p_min', str(inputs['obv_p_min']),
        '--obv_p_max', str(inputs['obv_p_max']),
        '--obv_persist_min', str(inputs['obv_persist_min']),
        '--obv_persist_max', str(inputs['obv_persist_max']),
        '--rsi_period_min', str(inputs['rsi_period_min']),
        '--rsi_period_max', str(inputs['rsi_period_max']),
        '--rsi_threshold_buy_min', str(inputs['rsi_threshold_buy_min']),
        '--rsi_threshold_buy_max', str(inputs['rsi_threshold_buy_max']),
        '--rsi_threshold_sell_min', str(inputs['rsi_threshold_sell_min']),
        '--rsi_threshold_sell_max', str(inputs['rsi_threshold_sell_max']),
        '--stop_loss_pct_min', str(inputs['stop_loss_pct_min']),
        '--stop_loss_pct_max', str(inputs['stop_loss_pct_max']),
        '--number_of_cores', str(n_jobs),
        '--start_date', str(inputs['start_date']),
        '--end_date', str(inputs['end_date']),
        '--ideal_trade_frequency_hours_min', str(inputs['ideal_trade_frequency_hours_min']),
        '--ideal_trade_frequency_hours_max', str(inputs['ideal_trade_frequency_hours_max']),
        '--init_points', str(inputs['init_points']),
        '--iter_points', str(inputs['iter_points']),
    ])

    if os_name == "Linux" or os_name == "Darwin":
        subprocess.Popen(f'gnome-terminal -- bash -c "{command}; exec bash"', shell=True)
    elif os_name == "Windows":
        subprocess.Popen(f'cmd /c start cmd /k "{command}"', shell=True)
    else:
        print(f"Unsupported operating system: {os_name}")
