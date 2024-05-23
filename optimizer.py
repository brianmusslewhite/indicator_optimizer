import argparse
import math
import numpy as np
import re
import signal
import os
from datetime import datetime
from bayes_opt import BayesianOptimization, UtilityFunction
from joblib import Parallel, delayed
from math import ceil
from tqdm import tqdm
from pyDOE import lhs

from load_data import DataLoader
from indicators import calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_cci, calculate_obv
from plot_data import visualize_top_results, plot_trades, plot_parameter_sensitivity

MINUTES_IN_DAY = 1440
MAX_HOLD_TIME_MINUTES = MINUTES_IN_DAY * 2
INTERRUPTED = False
MIN_DESIRED_TRADE_FREQUENCY_DAYS = 1


def signal_handler(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True
    print("Signal received, stopping optimization...")
    raise KeyboardInterrupt


class SignalOptimizer:
    def __init__(self, filepath, pbounds, number_of_cores, start_date, end_date, init_points, iter_points):
        # Basic properties
        self.filepath = filepath
        self.PBOUNDS = pbounds
        self.start_date = start_date
        self.end_date = end_date
        self.number_of_cores = number_of_cores
        self.init_points = init_points
        self.iter_points = iter_points

        # Extract dataset name and data frequency from filepath
        self.dataset_name = os.path.basename(self.filepath).split('.')[0]
        match = re.search(r"(\d+)min", self.filepath)
        self.data_frequency_in_minutes = int(match.group(1)) if match else None

        # Data loading
        self.data_loader = DataLoader(filepath, start_date, end_date)
        self.data = self.data_loader.data

        # Directory for output plots
        self.plot_subfolder = os.path.join('indicator_optimizer_plots', self.dataset_name)
        os.makedirs(self.plot_subfolder, exist_ok=True)

        # Timestamp for this operation
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Operational parameters
        self.min_desired_trade_frequency_days = MIN_DESIRED_TRADE_FREQUENCY_DAYS
        self.data_duration_hours = ((self.data.index.max() - self.data.index.min()) * self.data_frequency_in_minutes) / 60
        self.bad_result_number = -500
        self.max_penalty = 150

        # Results storage
        self.best_buy_points = []
        self.best_sell_points = []
        self.best_performance = float('-inf')
        self.param_to_results = {}
        self.total_percent_gain = 0

    def evaluate_performance(self, bb_p, bb_dev_low, bb_dev_up, macd_fast_p, macd_slow_p, macd_sig_p, obv_p, obv_persist, rsi_p, stp_ls_pct, idl_trd_frq_hrs):
        upper_band, lower_band = calculate_bollinger_bands(self.data, int(bb_p), bb_dev_low, bb_dev_up)
        macd, signal_line = calculate_macd(macd_fast_p, macd_slow_p, int(macd_sig_p))
        obv = calculate_obv(self.data, int(obv_p))
        rsi = calculate_rsi(self.data, rsi_p)
        min_trades = int(np.ceil(self.data_duration_hours / idl_trd_frq_hrs))

        initial_balance = 1.0
        current_balance = initial_balance

        position_open = False
        entry_price = 0.0
        buy_points = []
        sell_points = []
        trade_results = []
        min_prifit_ratio = 0.8

        obv_signal_active = False
        obv_signal_counter = 0

        for i in range(1, len(self.data)):
            current_price = self.data['close'].iloc[i]
            signals_met = 0

            if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
                signals_met += 1
            if current_price < lower_band.iloc[i]:
                signals_met += 1
            
            if obv.iloc[i] > obv.iloc[i-1]:
                signals_met += 1
                obv_signal_active = True
                obv_signal_counter = 0
            elif obv_signal_active:
                obv_signal_counter += 1
                signals_met += 1
                if obv_signal_counter >= obv_persist:
                    obv_signal_active = False

            # Logic for opening position
            if signals_met >= 3 and not position_open:
                buy_points.append(self.data.index[i])
                position_open = True
                entry_price = current_price

            # Logic for closing position
            if position_open:
                # Stop loss or Take profit
                if current_price <= entry_price * (1 - stp_ls_pct / 100) or current_price > upper_band.iloc[i]:
                    sell_points.append(self.data.index[i])
                    trade_return = (current_price - entry_price) / entry_price
                    trade_results.append(trade_return)
                    current_balance *= (1 + trade_return)
                    position_open = False

        total_percent_gain = (current_balance - initial_balance) / initial_balance * 100
        performance = total_percent_gain
        total_trades = len(trade_results)
        profitable_trades = sum(1 for result in trade_results if result > 0)
        profit_ratio = profitable_trades / total_trades if total_trades > 0 else 0

        # Penalty for trade frequency
        if total_trades < min_trades:
            trade_deficit_normal = (min_trades - total_trades) / min_trades
            trade_penalty = 2*math.exp(trade_deficit_normal)
            # print(f"Trade Penalty: {trade_penalty}")
            performance -= trade_penalty

        # Penalty for profit ratio
        if profit_ratio < min_prifit_ratio:
            pr_deficit_normal = (min_prifit_ratio - profit_ratio) / min_prifit_ratio
            pr_penalty = 2*math.exp(pr_deficit_normal)
            # print(f"Profit Ratio Penalty: {pr_penalty}")
            performance -= pr_penalty

        return performance, buy_points, sell_points, total_percent_gain, profit_ratio

    def evaluate_wrapper(self, params):
        performance, _, _, total_percent_gain, profit_ratio = self.evaluate_performance(**params)
        print(f"Performance: {performance:8.2f}, Profit Ratio: {profit_ratio:8.2f}, Total Percent Gain: {total_percent_gain:8.2f}")
        return performance

    def generate_lhs_samples(self, pbounds, num_samples):
        num_params = len(pbounds)
        # Generate LHS samples within [0, 1]
        lhs_samples = lhs(num_params, samples=num_samples)

        samples = []
        for i in range(num_samples):
            sample = {}
            for param_idx, (key, (min_val, max_val)) in enumerate(pbounds.items()):
                # Scale each sample to the parameter's range
                sample[key] = min_val + (max_val - min_val) * lhs_samples[i, param_idx]
            samples.append(sample)

        return samples

    def optimize(self):
        global INTERRUPTED
        signal.signal(signal.SIGINT, signal_handler)
        optimizer = BayesianOptimization(
            f=None,
            pbounds=self.PBOUNDS,
            random_state=1,
            verbose=2,
            allow_duplicate_points=True
        )
        init_points = self.generate_lhs_samples(self.PBOUNDS, self.init_points)

        batch_size = self.number_of_cores
        num_batches = ceil(len(init_points) / batch_size)

        try:
            with tqdm(total=self.init_points, desc=f"Init {self.filepath}") as pbar:
                for batch_idx in range(num_batches):
                    if INTERRUPTED:
                        print("Ctrl+C, breaking out of initial processing.")
                        break
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(init_points))
                    batch_points = init_points[start_idx:end_idx]

                    # Process each batch in parallel
                    batch_results = Parallel(n_jobs=self.number_of_cores, backend="multiprocessing")(
                        delayed(self.evaluate_wrapper)(params) for params in batch_points
                    )

                    for idx, performance in enumerate(batch_results):
                        if performance != -1E9:
                            optimizer.register(params=batch_points[idx], target=performance)

                    # Update tqdm with the number of points processed in this batch
                    pbar.update(len(batch_points))
                    print("\n")

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Proceeding with results obtained so far.")
        try:
            # Sequential Bayesian optimization
            pbar_counter = 0
            top_performance = 0
            with tqdm(total=self.iter_points, desc=f"Optimizing {self.filepath}") as pbar:
                initial_kappa = 15.0
                min_kappa = 2.576
                decay_rate_kappa = 0.1

                initial_xi = 0.1
                final_xi = 0.01
                decay_rate_xi = 0.005
                for i in range(self.iter_points):
                    if INTERRUPTED:
                        print("Ctrl+C, breaking out of initial processing.")
                        break

                    current_kappa = max(min_kappa, initial_kappa - decay_rate_kappa * i)
                    current_xi = max(final_xi, initial_xi - decay_rate_xi * i)
                    utility = UtilityFunction(kind="ucb", kappa=current_kappa, xi=current_xi)

                    next_params = optimizer.suggest(utility)
                    performance, buy_points, sell_points, total_percent_gain, profit_ratio = self.evaluate_performance(**next_params)
                    optimizer.register(params=next_params, target=performance)
                    pbar_counter += 1

                    if pbar_counter % 100 == 0:
                        pbar.update(100)
                        print()
                    if performance > top_performance:
                        top_performance = performance
                        print(f"New top performance: {performance:8.2f}, Profit Ratio: {profit_ratio:8.2f}, Total percent gain: {total_percent_gain:8.2f}")
                    else:
                        print(f"Performance: {performance:8.2f}, Profit Ratio: {profit_ratio:8.2f}, Total Percent Gain: {total_percent_gain:8.2f}")

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Proceeding with results obtained so far.")

        # Process and return the results obtained so far, regardless of whether there was an interruption
        sorted_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)
        return sorted_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run Signal Optimizer")
    # Stochastic Oscillator Parameters
    parser.add_argument("--stoch_k_period_min", type=int, default=5, help="Minimum Stochastic %K period")
    parser.add_argument("--stoch_k_period_max", type=int, default=14, help="Maximum Stochastic %K period")
    parser.add_argument("--stoch_d_period_min", type=int, default=3, help="Minimum Stochastic %D period")
    parser.add_argument("--stoch_d_period_max", type=int, default=5, help="Maximum Stochastic %D period")
    parser.add_argument("--stoch_slowing_min", type=int, default=3, help="Minimum Stochastic slowing period")
    parser.add_argument("--stoch_slowing_max", type=int, default=5, help="Maximum Stochastic slowing period")
    parser.add_argument("--stoch_threshold_min", type=int, default=10, help="Minimum Stoch threshold for oversold condition")
    parser.add_argument("--stoch_threshold_max", type=int, default=30, help="Maximum Stoch threshold for oversold condition")
    # BB Parameters
    parser.add_argument("--bb_period_min", type=int, default=5, help="Minimum Bollinger Bands period")
    parser.add_argument("--bb_period_max", type=int, default=20, help="Maximum Bollinger Bands period")
    parser.add_argument("--bb_dev_low_min", type=float, default=1.5, help="Minimum Bollinger Bands deviation")
    parser.add_argument("--bb_dev_low_max", type=float, default=2.5, help="Maximum Bollinger Bands deviation")
    parser.add_argument("--bb_dev_up_min", type=float, default=1.5, help="Minimum Bollinger Bands deviation")
    parser.add_argument("--bb_dev_up_max", type=float, default=2.5, help="Maximum Bollinger Bands deviation")
    parser.add_argument("--cci_p_min", type=int, default=10, help="Minimum cci period")
    parser.add_argument("--cci_p_max", type=int, default=30, help="Maximum cci period")
    parser.add_argument("--obv_p_min", type=int, default=1, help="Minimum obv period")
    parser.add_argument("--obv_p_max", type=int, default=10, help="Maximum obv period")
    parser.add_argument("--obv_persist_min", type=int, default=1, help="Minimum OBV persistence")
    parser.add_argument("--obv_persist_max", type=int, default=4, help="Maximum OBV persistence")
    # Sell Parameters
    parser.add_argument("--stop_loss_pct_min", type=float, default=1, help="Minimum stop loss percentage")
    parser.add_argument("--stop_loss_pct_max", type=float, default=2, help="Maximum stop loss percentage")
    # Inputs
    parser.add_argument("--filename", type=str, required=True, help="Input CSV file name")
    parser.add_argument("--number_of_cores", type=int, default=1, help="Number of CPU cores to use during initial search")
    parser.add_argument("--start_date", type=str, default='2023-10-30', help="Start date for optimization")
    parser.add_argument("--end_date", type=str, default='2023-11-29', help="End date for optimization")
    parser.add_argument("--ideal_trade_frequency_hours_min", type=float, default=24, help="Ideal trading frequency in hours min")
    parser.add_argument("--ideal_trade_frequency_hours_max", type=float, default=(24*7), help="Ideal trading frequency in hours max")
    parser.add_argument("--init_points", type=int, default=100, help="Number of initial points to search")
    parser.add_argument("--iter_points", type=int, default=100, help="Number of optimization itterations")
    parser.add_argument("--pair_points", type=int, default=100, help="Number of points to show in the pair graph")
    return parser.parse_args()


def run_optimization(filename, pbounds, number_of_cores, start_date, end_date, init_points, iter_points):
    optimizer = SignalOptimizer(filename, pbounds, number_of_cores, start_date, end_date, init_points, iter_points)
    sorted_results = optimizer.optimize()

    if sorted_results:
        best_params = sorted_results[0]['params']
        final_performance, final_buy_points, final_sell_points, total_percent_gain, profit_ratio = optimizer.evaluate_performance(**best_params)

        print(f"Optimized Indicator Parameters: {best_params}")
        print(f"Best Performance: {final_performance}")
        print(f"Profit Ratio: {profit_ratio}")
        print(f"Percent Gain: {total_percent_gain}")

        visualize_top_results(sorted_results, optimizer.dataset_name, optimizer.start_date, optimizer.end_date, optimizer.time_now, optimizer.plot_subfolder)
        plot_trades(optimizer.data, final_buy_points, final_sell_points, optimizer.dataset_name, optimizer.start_date, optimizer.end_date, optimizer.time_now, optimizer.plot_subfolder)
        plot_parameter_sensitivity(sorted_results, optimizer.dataset_name, start_date, end_date, optimizer.time_now, optimizer.plot_subfolder)

if __name__ == "__main__":
    args = parse_args()
    PBOUNDS = {
        # 'stoch_k_p': (args.stoch_k_period_min, args.stoch_k_period_max),
        # 'stoch_slow_k_p': (args.stoch_slowing_min, args.stoch_slowing_max),
        # 'stoch_slow_d_p': (args.stoch_d_period_min, args.stoch_d_period_max),
        # 'stoch_thr': (args.stoch_threshold_min, args.stoch_threshold_max),
        'bb_p': (args.bb_period_min, args.bb_period_max),
        'bb_dev_low': (args.bb_dev_low_min, args.bb_dev_low_max),
        'bb_dev_up': (args.bb_dev_up_min, args.bb_dev_up_max),
        'cci_p': (args.cci_p_min, args.cci_p_max),
        'obv_p': (args.obv_p_min, args.obv_p_max),
        'obv_persist': (args.obv_persist_min, args.obv_persist_max),
        'stp_ls_pct': (args.stop_loss_pct_min, args.stop_loss_pct_max),
        'idl_trd_frq_hrs': (args.ideal_trade_frequency_hours_min, args.ideal_trade_frequency_hours_max)
    }

    run_optimization(args.filename, PBOUNDS, args.number_of_cores, args.start_date, args.end_date, args.init_points, args.iter_points)
