import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import signal
import os
import pywt
from datetime import datetime
from bayes_opt import BayesianOptimization, UtilityFunction
from joblib import Parallel, delayed
from math import ceil
from tqdm import tqdm
from pyDOE import lhs


MAX_HOLD_TIME = (60*24)*3  # in minutes
INTERRUPTED = False


def signal_handler(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True
    print("Signal received, stopping optimization...")
    raise KeyboardInterrupt


class SignalOptimizer:
    def __init__(self, filepath, pbounds, number_of_cores, start_date, end_date, init_points, iter_points, pair_points):
        self.PBOUNDS = pbounds
        self.filepath = filepath
        self.number_of_cores = number_of_cores
        self.start_date = start_date
        self.end_date = end_date
        self.init_points = init_points
        self.iter_points = iter_points
        self.pair_points = pair_points
        self.dataset_name = os.path.basename(self.filepath).split('.')[0]

        self.plot_subfolder = os.path.join('indicator_optimizer_plots', self.dataset_name)
        os.makedirs(self.plot_subfolder, exist_ok=True)
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.load_and_prepare_data()

        self.best_buy_points = []
        self.best_sell_points = []
        self.best_performance = float('-inf')
        self.param_to_results = {}
        self.total_percent_gain = 0

        self.min_desired_trade_frequency_days = 7

    def load_and_prepare_data(self):
        processed_data_path = os.path.join('Processed Data', self.filepath)

        self.data = pd.read_csv(
            processed_data_path,
            usecols=['time', 'close', 'high', 'low', 'volume'],
            parse_dates=['time'],
            index_col='time',
            dtype={'close': 'float32', 'high': 'float32', 'low': 'float32', 'volume': 'float32'}
        )

        available_start_date = self.data.index.min()
        available_end_date = self.data.index.max()
        start_date = max(pd.Timestamp(self.start_date), available_start_date)
        end_date = min(pd.Timestamp(self.end_date), available_end_date)

        if start_date > end_date:
            raise ValueError("Adjusted start date is after the adjusted end date. No data available for the given range.")
        if start_date != pd.Timestamp(self.start_date) or end_date != pd.Timestamp(self.end_date):
            print(f"Date range adjusted based on available data: {start_date.date()} to {end_date.date()}")
        self.data = self.data.loc[start_date:end_date]
        if self.data.empty:
            raise ValueError(f"No data available between {start_date} and {end_date} after adjustments.")

        match = re.search(r"(\d+)min", self.filepath)
        self.data_frequency_in_minutes = int(match.group(1)) if match else None

        self.data.ffill(inplace=True)
        self.data.reset_index(inplace=True)

        self.data['close_init'] = self.data['close'].copy()
        self.data['close_transformed'] = self.apply_wavelet_transform(self.data['close'].values)
        self.data['close'] = self.data['close_transformed'].copy()

    def apply_wavelet_transform(self, data, wavelet='db1', mode='smooth', level=1):
        coeffs = pywt.wavedec(data, wavelet, mode=mode, level=level)
        coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]  # Keep only approximation coefficients
        reconstructed_signal = pywt.waverec(coeffs, wavelet, mode=mode)

        # Ensure the reconstructed signal is the same length as the input data
        reconstructed_signal = reconstructed_signal[:len(data)]

        return pd.Series(reconstructed_signal, index=self.data.index[:len(reconstructed_signal)])  # Adjust the index

    def calculate_rsi(self, rsi_period, rsi_threshold):
        rsi_period = int(rsi_period)
        delta = self.data['close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi, rsi_threshold

    def calculate_ema(self, fast_period, slow_period):
        fast_ema = self.data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.data['close'].ewm(span=slow_period, adjust=False).mean()
        return fast_ema, slow_ema

    def calculate_stochastic_oscillator(self, k_period, slow_k_period, slow_d_period):
        k_period = int(k_period)
        slow_k_period = int(slow_k_period)
        slow_d_period = int(slow_d_period)

        low_min = self.data['low'].rolling(window=k_period).min()
        high_max = self.data['high'].rolling(window=k_period).max()
        fast_k = ((self.data['close'] - low_min) / (high_max - low_min)) * 100

        slow_k = fast_k.rolling(window=slow_k_period).mean()
        slow_d = slow_k.rolling(window=slow_d_period).mean()
        stoch_avg = (slow_k + slow_d) / 2
        return stoch_avg

    def calculate_obv(self, ema_period):
        ema_period = int(ema_period)
        obv = np.where(self.data['close'] > self.data['close'].shift(), self.data['volume'], -self.data['volume']).cumsum()
        obv_ema = pd.Series(obv).ewm(span=ema_period, adjust=False).mean()
        return obv_ema

    def evaluate_performance(self, rsi_p, rsi_thr, short_ema_p, long_ema_p, ema_persistence, stoch_k_p, stoch_slow_k_p, stoch_slow_d_p, stoch_thr, obv_ema_p, arm_pct, arm_stp_ls_pct, stp_ls_pct):
        rsi, _ = self.calculate_rsi(rsi_p, rsi_thr)
        fast_ema, slow_ema = self.calculate_ema(short_ema_p, long_ema_p)
        stoch_avg = self.calculate_stochastic_oscillator(stoch_k_p, stoch_slow_k_p, stoch_slow_d_p)
        obv_ema_values = self.calculate_obv(obv_ema_p)

        initial_balance = 1.0
        current_balance = initial_balance

        position_open = False
        entry_price = 0.0
        highest_price_after_buy = 0.0
        armed = False
        entry_index = -1
        buy_points = []
        sell_points = []
        returns = []

        ema_signal_active = False
        ema_signal_counter = 0

        for i in range(1, len(self.data)):
            current_price = self.data['close'].iloc[i]
            signals_met = 0

            if rsi.iloc[i] < rsi_thr:
                signals_met += 1
            if fast_ema.iloc[i] > slow_ema.iloc[i] and fast_ema.iloc[i-1] <= slow_ema.iloc[i-1] and short_ema_p < long_ema_p:
                signals_met += 1
                ema_signal_active = True
                ema_signal_counter = 0
            elif ema_signal_active:
                ema_signal_counter += 1
                signals_met += 1
                if ema_signal_counter > ema_persistence:
                    ema_signal_active = False
            if stoch_avg.iloc[i] <= stoch_thr:
                signals_met += 1
            if obv_ema_values.iloc[i] > obv_ema_values.iloc[i-1]:
                signals_met += 1

            # Logic for opening position
            if signals_met >= 3 and not position_open:
                buy_points.append(self.data.index[i])
                position_open = True
                entry_price = current_price
                highest_price_after_buy = current_price
                entry_index = i

            # Logic for closing position
            if position_open:
                holding_time = i - entry_index
                max_holding_time_reached = holding_time >= (MAX_HOLD_TIME / self.data_frequency_in_minutes)

                # Stop loss
                if current_price <= entry_price * (1 - stp_ls_pct / 100):
                    sell_points.append(self.data.index[i])
                    trade_return = (current_price - entry_price) / entry_price
                    capital_at_risk = current_balance * 0.10
                    profit_from_trade = capital_at_risk * trade_return
                    current_balance += profit_from_trade
                    position_open = False
                    armed = False
                    entry_index = -1
                    returns.append(current_price - entry_price)

                # Arming stop loss and max holding time
                if not armed and current_price >= entry_price * (1 + arm_pct / 100):
                    armed = True

                if armed and current_price <= highest_price_after_buy * (1 - arm_stp_ls_pct / 100) or max_holding_time_reached:
                    sell_points.append(self.data.index[i])
                    trade_return = (current_price - entry_price) / entry_price
                    capital_at_risk = current_balance * 0.10
                    profit_from_trade = capital_at_risk * trade_return
                    current_balance += profit_from_trade
                    position_open = False
                    armed = False
                    entry_index = -1
                    returns.append(current_price - entry_price)

        objective_function = 0
        profit_ratio = 0
        total_percent_gain = (current_balance - initial_balance) / initial_balance * 100
        total_num_trades = len(returns)
        returns = np.array(returns)

        # Build objective function
        if total_num_trades > 0:
            profitable_trades = np.sum(returns > 0)
            variance_of_returns = np.var(returns)

            length_of_data_days = ((len(self.data) * self.data_frequency_in_minutes) / (60*24))
            minimum_trades_required = length_of_data_days / self.min_desired_trade_frequency_days

            profit_ratio = profitable_trades / total_num_trades
            min_target_profit_ratio = 0.9

            pr_weight = 0
            var_weight = 0 * 0.001 * 1E8
            pg_weight = 1
            total_trade_penalty_weight = 1
            profit_ratio_penalty_weight = 1

            profit_ratio_factor = pr_weight * profit_ratio
            percent_gain_factor = pg_weight * total_percent_gain
            variance_factor = var_weight * variance_of_returns

            total_num_trades_penalty = 0
            profit_ratio_penalty = 0

            # Penalty if minimum number of trades is not met
            if total_num_trades < minimum_trades_required:
                difference_ratio = (minimum_trades_required - total_num_trades) / total_num_trades
                total_num_trades_penalty = total_trade_penalty_weight * min(np.exp(difference_ratio/5), 1E9)

            # Penalty if profit ratio is not met
            if profit_ratio < min_target_profit_ratio:
                diff_from_target = min_target_profit_ratio - profit_ratio
                profit_ratio_penalty = profit_ratio_penalty_weight * min(np.exp(diff_from_target), 1E9)  # diff_from_target*profit_ratio_penalty_weight  # 

            objective_function = (profit_ratio_factor + percent_gain_factor) - variance_factor - total_num_trades_penalty - profit_ratio_penalty

            print(f"OF:{objective_function:8.2f}, PR,PG:{profit_ratio_factor:6.2f},{percent_gain_factor:6.2f}, VarP,#TrdP,PRP:{variance_factor:7.2f},{total_num_trades_penalty:7.2f},{profit_ratio_penalty:7.2f}")
        else:
            objective_function = -1E9
            print("No Trades!")
        return objective_function, buy_points, sell_points, total_percent_gain, profit_ratio

    def evaluate_wrapper(self, params):
        performance, buy_points, sell_points, _, _,  = self.evaluate_performance(**params)
        self.param_to_results[str(params)] = (buy_points, sell_points)
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
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        init_points = self.generate_lhs_samples(self.PBOUNDS, self.init_points)

        batch_size = self.number_of_cores
        num_batches = ceil(len(init_points) / batch_size)

        try:
            with tqdm(total=self.init_points, desc=f"Init {self.filepath}") as pbar:
                for batch_idx in range(num_batches):
                    if INTERRUPTED:
                        print("Ctrl+C, breaking out of initial processing.")
                        break  # Exit the loop if an interruption was signaled
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(init_points))
                    batch_points = init_points[start_idx:end_idx]

                    # Process each batch in parallel
                    batch_results = Parallel(n_jobs=self.number_of_cores, backend="multiprocessing")(
                        delayed(self.evaluate_wrapper)(params) for params in batch_points
                    )

                    # Register the results of each batch
                    for idx, performance in enumerate(batch_results):
                        optimizer.register(params=batch_points[idx], target=performance)

                    # Update tqdm with the number of points processed in this batch
                    pbar.update(len(batch_points))

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Proceeding with results obtained so far.")
        try:
            # Sequential Bayesian optimization
            pbar_counter = 0
            top_performance = 0
            with tqdm(total=self.iter_points, desc=f"Optimizing {self.filepath}") as pbar:
                for _ in range(self.iter_points):
                    if INTERRUPTED:
                        print("Ctrl+C, breaking out of initial processing.")
                        break  # Exit the loop if an interruption was signaled
                    next_params = optimizer.suggest(utility)
                    performance, buy_points, sell_points, total_percent_gain, profit_ratio = self.evaluate_performance(**next_params)
                    optimizer.register(params=next_params, target=performance)
                    pbar_counter += 1
                    if pbar_counter % 100 == 0:
                        pbar.update(100)
                    if performance > top_performance:
                        top_performance = performance
                        print(f"New top performance: {performance:.2f}, Profit Ratio: {profit_ratio}, Total percent gain: {total_percent_gain:.2f}")
                        #  self.plot_trades(buy_points, sell_points)

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Proceeding with results obtained so far.")

        # Process and return the results obtained so far, regardless of whether there was an interruption
        sorted_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)
        self.top_results = sorted_results[:self.pair_points] if len(sorted_results) >= self.pair_points else sorted_results
        return self.top_results

    def visualize_top_results(self):
        if not self.top_results:
            print("No results to visualize.")
            return

        results_df = pd.DataFrame([res['params'] for res in self.top_results])
        results_df['profit'] = [res['target'] for res in self.top_results]
        filtered_results_df = results_df[results_df['profit'] != 0]

        if filtered_results_df.empty:
            print("No results to visualize after filtering out zero profits.")
            return

        pairplot = sns.pairplot(filtered_results_df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height=2)
        pairplot.figure.suptitle(f'{self.dataset_name}', size=12)

        filename = f"{self.dataset_name}_PairPlot_{self.start_date}_to_{self.end_date}_Date_{self.time_now}.png"
        plt.savefig(os.path.join(self.plot_subfolder, filename))
        plt.show()

    def plot_trades(self, buy_points, sell_points):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['time'], self.data['close_init'], label='Close Price', alpha=0.3, color='#7f7f7f')
        plt.plot(self.data['time'], self.data['close_transformed'], label='Wavelet Transformed Close Price', alpha=0.5, linestyle='--', color='#1f77b4')

        plt.scatter(self.data.loc[self.data.index.isin(buy_points), 'time'], self.data.loc[self.data.index.isin(buy_points), 'close'], label='Buy', marker='^', color='green', alpha=1)
        plt.scatter(self.data.loc[self.data.index.isin(sell_points), 'time'], self.data.loc[self.data.index.isin(sell_points), 'close'], label='Sell', marker='v', color='red', alpha=1)

        plt.title(f'{self.dataset_name}_{self.start_date}_to_{self.end_date}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()

        filename = f"{self.dataset_name}_BUYSELLResults_{self.start_date}_to_{self.end_date}_Date_{self.time_now}.png"
        plt.savefig(os.path.join(self.plot_subfolder, filename))
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Signal Optimizer")
    # RSI Parameters
    parser.add_argument("--rsi_period_min", type=int, default=5, help="Minimum RSI period")
    parser.add_argument("--rsi_period_max", type=int, default=14, help="Maximum RSI period")
    parser.add_argument("--rsi_threshold_min", type=int, default=20, help="Minimum RSI threshold for oversold condition")
    parser.add_argument("--rsi_threshold_max", type=int, default=30, help="Maximum RSI threshold for oversold condition")
    # Stochastic Oscillator Parameters
    parser.add_argument("--stoch_k_period_min", type=int, default=5, help="Minimum Stochastic %K period")
    parser.add_argument("--stoch_k_period_max", type=int, default=14, help="Maximum Stochastic %K period")
    parser.add_argument("--stoch_d_period_min", type=int, default=3, help="Minimum Stochastic %D period")
    parser.add_argument("--stoch_d_period_max", type=int, default=5, help="Maximum Stochastic %D period")
    parser.add_argument("--stoch_slowing_min", type=int, default=3, help="Minimum Stochastic slowing period")
    parser.add_argument("--stoch_slowing_max", type=int, default=5, help="Maximum Stochastic slowing period")
    # EMA Parameters for potentially using with OBV or price trends
    parser.add_argument("--ema_short_period_min", type=int, default=12, help="Minimum short EMA period")
    parser.add_argument("--ema_short_period_max", type=int, default=26, help="Maximum short EMA period")
    parser.add_argument("--ema_long_period_min", type=int, default=26, help="Minimum long EMA period")
    parser.add_argument("--ema_long_period_max", type=int, default=50, help="Maximum long EMA period")
    parser.add_argument("--ema_persistence_min", type=int, default=1, help="Minimum EMA persistence")
    parser.add_argument("--ema_persistence_max", type=int, default=4, help="Maximum EMA persistence")
    # OBV EMA Parameters
    parser.add_argument("--obv_ema_period_min", type=int, default=3, help="Minimum OBV EMA period")
    parser.add_argument("--obv_ema_period_max", type=int, default=10, help="Maximum OBV EMA period")
    # Sell Parameters
    parser.add_argument("--arming_pct_min", type=float, default=0.6, help="Minimum arming percentage for stop loss")
    parser.add_argument("--arming_pct_max", type=float, default=1.5, help="Maximum arming percentage for stop loss")
    parser.add_argument("--arm_stop_loss_pct_min", type=float, default=0.1, help="Minimum stop loss percentage after arming")
    parser.add_argument("--arm_stop_loss_pct_max", type=float, default=0.3, help="Maximum stop loss percentage after arming")
    parser.add_argument("--stop_loss_pct_min", type=float, default=1, help="Minimum stop loss percentage")
    parser.add_argument("--stop_loss_pct_max", type=float, default=2, help="Maximum stop loss percentage")

    # Inputs
    parser.add_argument("--filename", type=str, required=True, help="Input CSV file name")
    parser.add_argument("--number_of_cores", type=int, default=1, help="Number of CPU cores to use during initial search")
    parser.add_argument("--start_date", type=str, default='2023-10-30', help="Start date for optimization")
    parser.add_argument("--end_date", type=str, default='2023-11-29', help="End date for optimization")
    parser.add_argument("--init_points", type=int, default=100, help="Number of initial points to search")
    parser.add_argument("--iter_points", type=int, default=100, help="Number of optimization itterations")
    parser.add_argument("--pair_points", type=int, default=100, help="Number of points to show in the pair graph")
    return parser.parse_args()


def run_optimization(filename, pbounds, number_of_cores, start_date, end_date, init_points, iter_points, pair_points):
    optimizer = SignalOptimizer(filename, pbounds, number_of_cores, start_date, end_date, init_points, iter_points, pair_points)
    top_results = optimizer.optimize()

    if top_results:
        best_params = top_results[0]['params']
        final_performance, final_buy_points, final_sell_points, total_percent_gain, profit_ratio = optimizer.evaluate_performance(**best_params)

        print(f"Optimized Indicator Parameters: {best_params}")
        print(f"Best Performance: {final_performance}")
        print(f"Profit Ratio: {profit_ratio}")
        print(f"Percent Gain: {total_percent_gain}")

        optimizer.visualize_top_results()
        optimizer.plot_trades(final_buy_points, final_sell_points)


if __name__ == "__main__":
    args = parse_args()
    PBOUNDS = {
        'rsi_p': (args.rsi_period_min, args.rsi_period_max),
        'rsi_thr': (args.rsi_threshold_min, args.rsi_threshold_max),
        'short_ema_p': (args.ema_short_period_min, args.ema_short_period_max),
        'long_ema_p': (args.ema_long_period_min, args.ema_long_period_max),
        'ema_persistence': (args.ema_persistence_min, args.ema_persistence_max),
        'stoch_k_p': (args.stoch_k_period_min, args.stoch_k_period_max),
        'stoch_slow_k_p': (args.stoch_slowing_min, args.stoch_slowing_max),
        'stoch_slow_d_p': (args.stoch_d_period_min, args.stoch_d_period_max),
        'stoch_thr': (args.stoch_k_period_min, args.stoch_k_period_max),
        'obv_ema_p': (args.obv_ema_period_min, args.obv_ema_period_max),
        'arm_pct': (args.arming_pct_min, args.arming_pct_max),
        'arm_stp_ls_pct': (args.arm_stop_loss_pct_min, args.arm_stop_loss_pct_max),
        'stp_ls_pct': (args.stop_loss_pct_min, args.stop_loss_pct_max)
    }

    run_optimization(args.filename, PBOUNDS, args.number_of_cores, args.start_date, args.end_date, args.init_points, args.iter_points, args.pair_points)
