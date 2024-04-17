import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import signal
import os
from datetime import datetime
from bayes_opt import BayesianOptimization


INTERRUPTED = False
def signal_handler(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True
    print("Signal received, stopping optimization...")
    raise KeyboardInterrupt


class SignalOptimizer:
    def __init__(self, filepath, pbounds, number_of_cores, start_date, end_date, init_points, iter_points, pair_points):
        self.pbounds = pbounds
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

    def calculate_bollinger_bands(self, period, deviation_lower, deviation_upper):
        period = int(period)
        sma = self.data['close'].rolling(window=period).mean()
        std = self.data['close'].rolling(window=period).std()
        upper_band = sma + (std * deviation_upper)
        lower_band = sma - (std * deviation_lower)
        return upper_band, lower_band

    def evaluate_performance(self, stoch_k_p, stoch_slow_k_p, stoch_slow_d_p, stoch_thr, bb_p, bb_dev_low, bb_dev_up, arm_pct, arm_stp_ls_pct, stp_ls_pct):
        stoch_avg = self.calculate_stochastic_oscillator(stoch_k_p, stoch_slow_k_p, stoch_slow_d_p)
        upper_band, lower_band = self.calculate_bollinger_bands(bb_p, bb_dev_low, bb_dev_up)

        initial_balance = 1.0
        current_balance = initial_balance

        position_open = False
        entry_price = 0.0
        highest_price_after_buy = 0.0
        buy_points = []
        sell_points = []
        returns = []

        for i in range(1, len(self.data)):
            current_price = self.data['close'].iloc[i]
            signals_met = 0

            if stoch_avg.iloc[i] <= stoch_thr:
                signals_met += 1
            if current_price < lower_band.iloc[i]:
                signals_met += 1

            # Logic for opening position
            if signals_met >= 2 and not position_open:
                buy_points.append(self.data.index[i])
                position_open = True
                entry_price = current_price
                highest_price_after_buy = current_price

            # Logic for closing position
            if position_open:
                # Stop loss
                if current_price <= entry_price * (1 - stp_ls_pct / 100):
                    sell_points.append(self.data.index[i])
                    trade_return = (current_price - entry_price) / entry_price
                    capital_at_risk = current_balance * 0.10
                    profit_from_trade = capital_at_risk * trade_return
                    current_balance += profit_from_trade
                    position_open = False
                    returns.append(current_price - entry_price)

                if current_price > upper_band.iloc[i]:
                    sell_points.append(self.data.index[i])
                    trade_return = (current_price - entry_price) / entry_price
                    capital_at_risk = current_balance * 0.10
                    profit_from_trade = capital_at_risk * trade_return
                    current_balance += profit_from_trade
                    position_open = False
                    returns.append(current_price - entry_price)

        total_percent_gain = (current_balance - initial_balance) / initial_balance * 100
        total_num_trades = len(returns)
        returns = np.array(returns)

        # Build objective function
        if total_num_trades > 0:
            profitable_trades = np.sum(returns > 0)
            profit_ratio = profitable_trades / total_num_trades
            objective_function = total_percent_gain
        else:
            objective_function = -1E9
            profit_ratio = 0
        return objective_function, buy_points, sell_points, total_percent_gain, profit_ratio

    def evaluate_wrapper(self, **params):
        performance, buy_points, sell_points, _, _,  = self.evaluate_performance(**params)
        self.param_to_results[str(params)] = (buy_points, sell_points)
        return performance

    def optimize(self):
        global INTERRUPTED
        signal.signal(signal.SIGINT, signal_handler)

        try:
            optimizer = BayesianOptimization(
                f=self.evaluate_wrapper,
                pbounds=self.pbounds,
                random_state=1,
                verbose=2
            )
            optimizer.maximize(init_points=self.init_points, n_iter=self.iter_points)
        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Proceeding with results obtained so far.")

        # Process and return the results obtained so far, regardless of whether there was an interruption
        sorted_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)
        self.top_results = sorted_results[:self.pair_points] if len(sorted_results) >= self.pair_points else sorted_results
        return self.top_results

    def plot_trades(self, buy_points, sell_points):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['time'], self.data['close'], label='Close Price', alpha=0.3, color='#7f7f7f')

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

        optimizer.plot_trades(final_buy_points, final_sell_points)


if __name__ == "__main__":
    args = parse_args()
    PBOUNDS = {
        'stoch_k_p': (args.stoch_k_period_min, args.stoch_k_period_max),
        'stoch_slow_k_p': (args.stoch_slowing_min, args.stoch_slowing_max),
        'stoch_slow_d_p': (args.stoch_d_period_min, args.stoch_d_period_max),
        'stoch_thr': (args.stoch_threshold_min, args.stoch_threshold_max),
        'bb_p': (args.bb_period_min, args.bb_period_max),
        'bb_dev_low': (args.bb_dev_low_min, args.bb_dev_low_max),
        'bb_dev_up': (args.bb_dev_up_min, args.bb_dev_up_max),
        'arm_pct': (args.arming_pct_min, args.arming_pct_max),
        'arm_stp_ls_pct': (args.arm_stop_loss_pct_min, args.arm_stop_loss_pct_max),
        'stp_ls_pct': (args.stop_loss_pct_min, args.stop_loss_pct_max)
    }

    run_optimization(args.filename, PBOUNDS, args.number_of_cores, args.start_date, args.end_date, args.init_points, args.iter_points, args.pair_points)
