import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import signal
import os
from datetime import datetime
from bayes_opt import BayesianOptimization, UtilityFunction
from joblib import Parallel, delayed
from math import ceil
from tqdm import tqdm


START_DATE = '2023-11-30'
END_DATE = '2023-12-31'
INIT_POINTS = 500
N_ITER = 1000
PAIR_POINTS = INIT_POINTS + N_ITER
MAX_HOLD_TIME = 48  # 12 hours in minutes

INTERRUPTED = False


def signal_handler(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True
    print("Signal received, stopping optimization...")
    raise KeyboardInterrupt


class SignalOptimizer:
    def __init__(self, filepath, pbounds, number_of_cores):
        self.PBOUNDS = pbounds
        self.filepath = filepath
        self.number_of_cores = number_of_cores
        self.dataset_name = os.path.basename(self.filepath).split('.')[0]

        self.plot_subfolder = os.path.join('indicator_optimizer_plots', self.dataset_name)
        os.makedirs(self.plot_subfolder, exist_ok=True)
        self.time_now = datetime.now().strftime('%Y%m%d_%H%M%S')

        processed_data_path = os.path.join('Processed Data', filepath)

        self.data = pd.read_csv(
            processed_data_path,
            usecols=['time', 'close', 'high', 'low', 'volume'],
            parse_dates=['time'],
            index_col='time',
            dtype={'close': 'float32', 'high': 'float32', 'low': 'float32', 'volume': 'float32'}
        )

        available_start_date = self.data.index.min()
        available_end_date = self.data.index.max()
        start_date = max(pd.Timestamp(START_DATE), available_start_date)
        end_date = min(pd.Timestamp(END_DATE), available_end_date)

        if start_date > end_date:
            raise ValueError("Adjusted start date is after the adjusted end date. No data available for the given range.")
        if start_date != pd.Timestamp(START_DATE) or end_date != pd.Timestamp(END_DATE):
            print(f"Date range adjusted based on available data: {start_date.date()} to {end_date.date()}")
        self.data = self.data.loc[start_date:end_date]
        if self.data.empty:
            raise ValueError(f"No data available between {start_date} and {end_date} after adjustments.")

        self.data.ffill(inplace=True)
        self.data.reset_index(inplace=True)
        self.data_frequency_in_minutes = 15
        self.best_buy_points = []
        self.best_sell_points = []
        self.best_performance = float('-inf')
        self.param_to_results = {}

    def calculate_bollinger_bands(self, period, deviation_lower, deviation_upper):
        period = int(period)
        sma = self.data['close'].rolling(window=period).mean()
        std = self.data['close'].rolling(window=period).std()
        upper_band = sma + (std * deviation_upper)
        lower_band = sma - (std * deviation_lower)
        return upper_band, lower_band

    def calculate_macd(self, fast_period, slow_period, signal_period):
        signal_period = int(signal_period)
        exp1 = self.data['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = self.data['close'].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal_period, adjust=False).mean()
        return macd, signal_line

    def calculate_parabolic_sar(self, af_init, af_max):
        high = self.data['high'].values.astype('float32')
        low = self.data['low'].values.astype('float32')
        close = self.data['close'].values.astype('float32')

        sar = np.full(close.shape, np.nan, dtype='float32')
        sar[0] = close[0]

        uptrend = True
        af = af_init
        ep = high[0]

        for i in range(1, len(close)):
            if uptrend:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_init, af_max)
                if low[i] < sar[i]:
                    uptrend = False
                    af = af_init
                    ep = low[i]
                    sar[i] = max(high[i - 1], high[i])
            else:
                sar[i] = sar[i - 1] + af * (ep - sar[i - 1])
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_init, af_max)
                if high[i] > sar[i]:
                    uptrend = True
                    af = af_init
                    ep = high[i]
                    sar[i] = min(low[i - 1], low[i])

        return pd.Series(sar, index=self.data.index)

    def evaluate_performance(self, macd_fast_period, macd_slow_period, macd_signal_period, bb_period, bb_dev_lower, bb_dev_upper, sar_af, sar_af_max, arming_pct, stop_loss_pct):
        upper_band, lower_band = self.calculate_bollinger_bands(bb_period, bb_dev_lower, bb_dev_upper)
        macd, signal_line = self.calculate_macd(macd_fast_period, macd_slow_period, macd_signal_period)
        sar = self.calculate_parabolic_sar(sar_af, sar_af_max)

        initial_balance = 1.0
        current_balance = initial_balance

        position_open = False
        entry_price = 0.0
        highest_price_after_buy = 0.0
        armed = False
        entry_index = -1
        buy_points = []
        sell_points = []

        for i in range(1, len(self.data)):
            current_price = self.data['close'].iloc[i]
            signals_met = 0

            if macd.iloc[i] > signal_line.iloc[i] and macd.iloc[i-1] <= signal_line.iloc[i-1]:
                signals_met += 1
            if current_price < lower_band.iloc[i]:
                signals_met += 1
            if current_price > sar.iloc[i]:
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
                if not armed and current_price >= entry_price * (1 + arming_pct / 100):
                    armed = True

                if armed and current_price <= highest_price_after_buy * (1 - stop_loss_pct / 100) or max_holding_time_reached:
                    sell_points.append(self.data.index[i])
                    trade_return = (current_price - entry_price) / entry_price  # Calculate return of the trade
                    capital_at_risk = current_balance * 0.10  # 10% of current capital at risk
                    profit_from_trade = capital_at_risk * trade_return  # Profit from the 10% capital used
                    current_balance += profit_from_trade  # Update current balance with profit from trade
                    position_open = False
                    armed = False
                    entry_index = -1

        # Calculate the final percent gain over the entire period
        total_percent_gain = (current_balance - initial_balance) / initial_balance * 100
        return total_percent_gain, buy_points, sell_points

    def evaluate_wrapper(self, params):
        performance, buy_points, sell_points = self.evaluate_performance(**params)
        self.param_to_results[str(params)] = (buy_points, sell_points)
        return performance

    def optimize(self):
        global INTERRUPTED
        signal.signal(signal.SIGINT, signal_handler)
        optimizer = BayesianOptimization(
            f=None,
            pbounds=self.PBOUNDS,
            random_state=1,
            verbose=2
        )
        utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
        init_points = [optimizer.suggest(utility) for _ in range(INIT_POINTS)]

        batch_size = self.number_of_cores
        num_batches = ceil(len(init_points) / batch_size)

        try:
            for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
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

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Proceeding with results obtained so far.")
        try:
            # Sequential Bayesian optimization
            top_performance = 0
            with tqdm(total=N_ITER, desc="Optimizing points") as pbar:
                for _ in range(N_ITER):
                    if INTERRUPTED:
                        print("Ctrl+C, breaking out of initial processing.")
                        break  # Exit the loop if an interruption was signaled
                    next_params = optimizer.suggest(utility)
                    performance, _, _ = self.evaluate_performance(**next_params)
                    optimizer.register(params=next_params, target=performance)
                    pbar.update(1)
                    if performance > top_performance:
                        top_performance = performance
                        print(f"New top performance: {performance}")

        except KeyboardInterrupt:
            print("\nOptimization interrupted by user. Proceeding with results obtained so far.")

        # Process and return the results obtained so far, regardless of whether there was an interruption
        sorted_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)
        self.top_results = sorted_results[:PAIR_POINTS] if len(sorted_results) >= PAIR_POINTS else sorted_results
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

        filename = f"{self.dataset_name}_PairPlot_{START_DATE}_to_{END_DATE}_Date_{self.time_now}.png"
        plt.savefig(os.path.join(self.plot_subfolder, filename))
        plt.show()

    def plot_trades(self, buy_points, sell_points):
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['time'], self.data['close'], label='Close Price', alpha=0.5)

        plt.scatter(self.data.loc[self.data.index.isin(buy_points), 'time'], self.data.loc[self.data.index.isin(buy_points), 'close'], label='Buy', marker='^', color='green', alpha=1)
        plt.scatter(self.data.loc[self.data.index.isin(sell_points), 'time'], self.data.loc[self.data.index.isin(sell_points), 'close'], label='Sell', marker='v', color='red', alpha=1)

        plt.title(f'{self.dataset_name}_{START_DATE}_to_{END_DATE}')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()

        filename = f"{self.dataset_name}_BUYSELLResults_{START_DATE}_to_{END_DATE}_Date_{self.time_now}.png"
        plt.savefig(os.path.join(self.plot_subfolder, filename))
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Signal Optimizer")
    parser.add_argument("--filename", type=str, required=True, help="Input CSV file name")
    parser.add_argument("--macd_fast_min", type=int, default=5, help="Minimum MACD fast period")
    parser.add_argument("--macd_fast_max", type=int, default=15, help="Maximum MACD fast period")
    parser.add_argument("--macd_slow_min", type=int, default=12, help="Minimum MACD slow period")
    parser.add_argument("--macd_slow_max", type=int, default=26, help="Maximum MACD slow period")
    parser.add_argument("--macd_signal_min", type=int, default=9, help="Minimum MACD signal period")
    parser.add_argument("--macd_signal_max", type=int, default=18, help="Maximum MACD signal period")
    parser.add_argument("--bb_period_min", type=int, default=5, help="Minimum Bollinger Bands period")
    parser.add_argument("--bb_period_max", type=int, default=20, help="Maximum Bollinger Bands period")
    parser.add_argument("--bb_dev_lower_min", type=float, default=1.5, help="Minimum Bollinger Bands lower deviation")
    parser.add_argument("--bb_dev_lower_max", type=float, default=2.5, help="Maximum Bollinger Bands lower deviation")
    parser.add_argument("--bb_dev_upper_min", type=float, default=1.5, help="Minimum Bollinger Bands upper deviation")
    parser.add_argument("--bb_dev_upper_max", type=float, default=2.5, help="Maximum Bollinger Bands upper deviation")
    parser.add_argument("--sar_af_max", type=float, default=0.2, help="Maximum accelerating factor for SAR")
    parser.add_argument("--sar_af_min", type=float, default=0.02, help="Minimum accelerating factor for SAR")
    parser.add_argument("--sar_af_max_max", type=float, default=0.4, help="Maximum accelerating factor max for SAR")
    parser.add_argument("--sar_af_max_min", type=float, default=0.2, help="Minimum accelerating factor max for SAR")
    parser.add_argument("--arming_pct_min", type=float, default=0.6, help="Minimum arming percentage for stop loss")
    parser.add_argument("--arming_pct_max", type=float, default=1.5, help="Maximum arming percentage for stop loss")
    parser.add_argument("--stop_loss_pct_min", type=float, default=0.1, help="Minimum stop loss percentage")
    parser.add_argument("--stop_loss_pct_max", type=float, default=0.3, help="Maximum stop loss percentage")
    parser.add_argument("--number_of_cores", type=int, default=1, help="Number of CPU cores to use")
    return parser.parse_args()


def run_optimization(filename, pbounds, number_of_cores):
    optimizer = SignalOptimizer(filename, pbounds, number_of_cores)
    top_results = optimizer.optimize()

    if top_results:
        best_params = top_results[0]['params']
        final_performance, final_buy_points, final_sell_points = optimizer.evaluate_performance(**best_params)

        print(f"Optimized Indicator Parameters: {best_params}")
        print(f"Best Target Value (Performance Metric): {top_results[0]['target']}")

        optimizer.visualize_top_results()
        optimizer.plot_trades(final_buy_points, final_sell_points)


if __name__ == "__main__":
    args = parse_args()
    FILEPATH = args.filename
    PBOUNDS = {
        'macd_fast_period': (args.macd_fast_min, args.macd_fast_max),
        'macd_slow_period': (args.macd_slow_min, args.macd_slow_max),
        'macd_signal_period': (args.macd_signal_min, args.macd_signal_max),
        'bb_period': (args.bb_period_min, args.bb_period_max),
        'bb_dev_lower': (args.bb_dev_lower_min, args.bb_dev_lower_max),
        'bb_dev_upper': (args.bb_dev_upper_min, args.bb_dev_upper_max),
        'sar_af': (args.sar_af_min, args.sar_af_max),
        'sar_af_max': (args.sar_af_max_min, args.sar_af_max_max),
        'arming_pct': (args.arming_pct_min, args.arming_pct_max),
        'stop_loss_pct': (args.stop_loss_pct_min, args.stop_loss_pct_max)
    }

    run_optimization(FILEPATH, PBOUNDS, args.number_of_cores)
