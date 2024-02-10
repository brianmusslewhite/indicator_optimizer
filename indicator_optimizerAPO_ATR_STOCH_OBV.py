import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from bayes_opt import BayesianOptimization

import cProfile
pr = cProfile.Profile()
pr.enable()

# Constants and Initial Settings
START_DATE = '2023-06-01'
END_DATE = '2024-02-01'
INIT_POINTS = 20
N_ITER = 5
PAIR_POINTS = INIT_POINTS + N_ITER
MAX_HOLD_TIME = 720  # 12 hours in minutes


class SignalOptimizer:
    def __init__(self, filepath, pbounds):
        self.PBOUNDS = pbounds
        self.filepath = filepath
        processed_data_path = os.path.join('Processed Data', filepath)

        self.data = pd.read_csv(
            processed_data_path,
            usecols=['time', 'close', 'high', 'low', 'volume'],
            parse_dates=['time'],
            index_col='time',
            dtype={'close': 'float32', 'high': 'float32', 'low': 'float32', 'volume': 'float32'}
        )

        self.data = self.data.loc[START_DATE:END_DATE]
        self.data.ffill(inplace=True)
        self.data.reset_index(inplace=True)
        self.data_frequency_in_minutes = 1

    def calculate_apo(self, fast_period, slow_period):
        fast_ema = self.data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = self.data['close'].ewm(span=slow_period, adjust=False).mean()
        return fast_ema - slow_ema

    def calculate_atr(self, period):
        high_low = self.data['high'] - self.data['low']
        high_close = np.abs(self.data['high'] - self.data['close'].shift())
        low_close = np.abs(self.data['low'] - self.data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=int(period)).mean()

    def calculate_natr(self, atr_values, period):
        closing_prices = self.data['close']
        natr = (atr_values / closing_prices) * 100  # Normalize and convert to percentage
        return natr.rolling(window=period).mean()  # Apply rolling mean if needed

    def calculate_atr_ema(self, atr_values, period):
        return atr_values.ewm(span=period, adjust=False).mean()

    def calculate_stochastic_oscillator(self, k_period, slow_k_period, slow_d_period):
        # Ensure parameters are integers
        k_period = int(k_period)
        slow_k_period = int(slow_k_period)
        slow_d_period = int(slow_d_period)

        # Calculate Fast-K
        low_min = self.data['low'].rolling(window=k_period).min()
        high_max = self.data['high'].rolling(window=k_period).max()
        fast_k = ((self.data['close'] - low_min) / (high_max - low_min)) * 100

        # Calculate Slow-K (SMA of Fast-K)
        slow_k = fast_k.rolling(window=slow_k_period).mean()

        # Calculate Slow-D (SMA of Slow-K)
        slow_d = slow_k.rolling(window=slow_d_period).mean()

        return slow_k, slow_d

    def calculate_obv(self, ema_period):
        obv = np.where(self.data['close'] > self.data['close'].shift(), self.data['volume'], -self.data['volume']).cumsum()
        obv_ema = pd.Series(obv).ewm(span=ema_period, adjust=False).mean()
        return obv_ema

    def evaluate_performance(self, apo_fast_period, apo_slow_period, atr_ema_period, natr_period, stoch_k_period, slow_k_period, slow_d_period, obv_ema_period, arming_pct, stop_loss_pct, buy_signals_threshold):
        total_profit = 0.0
        position_open = False
        entry_price = 0.0
        highest_price_after_buy = 0.0
        armed = False
        entry_index = -1

        # Adjusted to include ATR calculation with natr_period, assuming its usage here instead
        atr_values = self.calculate_atr(natr_period)
        atr_ema_values = self.calculate_atr_ema(atr_values, atr_ema_period)
        apo_values = self.calculate_apo(apo_fast_period, apo_slow_period)
        stoch_k_values, stoch_d_values = self.calculate_stochastic_oscillator(stoch_k_period, slow_k_period, slow_d_period)
        obv_ema_values = self.calculate_obv(obv_ema_period)

        for i in range(1, len(self.data)):
            current_price = self.data['close'].iloc[i]
            signals_met = 0

            # Using ATR EMA as a volatility filter
            if atr_values.iloc[i] > atr_ema_values.iloc[i]:  # Market is considered volatile
                signals_met += 1  # Considering this as a condition met for trading

                # Additional conditions based on other indicators
                if apo_values.iloc[i] > 0 and apo_values.iloc[i-1] <= 0:
                    signals_met += 1
                if obv_ema_values.iloc[i] > obv_ema_values.iloc[i-1]:
                    signals_met += 1
                if stoch_k_values.iloc[i] > stoch_d_values.iloc[i] and stoch_k_values.iloc[i] < 80:
                    signals_met += 1

                # Buy condition based on signals met
                if signals_met >= buy_signals_threshold and not position_open:
                    position_open = True
                    entry_price = current_price
                    highest_price_after_buy = current_price
                    entry_index = i

            # Update highest price after buy for trailing stop loss
            if position_open and current_price > highest_price_after_buy:
                highest_price_after_buy = current_price
                if current_price >= entry_price * (1 + arming_pct / 100):
                    armed = True

            # Sell conditions: Stop loss or Maximum holding time reached
            if position_open:
                holding_time = i - entry_index
                max_holding_time_reached = holding_time >= (MAX_HOLD_TIME / self.data_frequency_in_minutes)
                stop_loss_triggered = current_price <= highest_price_after_buy * (1 - stop_loss_pct / 100) and armed

                if max_holding_time_reached or stop_loss_triggered:
                    total_profit += (current_price - entry_price) / entry_price
                    position_open = False
                    armed = False
                    entry_index = -1

        return total_profit

    def optimize(self):
        optimizer = BayesianOptimization(
            f=self.evaluate_performance,
            pbounds=self.PBOUNDS,
            random_state=1,
            verbose=2
        )

        optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)

        sorted_results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)
        self.top_results = sorted_results[:PAIR_POINTS] if len(sorted_results) >= PAIR_POINTS else sorted_results

        return self.top_results

    def visualize_top_results(self):
        if not self.top_results:
            print("No results to visualize.")
            return

        # Convert top results to a DataFrame for easier plotting
        results_df = pd.DataFrame([res['params'] for res in self.top_results])
        results_df['profit'] = [res['target'] for res in self.top_results]

        # Pairplot to explore the relationships between parameters and profit
        pairplot = sns.pairplot(results_df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height=2)
        pairplot.fig.suptitle('Parameter Relationships and Profit Distribution', y=1.02)  # Adjust title and its position

        # Save the visualization
        dataset_name = os.path.basename(self.filepath).split('.')[0]  # Assuming self.filepath exists in __init__
        subfolder = os.path.join('indicator_optimizer_plots', dataset_name)
        os.makedirs(subfolder, exist_ok=True)  # Create the directory if it doesn't exist

        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{dataset_name}_PairPlot_{START_DATE}_to_{END_DATE}_Date_{time_str}.png"
        plt.savefig(os.path.join(subfolder, filename))

        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Run Signal Optimizer")
    parser.add_argument("--filename", type=str, required=True, help="Input CSV file name")
    parser.add_argument("--apo_fast_min", type=int, default=5)
    parser.add_argument("--apo_fast_max", type=int, default=15)
    parser.add_argument("--apo_slow_min", type=int, default=10)
    parser.add_argument("--apo_slow_max", type=int, default=30)
    parser.add_argument("--atr_ema_period_min", type=int, default=10)
    parser.add_argument("--atr_ema_period_max", type=int, default=20)
    parser.add_argument("--natr_period_min", type=int, default=14)
    parser.add_argument("--natr_period_max", type=int, default=21)
    parser.add_argument("--stoch_k_period_min", type=int, default=5)
    parser.add_argument("--stoch_k_period_max", type=int, default=14)
    parser.add_argument("--slow_k_period_min", type=int, default=3)
    parser.add_argument("--slow_k_period_max", type=int, default=5)
    parser.add_argument("--slow_d_period_min", type=int, default=3)
    parser.add_argument("--slow_d_period_max", type=int, default=5)
    parser.add_argument("--obv_ema_period_min", type=int, default=10)
    parser.add_argument("--obv_ema_period_max", type=int, default=20)
    parser.add_argument("--arming_pct_min", type=float, default=0.6)
    parser.add_argument("--arming_pct_max", type=float, default=1.5)
    parser.add_argument("--stop_loss_pct_min", type=float, default=0.1)
    parser.add_argument("--stop_loss_pct_max", type=float, default=0.3)
    return parser.parse_args()


def run_optimization(filename, pbounds):
    optimizer = SignalOptimizer(filename, pbounds)
    top_results = optimizer.optimize()

    if top_results:
        print(f"Optimized Indicator Parameters: {top_results[0]['params']}")
        print(f"Best Target Value (Performance Metric): {top_results[0]['target']}")
        optimizer.visualize_top_results()


if __name__ == "__main__":
    args = parse_args()
    FILEPATH = args.filename
    PBOUNDS = {
        'apo_fast_period': (args.apo_fast_min, args.apo_fast_max),
        'apo_slow_period': (args.apo_slow_min, args.apo_slow_max),
        'atr_ema_period': (args.atr_ema_period_min, args.atr_ema_period_max),
        'natr_period': (args.natr_period_min, args.natr_period_max),  # Correctly included
        'stoch_k_period': (args.stoch_k_period_min, args.stoch_k_period_max),
        'slow_k_period': (args.slow_k_period_min, args.slow_k_period_max),
        'slow_d_period': (args.slow_d_period_min, args.slow_d_period_max),
        'obv_ema_period': (args.obv_ema_period_min, args.obv_ema_period_max),
        'arming_pct': (args.arming_pct_min, args.arming_pct_max),
        'stop_loss_pct': (args.stop_loss_pct_min, args.stop_loss_pct_max),
        'buy_signals_threshold': (3, 3),
    }

    run_optimization(FILEPATH, PBOUNDS)

    pr.disable()
    pr.print_stats(strip_dirs=True, sort='time').print_stats(50)
