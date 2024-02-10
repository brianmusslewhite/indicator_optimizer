import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import timedelta
from bayes_opt import BayesianOptimization

FILEPATH = 'transformed_data.csv'
START_DATE = '2023-05-13'
END_DATE = '2024-02-01'
PBOUNDS = {'fast_length': (5, 15), 'slow_length': (20, 30), 'signal_period': (9, 12),
           'rsi_period': (10, 20), 'rsi_value': (20, 80),
           'k_period': (10, 20), 'd_period': (3, 8), 'stoch_value': (20, 80)}
INIT_POINTS = 1000
N_ITER = 100
INVEST_PCT = 0.05  # Percentage of funds invested in each trade


class SignalOptimizer:
    def __init__(self, filepath):
        self.data = pd.read_csv(
            filepath,
            usecols=['time', 'close'],
            parse_dates=['time'],
            index_col='time',
            dtype={'close': 'float32'}
        )

        self.data = self.data.loc[START_DATE:END_DATE]
        self.data.fillna(method='ffill', inplace=True)
        self.data.reset_index(inplace=True)

        # Pre-calculate technical indicators
        self.macd_histograms = {}
        self.rsis = {}
        self.stochastics = {}

    def calculate_macd(self, close_prices, fast_length, slow_length, signal_period):
        fast_length = int(fast_length) if fast_length >= 1 else 1
        slow_length = int(slow_length) if slow_length >= 1 else 1
        signal_period = int(signal_period) if signal_period >= 1 else 1

        ema_fast = close_prices.ewm(span=fast_length, adjust=False).mean()
        ema_slow = close_prices.ewm(span=slow_length, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        macd_histogram = macd_line - signal_line
        return macd_histogram

    def calculate_rsi(self, close_prices, period):
        # Forcefully convert the period to an integer and ensure it's at least 1
        period = int(period) if period >= 1 else 1

        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()

        # Avoid division by zero
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_stochastic(self, close_prices, k_period, d_period):
        # Forcefully convert the periods to integers and ensure they're at least 1
        k_period = int(k_period) if k_period >= 1 else 1
        d_period = int(d_period) if d_period >= 1 else 1

        low_min = close_prices.rolling(window=k_period).min()
        high_max = close_prices.rolling(window=k_period).max()

        stoch_k = 100 * ((close_prices - low_min) / (high_max - low_min))
        stoch_d = stoch_k.rolling(window=d_period).mean()
        return stoch_k, stoch_d

    def get_macd_histogram(self, fast_length, slow_length, signal_period):
        key = (fast_length, slow_length, signal_period)
        if key not in self.macd_histograms:
            self.macd_histograms[key] = self.calculate_macd(self.data['close'], *key)
        return self.macd_histograms[key]

    def get_rsi(self, period):
        if period not in self.rsis:
            self.rsis[period] = self.calculate_rsi(self.data['close'], period)
        return self.rsis[period]

    def get_stochastic(self, k_period, d_period):
        key = (k_period, d_period)
        if key not in self.stochastics:
            self.stochastics[key] = self.calculate_stochastic(self.data['close'], *key)
        return self.stochastics[key]

    def optimize(self):
        optimizer = BayesianOptimization(
            f=self.evaluate_performance,
            pbounds=PBOUNDS,
            random_state=1
        )
        try:
            optimizer.maximize(init_points=INIT_POINTS, n_iter=N_ITER)
        except KeyboardInterrupt:  # If Ctrl+C is pressed during the execution of optimizer.maximize
            return optimizer.max
        else:
            return optimizer.max

    def plot_signals(self, best_result):
        macd_histogram = self.get_macd_histogram(best_result['params']['fast_length'],
                                                 best_result['params']['slow_length'],
                                                 best_result['params']['signal_period'])
        rsi = self.get_rsi(best_result['params']['rsi_period'])
        stoch_k, stoch_d = self.get_stochastic(best_result['params']['k_period'],
                                               best_result['params']['d_period'])

        buy_signals = ((macd_histogram > 0) & (rsi < 70) & (stoch_k > stoch_d))
        sell_signals = ((macd_histogram < 0) & (rsi > 30) & (stoch_k < stoch_d))

        # Additional logic for filtering signals
        position_open = False
        for i in range(len(sell_signals)):
            if buy_signals[i]:
                position_open = True
            if sell_signals[i] and position_open:
                position_open = False
                continue
            if sell_signals[i] and not position_open:
                sell_signals[i] = False

        plt.figure(figsize=(15, 7))
        plt.plot(self.data['time'], self.data['close'], label='Close Price', alpha=0.5)
        plt.scatter(self.data['time'][buy_signals], self.data['close'][buy_signals], color='green', label='Buy Signal', marker='^', alpha=1)
        plt.scatter(self.data['time'][sell_signals], self.data['close'][sell_signals], color='red', label='Sell Signal', marker='v', alpha=1)

        plt.title('Buy and Sell Signals on Close Price RSI MACD STO')
        plt.xlabel('Time')
        plt.ylabel('Close Price')
        plt.legend()

        # Use AutoDateLocator for handling the date axis
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())

        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    def evaluate_performance(self, fast_length, slow_length, signal_period, rsi_period, rsi_value, k_period, d_period, stoch_value):
        # Make sure the fast_length is always smaller than the slow_length
        if fast_length >= slow_length:
            return -1e10
        macd_histogram = self.get_macd_histogram(fast_length, slow_length, signal_period)
        rsi = self.get_rsi(rsi_period)
        stoch_k, stoch_d = self.get_stochastic(k_period, d_period)

        buy_signals = ((macd_histogram > 0) & (rsi <= rsi_value) & (stoch_k >= stoch_value))
        sell_signals = ((macd_histogram < 0) & (rsi >= rsi_value) & (stoch_k <= stoch_value))

        cash = 100  # Arbitrary starting with a cash amount to invest
        shares_owned = 0
        position_open = False
        position_open_date = None  # Track time of position opening

        for i, current_price in enumerate(self.data['close']):
            if position_open and (sell_signals[i] or pd.to_datetime(self.data['time'][i]) - position_open_date > timedelta(days=4)):
                # Sell shares if sell signal or position held for over 4 days
                cash += shares_owned * current_price
                position_open = False
                shares_owned = 0

            elif not position_open and buy_signals[i]:
                # Open position
                invest_amount = cash * INVEST_PCT
                shares_bought = invest_amount / current_price
                cash -= invest_amount
                shares_owned += shares_bought
                position_open = True

                # Update the time the position was opened
                position_open_date = pd.to_datetime(self.data['time'][i])

        # Final asset value (if we had shares sell them at the final price)
        final_price = self.data['close'].iloc[-1]
        equity = shares_owned * final_price
        total_asset_value = cash + equity

        # Total return percentage
        total_return_percent = ((total_asset_value - 100) / 100) * 100

        return -total_return_percent


def run_optimization():
    optimizer = SignalOptimizer(FILEPATH)
    best_result = optimizer.optimize()

    print(f"Optimized Indicator Parameters: {best_result['params']}")
    print(f"Best Target Value (Performance Metric): {best_result['target']}")

    optimizer.plot_signals(best_result)


if __name__ == "__main__":
    run_optimization()
