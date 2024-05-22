import os
import pandas as pd
import re

class DataLoader():
    def __init__(self, filepath, start_date, end_date):
        self.filepath = filepath
        self.start_date = start_date
        self.end_date = end_date
        self.load_and_prepare_data()

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
