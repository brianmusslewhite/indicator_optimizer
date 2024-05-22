import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor


def process_trading_history(file_path, data_frequency_minutes, output_folder):
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = ['timestamp', 'price', 'volume']

        # Convert timestamps and set index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('timestamp')

        # Resample and calculate OHLCV
        ohlcv = df['price'].resample(f'{data_frequency_minutes}min').ohlc()
        volume = df['volume'].resample(f'{data_frequency_minutes}min').sum()
        ohlcv.ffill(inplace=True)

        # Add volume and column names
        ohlcv = ohlcv[['open', 'high', 'low', 'close']]
        ohlcv['volume'] = volume

        # Reset index
        ohlcv = ohlcv.reset_index()
        ohlcv.columns = ['time', 'open', 'high', 'low', 'close', 'volume']

        # Construct output file path
        base_name = os.path.basename(file_path)
        output_path = os.path.join(output_folder, os.path.splitext(base_name)[0] + f"_{data_frequency_minutes}min_Kraken.csv")

        # Write to a new CSV
        ohlcv.to_csv(output_path, index=False)
        print(f"Processed and saved: {output_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    data_frequency_minutes = [5, 15, 30]
    output_folder = "Processed Data"
    file_paths = [
        r'~/Downloads/Kraken_Trading_History/DASHXBT.csv',
        r'~/Downloads/Kraken_Trading_History/MATICXBT.csv',
        r'~/Downloads/Kraken_Trading_History/LINKXBT.csv',
        r'~/Downloads/Kraken_Trading_History/BCHXBT.csv',
        r'~/Downloads/Kraken_Trading_History/ETHXBT.csv',
        r'~/Downloads/Kraken_Trading_History/EOSXBT.csv',
        r'~/Downloads/Kraken_Trading_History/XRPXBT.csv',
        r'~/Downloads/Kraken_Trading_History/LTCXBT.csv',
        r'~/Downloads/Kraken_Trading_History/XDGXBT.csv',
        r'~/Downloads/Kraken_Trading_History/XLMXBT.csv',
        r'~/Downloads/Kraken_Trading_History/XRPXBT.csv'
    ]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ThreadPoolExecutor() as executor:
        for file_path in file_paths:
            for frequency in data_frequency_minutes:
                executor.submit(process_trading_history, file_path, frequency, output_folder)
