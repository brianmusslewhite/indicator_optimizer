import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def visualize_top_results(top_results, dataset_name, start_date, end_date, time_now, plot_subfolder):
    if not top_results:
        print("No results to visualize.")
        return

    results_df = pd.DataFrame([res['params'] for res in top_results])
    results_df['profit'] = [res['target'] for res in top_results]
    filtered_results_df = results_df[results_df['profit'] != 0]

    if filtered_results_df.empty:
        print("No results to visualize after filtering out zero profits.")
        return

    pairplot = sns.pairplot(filtered_results_df, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height=2)
    pairplot.figure.suptitle(f'{dataset_name}', size=12)

    filename = f"{dataset_name}_PairPlot_{start_date}_to_{end_date}_Date_{time_now}.png"
    plt.savefig(os.path.join(plot_subfolder, filename))
    # plt.show()

def plot_trades(data, buy_points, sell_points, dataset_name, start_date, end_date, time_now, plot_subfolder):
    plt.figure(figsize=(14, 7))
    plt.plot(data['time'], data['close'], label='Close Price', alpha=0.3, color='#7f7f7f')

    plt.scatter(data.loc[data.index.isin(buy_points), 'time'], data.loc[data.index.isin(buy_points), 'close'], label='Buy', marker='^', color='green', alpha=1)
    plt.scatter(data.loc[data.index.isin(sell_points), 'time'], data.loc[data.index.isin(sell_points), 'close'], label='Sell', marker='v', color='red', alpha=1)

    plt.title(f'{dataset_name}_{start_date}_to_{end_date}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()

    filename = f"{dataset_name}_BUYSELLResults_{start_date}_to_{end_date}_Date_{time_now}.png"
    plt.savefig(os.path.join(plot_subfolder, filename))
    plt.show()

def plot_parameter_sensitivity(top_results, dataset_name, start_date, end_date, time_now, plot_subfolder):
    if not top_results:
        print("No results for sensitivity analysis.")
        return

    # Create a DataFrame from results
    results_df = pd.DataFrame([res['params'] for res in top_results])
    results_df['profit'] = [res['target'] for res in top_results]

    # Plotting
    num_params = len(results_df.columns) - 1  # exclude the profit column
    fig, axs = plt.subplots(nrows=num_params, figsize=(10, 5 * num_params))

    for i, param in enumerate(results_df.columns[:-1]):  # exclude the profit column
        sns.scatterplot(x=param, y='profit', data=results_df, ax=axs[i], color='blue', edgecolor='black')
        # axs[i].set_title(f'Sensitivity of Profit to {param}')
        axs[i].set_ylabel('Profit')
        axs[i].set_xlabel(param)
        axs[i].text(0.01, 0.05, f'{param}', transform=axs[i].transAxes, verticalalignment='top', fontsize=18, color='black')


    plt.tight_layout()
    # fig.suptitle(f'Sensitivity Analysis from {start_date} to {end_date}', fontsize=16)

    filename = f"{dataset_name}_Sensitivity_{start_date}_to_{end_date}_Date_{time_now}.png"
    plt.savefig(os.path.join(plot_subfolder, filename))
    plt.show()