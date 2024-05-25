import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_pair_plot(results, dataset_name, start_date, end_date, time_now, plot_subfolder):
    if results.empty:
        print("No results to visualize.")
        return

    filtered_results = results[results['profit_ratio'] != 0]
    if filtered_results.empty:
        print("No results to visualize after filtering out zero profits.")
        return
    
    columns_to_plot = [col for col in filtered_results.columns if col not in ['buy_points', 'sell_points']]
    head_filtered_results = filtered_results.sort_values('performance', ascending=False).head(200)
    plot_data = head_filtered_results[columns_to_plot]

    pairplot = sns.pairplot(plot_data, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, height=2)
    pairplot.figure.suptitle(f'{dataset_name}', size=12)

    filename = f"{dataset_name}_PairPlot_{start_date}_to_{end_date}_Date_{time_now}.png"
    plt.savefig(os.path.join(plot_subfolder, filename))
    plt.show()
    plt.close()

def plot_trades_on_data(data, buy_points, sell_points, dataset_name, start_date, end_date, time_now, plot_subfolder):
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
    plt.close()

def plot_parameter_sensitivity(results, dataset_name, start_date, end_date, time_now, plot_subfolder):
    if results.empty:
        print("No results for sensitivity analysis.")
        return

    positive_gain_results = results[results['total_percent_gain'] > 0]
    if positive_gain_results.empty:
        print("No positive gain results for sensitivity analysis.")
        return
    
    positive_gain_and_performance_results = positive_gain_results[positive_gain_results['performance'] > 0]
    if positive_gain_and_performance_results.empty:
        print("No positive gain and performance results for sensitivity analysis.")
        return
    
    positive_gain_and_performance_and_pr_results = positive_gain_and_performance_results[positive_gain_and_performance_results['profit_ratio'] > 0.5]
    if positive_gain_and_performance_and_pr_results.empty:
        print("No positive gain and performance results for sensitivity analysis.")
        return

    param_cols = [col for col in positive_gain_and_performance_and_pr_results.columns if col not in ['performance', 'total_percent_gain', 'profit_ratio', 'buy_points', 'sell_points']]
    num_params = len(param_cols)
    fig, axs = plt.subplots(nrows=num_params, figsize=(10, 5 * num_params))

    for i, param in enumerate(param_cols):
        sns.scatterplot(x=param, y='performance', data=positive_gain_and_performance_and_pr_results, ax=axs[i], color='blue', edgecolor='black')
        axs[i].set_ylabel('Performance')
        axs[i].text(0.01, 0.95, f'{param}', transform=axs[i].transAxes, verticalalignment='top', fontsize=18, color='black')
        axs[i].set_xlabel(' ')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle(f'{dataset_name}_{start_date}_to_{end_date}')

    filename = f"{dataset_name}_Sensitivity_{start_date}_to_{end_date}_Date_{time_now}.png"
    plt.savefig(os.path.join(plot_subfolder, filename))
    plt.show()
    plt.close()