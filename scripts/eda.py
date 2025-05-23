import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

def perform_eda():
    """
    Performs Exploratory Data Analysis (EDA) on the daily sales data.
    """
    input_file_path = 'sales_forecasting_retail/data/daily_sales.csv'
    output_dir = 'sales_forecasting_retail/reports/'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load the daily sales data
    try:
        daily_sales_df = pd.read_csv(input_file_path, index_col='Purchase Date', parse_dates=True)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if daily_sales_df.empty:
        print("Error: Loaded data is empty. Cannot perform EDA.")
        # Create an empty summary file to indicate an attempt was made
        with open(os.path.join(output_dir, 'eda_summary.txt'), 'w') as f:
            f.write("EDA could not be performed because the input data was empty or missing.\n")
        return

    # Ensure the index is a DatetimeIndex
    if not isinstance(daily_sales_df.index, pd.DatetimeIndex):
        print("Error: Index is not a DatetimeIndex. Please check data loading.")
        return

    # --- Time Series Plot ---
    plt.figure(figsize=(12, 6))
    daily_sales_df['Total Sales'].plot()
    plt.title('Daily Sales Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Sales (USD)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'daily_sales_plot.png'))
    plt.close()
    print(f"Saved daily sales plot to {os.path.join(output_dir, 'daily_sales_plot.png')}")

    # --- Seasonal Decomposition ---
    # Determine period: if less than 2 years of data, decomposition might be problematic or need adjustment
    # For daily data, common periods are 7 (weekly), 30 (monthly), 365 (yearly)
    # We'll use 30 for monthly as a starting point.
    # The statsmodels seasonal_decompose requires at least 2 full periods of data.
    period = 30 
    if len(daily_sales_df) >= 2 * period:
        try:
            decomposition = sm.tsa.seasonal_decompose(daily_sales_df['Total Sales'], model='additive', period=period)
            fig = decomposition.plot()
            fig.set_size_inches(12, 8)
            plt.suptitle('Seasonal Decomposition of Daily Sales (Period=30)', y=0.95) # Add a main title
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
            plt.savefig(os.path.join(output_dir, 'seasonal_decomposition_plot.png'))
            plt.close(fig)
            print(f"Saved seasonal decomposition plot to {os.path.join(output_dir, 'seasonal_decomposition_plot.png')}")
            seasonal_component_available = True
        except Exception as e:
            print(f"Error during seasonal decomposition: {e}. Plot not saved.")
            seasonal_component_available = False
            # Create a placeholder or note if decomposition fails
            with open(os.path.join(output_dir, 'seasonal_decomposition_plot.png.txt'), 'w') as f:
                 f.write(f"Seasonal decomposition could not be performed. Error: {e}\n")

    else:
        print(f"Not enough data for seasonal decomposition with period {period} (requires at least {2*period} data points). Skipping.")
        seasonal_component_available = False
        # Create a placeholder or note if decomposition is skipped
        with open(os.path.join(output_dir, 'seasonal_decomposition_plot.png.txt'), 'w') as f:
            f.write(f"Seasonal decomposition skipped due to insufficient data for period {period}.\n")


    # --- ACF and PACF Plots ---
    # Check if there's enough data for ACF/PACF
    # Generally, nlags should be less than n_observations / 2
    # Default nlags in plot_acf/plot_pacf is min(10 * log10(nobs), nobs - 1)
    if len(daily_sales_df['Total Sales']) > 20: # Arbitrary threshold to ensure enough data
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        try:
            sm.graphics.tsa.plot_acf(daily_sales_df['Total Sales'], ax=axes[0], lags=min(40, len(daily_sales_df)//2 - 1))
            axes[0].set_title('Autocorrelation Function (ACF)')
            sm.graphics.tsa.plot_pacf(daily_sales_df['Total Sales'], ax=axes[1], lags=min(40, len(daily_sales_df)//2 - 1), method='ywm') # ywm can be more stable for some series
            axes[1].set_title('Partial Autocorrelation Function (PACF)')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'acf_pacf_plots.png'))
            plt.close(fig)
            print(f"Saved ACF and PACF plots to {os.path.join(output_dir, 'acf_pacf_plots.png')}")
            acf_pacf_available = True
        except Exception as e:
            print(f"Error generating ACF/PACF plots: {e}. Plots not saved.")
            acf_pacf_available = False
            # Create a placeholder or note if ACF/PACF fails
            with open(os.path.join(output_dir, 'acf_pacf_plots.png.txt'), 'w') as f:
                 f.write(f"ACF/PACF plot generation failed. Error: {e}\n")
    else:
        print("Not enough data to generate meaningful ACF/PACF plots. Skipping.")
        acf_pacf_available = False
        # Create a placeholder or note if ACF/PACF is skipped
        with open(os.path.join(output_dir, 'acf_pacf_plots.png.txt'), 'w') as f:
            f.write("ACF/PACF plot generation skipped due to insufficient data.\n")


    # --- EDA Summary ---
    summary_lines = []
    summary_lines.append("Exploratory Data Analysis (EDA) Summary:")
    summary_lines.append("========================================")

    # Trend observation (visual inspection of time series plot)
    # This is a simplified heuristic. More robust trend detection would require statistical tests.
    if not daily_sales_df['Total Sales'].empty:
        first_half_mean = daily_sales_df['Total Sales'].iloc[:len(daily_sales_df)//2].mean()
        second_half_mean = daily_sales_df['Total Sales'].iloc[len(daily_sales_df)//2:].mean()
        if second_half_mean > first_half_mean * 1.1: # 10% increase
            summary_lines.append("- Overall Trend: Apparent upward trend in sales over time.")
        elif first_half_mean > second_half_mean * 1.1: # 10% decrease
            summary_lines.append("- Overall Trend: Apparent downward trend in sales over time.")
        else:
            summary_lines.append("- Overall Trend: Sales appear relatively stable or have no strong linear trend.")
    else:
        summary_lines.append("- Overall Trend: Could not be determined due to empty data.")


    # Seasonality observation
    if seasonal_component_available:
        summary_lines.append("- Seasonality: Seasonal decomposition plot (period=30) should be inspected to identify monthly patterns. Look for repeating cycles in the 'Seasonal' component.")
    else:
        summary_lines.append("- Seasonality: Seasonal decomposition was not performed or failed. Check logs and 'seasonal_decomposition_plot.png.txt'.")

    # ACF/PACF observations
    if acf_pacf_available:
        summary_lines.append("- ACF/PACF Plots: ")
        summary_lines.append("  - ACF: Observe if it tails off or cuts off. A cut-off after q lags suggests an MA(q) model.")
        summary_lines.append("  - PACF: Observe if it tails off or cuts off. A cut-off after p lags suggests an AR(p) model.")
        summary_lines.append("  - If both ACF and PACF tail off, an ARMA(p,q) model might be appropriate.")
        summary_lines.append("  - Significant spikes at seasonal lags (e.g., lag 30, 60 for monthly) in ACF/PACF would indicate seasonality not captured by the decomposition.")
    else:
        summary_lines.append("- ACF/PACF Plots: Not generated due to insufficient data or errors. Check logs and 'acf_pacf_plots.png.txt'.")

    summary_text = "\n".join(summary_lines)
    try:
        with open(os.path.join(output_dir, 'eda_summary.txt'), 'w') as f:
            f.write(summary_text)
        print(f"Saved EDA summary to {os.path.join(output_dir, 'eda_summary.txt')}")
    except Exception as e:
        print(f"Error saving EDA summary: {e}")


if __name__ == "__main__":
    print("Starting Exploratory Data Analysis (EDA)...")
    perform_eda()
    print("Exploratory Data Analysis (EDA) completed.")
