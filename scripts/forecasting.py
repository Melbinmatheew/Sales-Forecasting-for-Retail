import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# Attempt to import Prophet
try:
    from prophet import Prophet
    prophet_available_for_forecasting = True
    print("Prophet library imported successfully for forecasting.")
except ImportError:
    prophet_available_for_forecasting = False
    print("Prophet library not found. Prophet forecasting will be skipped if selected.")
    print("To install Prophet, run: pip install prophet")


def generate_forecasts(best_model_name, data_file_path='sales_forecasting_retail/data/daily_sales_with_features.csv', forecast_periods=30, reports_dir='sales_forecasting_retail/reports/'):
    """
    Retrains the selected best model on the full dataset and generates future forecasts.
    Saves the forecasts and a plot.
    """
    print(f"\n--- Generating forecasts using {best_model_name} model ---")

    # Create reports directory if it doesn't exist
    if not os.path.exists(reports_dir):
        try:
            os.makedirs(reports_dir)
            print(f"Created directory: {reports_dir}")
        except OSError as e:
            print(f"Error creating directory {reports_dir}: {e}")
            return

    # Load data
    try:
        # Assuming 'Purchase Date' was saved as index in the feature engineering step
        data = pd.read_csv(data_file_path, index_col='Purchase Date', parse_dates=True)
        print(f"Data loaded successfully from {data_file_path}")
    except FileNotFoundError:
        print(f"Error: Input data file not found at {data_file_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Select the target variable - using 'Total Sales' for consistency with previous scripts
    sales_column_name = 'Total Sales'
    if sales_column_name not in data.columns:
        print(f"Error: Sales column '{sales_column_name}' not found in the data.")
        # Fallback attempt (though 'Total Sales' should be correct based on data_preprocessing.py)
        # if 'Purchase Amount (USD)' in data.columns:
        #     sales_column_name = 'Purchase Amount (USD)'
        #     print(f"Using fallback column: '{sales_column_name}'")
        # else:
        #     print(f"Neither '{sales_column_name}' nor 'Purchase Amount (USD)' found. Exiting forecast generation.")
        #     return
        return


    print(f"Using '{sales_column_name}' as the target variable for forecasting.")

    # Handle NaNs from lag features by dropping them
    initial_rows = len(data)
    data.dropna(subset=[sales_column_name], inplace=True) # Drop rows where target is NaN
    # Also consider dropping rows if other features used by model are NaN, though for non-ML models it's mainly target.
    # For ARIMA/ES/Prophet, only the target series (y_full) is directly used from this dataframe.
    # If regressors were used with Prophet, those columns would also need NaN handling.
    data.dropna(inplace=True) # General dropna for other feature NaNs from lags if any remain.
    rows_dropped = initial_rows - len(data)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows containing NaN values.")

    if data.empty or data[sales_column_name].empty:
        print("Error: Data is empty or sales column has no data after NaN removal. Cannot proceed with forecasting.")
        return

    y_full = data[sales_column_name]

    # --- Model Retraining and Forecasting ---
    forecast_values = None
    model_fit_error = False

    if best_model_name == 'ARIMA':
        print(f"Retraining ARIMA(5,1,0) model on full data ({len(y_full)} points)...")
        try:
            model_arima = ARIMA(y_full, order=(5,1,0)) # Using same order as in training
            model_arima_fit = model_arima.fit()
            forecast_values = model_arima_fit.forecast(steps=forecast_periods)
            print("ARIMA forecast generated.")
        except Exception as e:
            print(f"Error during ARIMA model fitting or forecasting: {e}")
            model_fit_error = True

    elif best_model_name == 'ExponentialSmoothing':
        print(f"Retraining ExponentialSmoothing model on full data ({len(y_full)} points)...")
        try:
            # Ensure enough data for seasonal component if seasonal_periods > 1
            seasonal_periods = 7
            if len(y_full) >= 2 * seasonal_periods:
                 model_es = ExponentialSmoothing(y_full, seasonal_periods=seasonal_periods, trend='add', seasonal='add', initialization_method='estimated')
            else:
                print(f"Warning: Not enough data for seasonal component (need {2*seasonal_periods}, have {len(y_full)}). Using non-seasonal Exponential Smoothing.")
                model_es = ExponentialSmoothing(y_full, trend='add', initialization_method='estimated')
            
            model_es_fit = model_es.fit()
            forecast_values = model_es_fit.forecast(steps=forecast_periods)
            print("Exponential Smoothing forecast generated.")
        except Exception as e:
            print(f"Error during Exponential Smoothing model fitting or forecasting: {e}")
            model_fit_error = True

    elif best_model_name == 'Prophet':
        if not prophet_available_for_forecasting:
            print("Prophet model was selected, but the library is not available. Skipping Prophet forecasting.")
            model_fit_error = True # Treat as an error for this selected model
        else:
            print(f"Retraining Prophet model on full data ({len(y_full)} points)...")
            try:
                df_prophet = pd.DataFrame({'ds': y_full.index, 'y': y_full.values})
                # Suppress Prophet's informational messages for cleaner output
                model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                                        stan_backend=None) # Avoids issues if cmdstanpy not fully configured
                model_prophet.fit(df_prophet, show_stdout_logs=False, show_pystan_logs=False)
                
                future_dates = model_prophet.make_future_dataframe(periods=forecast_periods, freq='D')
                prophet_forecast_df = model_prophet.predict(future_dates)
                # Extract only the forecast values for the future periods
                forecast_values = prophet_forecast_df['yhat'].iloc[-forecast_periods:]
                print("Prophet forecast generated.")
            except Exception as e:
                print(f"Error during Prophet model fitting or forecasting: {e}")
                model_fit_error = True
    
    elif best_model_name == 'Naive':
        print("Forecasting for Naive model: This typically involves repeating the last seasonal value or last actual value.")
        print("For simplicity, full forecast generation for Naive model is not implemented in this script beyond this message.")
        print("Consider implementing if Naive is often the best model and future values are needed.")
        return # Exiting as per instruction for Naive model

    else:
        print(f"Error: Model '{best_model_name}' is not supported for full retrain and forecast in this script.")
        return

    if forecast_values is None or model_fit_error:
        print(f"Could not generate forecast values for {best_model_name} due to previous errors or model not supported. Aborting.")
        return

    # --- Create Forecast DataFrame ---
    if not isinstance(forecast_values, pd.Series): # Prophet returns a DataFrame, others Series
        if isinstance(forecast_values, np.ndarray): # Convert numpy array to Series
             # Create date range for the forecast period
            last_date = y_full.index.max()
            forecast_date_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
            forecast_values = pd.Series(forecast_values, index=forecast_date_index)
        else: # e.g. Prophet output needs to be a Series with proper index
            # This case should be handled within Prophet block to ensure forecast_values is a Series
            # with the correct future dates as index. If Prophet block sets forecast_values correctly,
            # this part might not be strictly necessary but acts as a safeguard.
            if len(forecast_values) == forecast_periods:
                last_date = y_full.index.max()
                forecast_date_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
                # Assuming forecast_values from Prophet is already a Series but might need re-indexing if ds is not index
                # The Prophet block above does: forecast_values = prophet_forecast['yhat'].iloc[-forecast_periods:]
                # This should result in a Series. If its index is not DatetimeIndex, it needs fixing.
                # However, make_future_dataframe should give 'ds' column with dates.
                # Let's assume the Prophet block correctly prepares forecast_values as a Series with DatetimeIndex.
                # If it's a raw numpy array from prophet_forecast['yhat'], then:
                # forecast_values = pd.Series(forecast_values.values, index=forecast_date_index)
                pass # Assuming Prophet part results in a Series with appropriate index
            else:
                print("Error: Forecast values are not in a recognized Series format or length mismatch.")
                return


    # Ensure forecast_values is a Series with a DatetimeIndex
    if not isinstance(forecast_values.index, pd.DatetimeIndex):
        # This may happen if Prophet's output index is not set correctly
        # Re-create index if necessary
        print("Warning: Forecast values index is not DatetimeIndex. Attempting to fix.")
        last_date = y_full.index.max()
        forecast_date_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='D')
        if len(forecast_values) == len(forecast_date_index):
            forecast_values = pd.Series(forecast_values.values, index=forecast_date_index)
        else:
            print("Error: Cannot fix forecast index due to length mismatch. Aborting.")
            return


    forecast_df = pd.DataFrame({'Forecasted_Sales': forecast_values})
    forecast_df.index.name = 'Date'

    # --- Save Forecasts ---
    forecast_output_path = os.path.join(reports_dir, f'{best_model_name}_future_sales_forecasts.csv')
    try:
        forecast_df.to_csv(forecast_output_path)
        print(f"Forecasts saved to {forecast_output_path}")
    except Exception as e:
        print(f"Error saving forecasts: {e}")
        return # Cannot proceed to plot if saving failed

    # --- Plot Forecasts ---
    plt.figure(figsize=(14, 7))
    
    # Plot historical data (plot a subset for readability if very long)
    plot_history_points = min(len(y_full), 365 * 2) # Plot last 2 years or all if less
    plt.plot(y_full.index[-plot_history_points:], y_full.iloc[-plot_history_points:], label='Historical Daily Sales')
    
    # Plot forecasted data
    plt.plot(forecast_df.index, forecast_df['Forecasted_Sales'], label=f'{best_model_name} Forecast', color='red', linestyle='--')
    
    plt.title(f'{best_model_name} Sales Forecast vs Historical Data')
    plt.xlabel('Date')
    plt.ylabel(f'{sales_column_name} (USD)') # Use the actual sales column name
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    plot_output_path = os.path.join(reports_dir, f'{best_model_name}_sales_forecast_plot.png')
    try:
        plt.savefig(plot_output_path)
        print(f"Sales forecast plot saved to {plot_output_path}")
    except Exception as e:
        print(f"Error saving forecast plot: {e}")
    finally:
        plt.close() # Close the plot to free up memory

    print(f"--- Forecast generation for {best_model_name} completed ---")


if __name__ == "__main__":
    print("--- Running Forecasting Script (Example Test Runs) ---")
    
    # Define the path to the data with features
    data_path = 'sales_forecasting_retail/data/daily_sales_with_features.csv'
    # Define where reports (forecast CSV and plot) should be saved
    reports_output_dir = 'sales_forecasting_retail/reports/'

    # Example 1: Test with ARIMA (assuming it might be chosen)
    # Before running, ensure 'daily_sales_with_features.csv' exists.
    # This typically requires running data_preprocessing.py first.
    print("\nTesting with ARIMA model...")
    generate_forecasts(best_model_name='ARIMA', 
                       data_file_path=data_path, 
                       forecast_periods=30, 
                       reports_dir=reports_output_dir)

    # Example 2: Test with Exponential Smoothing
    print("\nTesting with ExponentialSmoothing model...")
    generate_forecasts(best_model_name='ExponentialSmoothing',
                       data_file_path=data_path,
                       forecast_periods=30,
                       reports_dir=reports_output_dir)

    # Example 3: Test with Prophet (if available)
    if prophet_available_for_forecasting:
        print("\nTesting with Prophet model...")
        generate_forecasts(best_model_name='Prophet', 
                           data_file_path=data_path, 
                           forecast_periods=30, 
                           reports_dir=reports_output_dir)
    else:
        print("\nSkipping Prophet test as the library is not available.")
        
    # Example 4: Test with Naive (to see the message)
    print("\nTesting with Naive model...")
    generate_forecasts(best_model_name='Naive',
                       data_file_path=data_path,
                       forecast_periods=30,
                       reports_dir=reports_output_dir)

    print("\n--- Forecasting Script Example Test Runs Finished ---")
