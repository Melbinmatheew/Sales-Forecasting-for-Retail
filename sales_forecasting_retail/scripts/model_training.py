import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# Attempt to import Prophet
try:
    from prophet import Prophet
    prophet_available = True
    print("Prophet library imported successfully.")
except ImportError:
    prophet_available = False
    print("Prophet library not found. Prophet modeling will be skipped.")
    print("To install Prophet, run: pip install prophet")


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles cases where y_true is zero to avoid division by zero.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # Mask for non-zero true values
    mask = y_true != 0
    if not np.any(mask): # All true values are zero
        return np.nan # Or 0, depending on how you want to define MAPE for this case
        
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def train_and_evaluate_models():
    """
    Trains and evaluates time series models using walk-forward validation.
    """
    input_file_path = 'sales_forecasting_retail/data/daily_sales_with_features.csv'
    output_dir = 'sales_forecasting_retail/reports/'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Load the data
    try:
        # Assuming 'Purchase Date' was saved as index in the feature engineering step
        data = pd.read_csv(input_file_path, index_col='Purchase Date', parse_dates=True)
        print(f"Data loaded successfully from {input_file_path}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Data Column Check ---
    # The target column for sales forecasting.
    # In daily_sales.csv, it was 'Total Sales'. Let's assume it remains 'Total Sales'
    # in daily_sales_with_features.csv. If not, this needs to be adjusted.
    sales_column_name = 'Total Sales' 
    if sales_column_name not in data.columns:
        print(f"Error: Sales column '{sales_column_name}' not found in the data.")
        # As a fallback, try 'Purchase Amount (USD)' if it was the original name from raw data
        # This part depends on how data_preprocessing.py names the aggregated sales column.
        # For now, we'll stick to 'Total Sales' as per its creation in preprocess_sales_data.
        # if 'Purchase Amount (USD)' in data.columns:
        #     sales_column_name = 'Purchase Amount (USD)'
        #     print(f"Using fallback column: '{sales_column_name}'")
        # else:
        #     print("Neither 'Total Sales' nor 'Purchase Amount (USD)' found. Exiting.")
        #     return
        return


    print(f"Using '{sales_column_name}' as the target variable for forecasting.")

    # Handle potential NaN values from lag features
    # Print how many rows are dropped
    initial_rows = len(data)
    data.dropna(inplace=True)
    rows_dropped = initial_rows - len(data)
    print(f"Dropped {rows_dropped} rows containing NaN values (likely due to lag features).")

    if data.empty:
        print("Error: Data is empty after dropping NaN values. Cannot proceed with model training.")
        return
    
    if len(data[sales_column_name]) == 0:
        print(f"Error: The sales column '{sales_column_name}' has no data after processing. Cannot proceed.")
        return

    # --- Walk-Forward Validation Setup ---
    # n_splits: Number of times we retrain and test the model.
    # For a robust evaluation, this should be large enough.
    # Let's set it to forecast for a certain period, e.g., 30 days if data allows.
    forecast_horizon = 30 
    initial_train_size = 180  # Minimum days for initial training (e.g., ~6 months)

    if len(data) < initial_train_size + 1: # Need at least one point to test
        print(f"Error: Not enough data for walk-forward validation. Need at least {initial_train_size + 1} data points after NaN removal, found {len(data)}.")
        return
        
    # Adjust n_splits to not exceed available data
    # n_splits will be the number of one-step-ahead forecasts we make
    n_splits = min(forecast_horizon, len(data) - initial_train_size) 
    if n_splits <=0:
        print(f"Error: Not enough data to perform any splits for walk-forward validation. Data length: {len(data)}, Initial train size: {initial_train_size}")
        return

    print(f"Walk-forward validation setup: Initial train size = {initial_train_size}, Number of splits/forecasts = {n_splits}")


    model_results = {
        'ARIMA': [], 'ExponentialSmoothing': [], 'Prophet': [], 'Naive': [], 'Actuals': []
    }
    model_metrics = {'Model': [], 'MAE': [], 'RMSE': [], 'MAPE': []}

    # --- Walk-Forward Loop ---
    print("Starting walk-forward validation...")
    for i in range(n_splits):
        train_end_idx = initial_train_size + i
        train_data = data.iloc[:train_end_idx]
        test_data = data.iloc[train_end_idx : train_end_idx + 1] # Forecast one step ahead

        if test_data.empty:
            print(f"Warning: Test data is empty at split {i+1}. Stopping walk-forward validation.")
            break

        y_train = train_data[sales_column_name]
        y_test_actual = test_data[sales_column_name].iloc[0]
        model_results['Actuals'].append(y_test_actual)

        print(f"\nSplit {i+1}/{n_splits}: Training up to index {train_data.index[-1].date()}, Testing for {test_data.index[0].date()}")

        # --- Naive Model (Seasonal Naive - value from 7 days ago) ---
        pred_naive = np.nan
        if len(y_train) >= 7:
            pred_naive = y_train.iloc[-7]
        elif len(y_train) > 0: # Simple naive if not enough data for seasonal
            pred_naive = y_train.iloc[-1]
        model_results['Naive'].append(pred_naive)
        print(f"  Naive Prediction: {pred_naive:.2f}" if not np.isnan(pred_naive) else "  Naive Prediction: NaN")


        # --- ARIMA Model ---
        pred_arima = np.nan
        try:
            model_arima = ARIMA(y_train, order=(5,1,0)) # (p,d,q) order
            model_arima_fit = model_arima.fit()
            pred_arima = model_arima_fit.forecast(steps=1).iloc[0]
        except Exception as e:
            print(f"  ARIMA Error for split {i+1}: {e}")
        model_results['ARIMA'].append(pred_arima)
        print(f"  ARIMA Prediction: {pred_arima:.2f}" if not np.isnan(pred_arima) else "  ARIMA Prediction: NaN")

        # --- Exponential Smoothing Model (Holt-Winters Additive) ---
        pred_es = np.nan
        try:
            # Requires at least 2 full seasonal periods if seasonal_periods is set and > 1
            if len(y_train) >= 2 * 7 : # For seasonal_periods=7
                 model_es = ExponentialSmoothing(y_train, seasonal_periods=7, trend='add', seasonal='add', initialization_method='estimated')
                 model_es_fit = model_es.fit()
                 pred_es = model_es_fit.forecast(steps=1).iloc[0]
            else:
                # Fallback to simpler ES if not enough data for seasonal component
                model_es_simple = ExponentialSmoothing(y_train, trend='add', initialization_method='estimated')
                model_es_simple_fit = model_es_simple.fit()
                pred_es = model_es_simple_fit.forecast(steps=1).iloc[0]
                if i == 0: print("  Exponential Smoothing: Not enough data for seasonal component, using simpler ES for early splits.")

        except Exception as e:
            print(f"  Exponential Smoothing Error for split {i+1}: {e}")
        model_results['ExponentialSmoothing'].append(pred_es)
        print(f"  Exponential Smoothing Prediction: {pred_es:.2f}" if not np.isnan(pred_es) else "  Exponential Smoothing Prediction: NaN")


        # --- Prophet Model --- (if available)
        pred_prophet = np.nan
        if prophet_available:
            try:
                df_prophet_train = pd.DataFrame({'ds': train_data.index, 'y': y_train.values})
                # Suppress Prophet's informational messages
                model_prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False, 
                                        stan_backend=None) # Suppress cmdstanpy messages if not fully configured
                # Add other seasonalities or regressors if feature engineering provides them
                # e.g., model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)

                model_prophet.fit(df_prophet_train, show_stdout_logs=False, show_pystan_logs=False)
                future_df = model_prophet.make_future_dataframe(periods=1, freq='D') # 'D' for daily
                forecast_prophet = model_prophet.predict(future_df)
                pred_prophet = forecast_prophet['yhat'].iloc[-1]
            except Exception as e:
                print(f"  Prophet Error for split {i+1}: {e}")
        model_results['Prophet'].append(pred_prophet) # Appends nan if prophet_available is False or if error
        if prophet_available:
             print(f"  Prophet Prediction: {pred_prophet:.2f}" if not np.isnan(pred_prophet) else "  Prophet Prediction: NaN")

    print("\nWalk-forward validation completed.")
    print("Calculating performance metrics...")

    # --- Calculate Metrics After Loop ---
    actuals_all = np.array(model_results['Actuals'])

    for model_name in ['Naive', 'ARIMA', 'ExponentialSmoothing', 'Prophet']:
        if model_name == 'Prophet' and not prophet_available:
            print("Skipping Prophet metrics as it was not available/run.")
            model_metrics['Model'].append(model_name)
            model_metrics['MAE'].append(np.nan)
            model_metrics['RMSE'].append(np.nan)
            model_metrics['MAPE'].append(np.nan)
            continue

        preds_all = np.array(model_results[model_name])

        # Filter out np.nan from predictions and corresponding actuals
        valid_indices = ~np.isnan(preds_all)
        if not np.any(valid_indices):
            print(f"No valid (non-NaN) predictions for {model_name}. Metrics will be NaN.")
            mae, rmse, mape = np.nan, np.nan, np.nan
        else:
            preds_filtered = preds_all[valid_indices]
            actuals_filtered = actuals_all[valid_indices]
            
            if len(actuals_filtered) == 0 or len(preds_filtered) == 0 : # Should not happen if valid_indices has any True
                 mae, rmse, mape = np.nan, np.nan, np.nan
            else:
                mae = mean_absolute_error(actuals_filtered, preds_filtered)
                rmse = np.sqrt(mean_squared_error(actuals_filtered, preds_filtered))
                mape = mean_absolute_percentage_error(actuals_filtered, preds_filtered)

        model_metrics['Model'].append(model_name)
        model_metrics['MAE'].append(mae)
        model_metrics['RMSE'].append(rmse)
        model_metrics['MAPE'].append(mape)
        print(f"  Metrics for {model_name}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.2f}%" if not np.isnan(mae) else f"  Metrics for {model_name}: All predictions were NaN.")

    # --- Save Metrics ---
    metrics_df = pd.DataFrame(model_metrics)
    metrics_output_path = os.path.join(output_dir, 'model_evaluation_metrics.csv')
    try:
        metrics_df.to_csv(metrics_output_path, index=False)
        print(f"\nModel evaluation metrics saved to {metrics_output_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")

    print("\n--- Model Metrics ---")
    print(metrics_df.to_string())


if __name__ == "__main__":
    print("--- Starting Model Training and Evaluation Pipeline ---")
    train_and_evaluate_models()
    print("\n--- Model Training and Evaluation Pipeline Finished ---")
