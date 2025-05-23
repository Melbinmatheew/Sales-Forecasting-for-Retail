import pandas as pd
import os
from data_preprocessing import preprocess_sales_data, engineer_features
from eda import perform_eda
from model_training import train_and_evaluate_models
from forecasting import generate_forecasts

def compare_and_select_model(metrics_file_path='sales_forecasting_retail/reports/model_evaluation_metrics.csv', report_file_path='sales_forecasting_retail/reports/model_selection_summary.txt'):
    """
    Loads model evaluation metrics, selects the best model based on MAE,
    prints results, saves a summary report, and returns the best model's name.
    """
    print(f"Attempting to load metrics from: {metrics_file_path}")

    # Ensure the directory for the report exists
    output_dir = os.path.dirname(report_file_path)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            return None # Cannot proceed if directory creation fails

    # Load metrics
    try:
        metrics_df = pd.read_csv(metrics_file_path)
    except FileNotFoundError:
        print(f"Error: Metrics file not found at {metrics_file_path}")
        try:
            with open(report_file_path, 'w') as f:
                f.write(f"Model selection failed: Metrics file not found at {metrics_file_path}\n")
            print(f"Empty report saved to {report_file_path} due to missing metrics file.")
        except Exception as e_save:
            print(f"Error saving empty report: {e_save}")
        return None
    except Exception as e:
        print(f"Error loading metrics file {metrics_file_path}: {e}")
        return None

    if metrics_df.empty:
        print("Error: Metrics file is empty. Cannot perform model selection.")
        try:
            with open(report_file_path, 'w') as f:
                f.write("Model selection failed: Metrics file was empty.\n")
            print(f"Empty report saved to {report_file_path} due to empty metrics file.")
        except Exception as e_save:
            print(f"Error saving empty report: {e_save}")
        return None

    # Validate required columns
    required_columns = ['Model', 'MAE', 'RMSE', 'MAPE']
    if not all(col in metrics_df.columns for col in required_columns):
        print(f"Error: Metrics file must contain columns: {', '.join(required_columns)}. Found: {', '.join(metrics_df.columns)}")
        try:
            with open(report_file_path, 'w') as f:
                f.write(f"Model selection failed: Metrics file is missing required columns. Expected {required_columns}.\n")
            print(f"Empty report saved to {report_file_path} due to missing columns in metrics file.")
        except Exception as e_save:
            print(f"Error saving empty report: {e_save}")
        return None

    # Model Comparison (select based on lowest MAE)
    # Drop rows where MAE is NaN before finding the minimum
    metrics_df_cleaned = metrics_df.dropna(subset=['MAE'])

    if metrics_df_cleaned.empty:
        print("Error: All models have NaN MAE values. Cannot select the best model.")
        summary_lines = ["Model Comparison Results:", metrics_df.to_string(index=False), "\nBest model selection failed: All models have NaN MAE values."]
        summary_report = "\n".join(summary_lines)
        try:
            with open(report_file_path, 'w') as f:
                f.write(summary_report)
            print(f"Report indicating NaN MAE issues saved to {report_file_path}")
        except Exception as e_save:
            print(f"Error saving report: {e_save}")
        return None

    best_model_df = metrics_df_cleaned.loc[[metrics_df_cleaned['MAE'].idxmin()]] # Keep as DataFrame
    
    if best_model_df.empty:
        print("Error: Could not determine the best model after filtering.")
        return None
        
    best_model_name = best_model_df.iloc[0]['Model']
    best_mae = best_model_df.iloc[0]['MAE']
    best_rmse = best_model_df.iloc[0]['RMSE']
    best_mape = best_model_df.iloc[0]['MAPE']

    # Print results to console
    print("\nModel Comparison Results:")
    print(metrics_df.to_string(index=False)) 
    print(f"\nBest model selected: {best_model_name} with MAE: {best_mae:.4f}, RMSE: {best_rmse:.4f}, MAPE: {best_mape:.4f}%")

    # Save Summary Report
    summary_lines = [
        "Model Selection Summary",
        "=======================",
        f"Best Model: {best_model_name}",
        "Selection Metric: Lowest MAE (Mean Absolute Error)",
        "\nSelected Model Metrics:",
        f"  - MAE: {best_mae:.4f}",
        f"  - RMSE: {best_rmse:.4f}",
        f"  - MAPE: {best_mape:.4f}%",
        "\nFull Model Comparison Table:",
        metrics_df.to_string(index=False)
    ]
    summary_report = "\n".join(summary_lines)

    try:
        with open(report_file_path, 'w') as f:
            f.write(summary_report)
        print(f"\nModel selection summary saved to {report_file_path}")
    except Exception as e:
        print(f"Error saving model selection summary: {e}")
        return None # Indicate failure if report saving fails

    return best_model_name


if __name__ == "__main__":
    print("--- Starting Sales Forecasting Pipeline ---")

    # Define file paths (centralized for clarity)
    raw_data_input_file = 'sales_forecasting_retail/data/shopping_trends.csv' # Assumed, as preprocess_sales_data uses it internally
    processed_data_output_file = 'sales_forecasting_retail/data/daily_sales.csv' # Output of preprocess_sales_data
    featured_data_file = 'sales_forecasting_retail/data/daily_sales_with_features.csv' # Output of engineer_features
    reports_dir = 'sales_forecasting_retail/reports/'
    metrics_file = os.path.join(reports_dir, 'model_evaluation_metrics.csv')
    selection_report_file = os.path.join(reports_dir, 'model_selection_summary.txt')
    
    # Ensure reports directory exists (some functions might do this, but good to have it early)
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)
        print(f"Created reports directory: {reports_dir}")

    pipeline_successful = True

    # Step 1: Data Preprocessing and Feature Engineering
    print("\n--- Step 1: Running Data Preprocessing and Feature Engineering ---")
    try:
        # preprocess_sales_data has its own input/output paths defined internally
        processed_path = preprocess_sales_data() 
        if processed_path: # It returns the output path on success
            # engineer_features also has default input/output paths
            engineer_features(input_file_path=processed_path, output_file_path=featured_data_file)
            print("Data Preprocessing and Feature Engineering Complete.")
        else:
            print("Error in preprocess_sales_data. Halting pipeline.")
            pipeline_successful = False
    except Exception as e:
        print(f"Error in Step 1 (Data Preprocessing/Feature Engineering): {e}")
        pipeline_successful = False

    # Step 2: Exploratory Data Analysis
    if pipeline_successful:
        print("\n--- Step 2: Running Exploratory Data Analysis ---")
        try:
            # perform_eda uses 'sales_forecasting_retail/data/daily_sales.csv' as input by default
            # and saves reports to sales_forecasting_retail/reports/
            perform_eda() 
            print("Exploratory Data Analysis Complete. Reports saved in sales_forecasting_retail/reports/")
        except Exception as e:
            print(f"Error in Step 2 (Exploratory Data Analysis): {e}")
            # This step is not critical for the rest of the pipeline, so don't set pipeline_successful to False
            # pipeline_successful = False 

    # Step 3: Model Training and Evaluation
    if pipeline_successful:
        print("\n--- Step 3: Running Model Training and Walk-Forward Validation ---")
        try:
            # train_and_evaluate_models uses 'sales_forecasting_retail/data/daily_sales_with_features.csv' as input
            # and saves metrics to 'sales_forecasting_retail/reports/model_evaluation_metrics.csv'
            train_and_evaluate_models()
            print("Model Training and Evaluation Complete. Metrics saved in sales_forecasting_retail/reports/")
        except Exception as e:
            print(f"Error in Step 3 (Model Training and Evaluation): {e}")
            pipeline_successful = False

    # Step 4: Model Comparison and Selection
    best_model_name = None
    if pipeline_successful:
        print("\n--- Step 4: Comparing Models and Selecting the Best ---")
        try:
            best_model_name = compare_and_select_model(metrics_file_path=metrics_file, report_file_path=selection_report_file)
            if best_model_name:
                print(f"Best model selected: {best_model_name}")
            else:
                print("Could not determine the best model from compare_and_select_model. Halting forecast generation.")
                pipeline_successful = False
        except Exception as e:
            print(f"Error in Step 4 (Model Comparison and Selection): {e}")
            pipeline_successful = False

    # Step 5: Generate Future Forecasts
    if pipeline_successful and best_model_name:
        print(f"\n--- Step 5: Generating Forecasts with {best_model_name} ---")
        try:
            forecast_periods = 30 # Configurable
            # generate_forecasts uses 'sales_forecasting_retail/data/daily_sales_with_features.csv' by default
            # and saves reports to sales_forecasting_retail/reports/
            generate_forecasts(best_model_name=best_model_name, 
                               data_file_path=featured_data_file, 
                               forecast_periods=forecast_periods,
                               reports_dir=reports_dir) # Pass reports_dir for consistency
            print("Forecasting Complete. Forecasts and plot saved in sales_forecasting_retail/reports/")
        except Exception as e:
            print(f"Error in Step 5 (Generate Future Forecasts): {e}")
            pipeline_successful = False
    elif pipeline_successful and not best_model_name:
        print("Skipping Step 5: Forecast Generation, as no best model was selected.")
        # pipeline_successful might already be False if best_model_name is None due to error in step 4

    if pipeline_successful:
        print("\n--- Sales Forecasting Pipeline Finished Successfully! ---")
    else:
        print("\n--- Sales Forecasting Pipeline Finished With Errors. Please check logs. ---")
