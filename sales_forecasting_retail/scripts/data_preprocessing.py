import pandas as pd

def preprocess_sales_data():
    """
    Loads sales data, preprocesses it, and saves daily sales data.
    """
    input_file_path = 'sales_forecasting_retail/data/shopping_trends.csv'
    output_file_path = 'sales_forecasting_retail/data/daily_sales.csv'

    try:
        # Load the dataset
        df = pd.read_csv(input_file_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file_path}")
        return

    # Check if 'Purchase Date' column exists
    if 'Purchase Date' not in df.columns:
        print("Error: 'Purchase Date' column not found in the input file.")
        return

    # Convert 'Purchase Date' column to datetime objects
    df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])

    # Aggregate 'Purchase Amount (USD)' by 'Purchase Date'
    # Ensure 'Purchase Amount (USD)' column exists
    if 'Purchase Amount (USD)' not in df.columns:
        print("Error: 'Purchase Amount (USD)' column not found.")
        return
        
    daily_sales = df.groupby('Purchase Date')['Purchase Amount (USD)'].sum().rename('Total Sales')

    # Sort the index
    daily_sales = daily_sales.sort_index()

    # Create a complete date range
    if not daily_sales.empty:
        min_date = daily_sales.index.min()
        max_date = daily_sales.index.max()
        complete_date_range = pd.date_range(start=min_date, end=max_date)

        # Reindex daily_sales with the complete date range, filling missing values with 0
        daily_sales = daily_sales.reindex(complete_date_range, fill_value=0)
    else:
        print("No sales data to process after aggregation.")
        # Create an empty series with a date index if daily_sales is empty, to avoid errors during save
        daily_sales = pd.Series(dtype='float64').rename('Total Sales')
        daily_sales.index = pd.to_datetime(daily_sales.index)


    # Save the processed daily_sales time series to CSV
    try:
        daily_sales.to_csv(output_file_path, header=True)
        print(f"Processed data saved to {output_file_path}")
        return output_file_path # Return the path for the next step
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return None


def engineer_features(input_file_path='sales_forecasting_retail/data/daily_sales.csv', output_file_path='sales_forecasting_retail/data/daily_sales_with_features.csv'):
    """
    Loads daily sales data, engineers time-based and lag features, and saves the result.
    """
    try:
        # Load the daily sales data
        # Ensure 'Purchase Date' (which becomes the index) is parsed as dates
        # and 'Total Sales' is numeric (already handled if loaded from daily_sales.csv correctly)
        df = pd.read_csv(input_file_path, index_col=0, parse_dates=True)
        # The column name after loading daily_sales.csv will be 'Total Sales'
        # If it's read directly from a file where it might have a different name, adjust here.
        # For this script, we assume 'Total Sales' is the correct column name.
    except FileNotFoundError:
        print(f"Error: Input file for feature engineering not found at {input_file_path}")
        return
    except Exception as e:
        print(f"Error loading data for feature engineering: {e}")
        return

    if df.empty:
        print("Error: Data loaded for feature engineering is empty.")
        return

    # Ensure the sales column is numeric (it should be, but a check doesn't hurt)
    if 'Total Sales' not in df.columns:
        print("Error: 'Total Sales' column not found in the input for feature engineering.")
        return
    df['Total Sales'] = pd.to_numeric(df['Total Sales'], errors='coerce')


    # --- Time-based Features ---
    print("Engineering time-based features...")
    df['day_of_week'] = df.index.dayofweek  # Monday=0, Sunday=6
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['quarter'] = df.index.quarter
    df['day_of_year'] = df.index.dayofyear
    print("Time-based features created.")

    # --- Lag Features ---
    print("Engineering lag features...")
    sales_col = 'Total Sales' # The column to lag
    lags = [1, 7, 30]
    for lag in lags:
        df[f'sales_lag_{lag}'] = df[sales_col].shift(lag)
    print("Lag features created. NaN values will be present for initial rows.")

    # Save the DataFrame with features
    try:
        df.to_csv(output_file_path, index=True) # index=True to save the date index
        print(f"Data with engineered features saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving data with features: {e}")


if __name__ == "__main__":
    print("--- Starting Data Processing Pipeline ---")

    # Step 1: Preprocess raw sales data
    print("\nStep 1: Running initial data preprocessing...")
    processed_data_path = preprocess_sales_data()
    if processed_data_path:
        print("Initial data preprocessing completed.")

        # Step 2: Engineer features
        print("\nStep 2: Running feature engineering...")
        # The engineer_features function uses its default input_file_path,
        # which should match the output_file_path of preprocess_sales_data.
        engineer_features() 
        print("Feature engineering completed.")
    else:
        print("Initial data preprocessing failed or did not return a path. Skipping feature engineering.")

    print("\n--- Data Processing Pipeline Finished ---")
