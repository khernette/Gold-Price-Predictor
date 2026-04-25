import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def download_data(ticker="GC=F", period="5y"):
    """
    Download historical gold price data from Yahoo Finance.
    XAUUSD=X works for Gold/USD, GC=F works for Gold Futures.
    """
    print(f"[{ticker}] Downloading data for last {period}...")
    data = yf.download(ticker, period=period, progress=False)
    data.reset_index(inplace=True)
    return data

def preprocess_data(data):
    """
    Preprocess data for FB Prophet: Date -> ds, Close -> y
    """
    print("Preprocessing data...")
    # Handle multi-level columns if any (common in latest yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    
    # Keep only Date and Close columns
    df = data[['Date', 'Close']].copy()
    
    # Rename columns to ds and y
    df.columns = ['ds', 'y']
    
    # Ensure ds is proper datetime
    df['ds'] = pd.to_datetime(df['ds'])
    
    # Convert 'y' to float explicitly
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    
    # Handle missing values (forward fill as per typical TS data, drop remaining)
    df['y'] = df['y'].ffill()
    df.dropna(inplace=True)
    
    return df

def split_data(df):
    """
    Splits dataset into 4 years Train and 1 year Test based on dates.
    """
    print("Splitting dataset into Training and Test sets...")
    
    # Identify the cutoff point (1 year before the max date)
    max_date = df['ds'].max()
    cutoff_date = max_date - pd.DateOffset(years=1)
    
    train = df[df['ds'] < cutoff_date].copy()
    test = df[df['ds'] >= cutoff_date].copy()
    
    print(f"  Training shape: {train.shape}")
    print(f"  Test shape: {test.shape}")
    
    return train, test

def auto_tune_prophet(train_df):
    """
    Performs grid search to find the best changepoint_prior_scale.
    Note: Cross-validation takes some time.
    """
    print("\nStarting hyperparameter tuning (cross-validation)...")
    param_grid = {  
        'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5],
    }
    
    best_rmse = float('inf')
    best_cps = 0.05 # Default fallback
    
    for cps in param_grid['changepoint_prior_scale']:
        m = Prophet(daily_seasonality=True, yearly_seasonality=True, changepoint_prior_scale=cps)
        m.add_country_holidays(country_name='US')
        # Suppress prophet logs during grid search
        m.fit(train_df)
        
        # initial training: 730 days (2 yrs), test every 180 days, forecast 90 days.
        try:
             df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='90 days')
             df_p = performance_metrics(df_cv)
             rmse = df_p['rmse'].mean()
             print(f"  changepoint_prior_scale: {cps} -> Mean RMSE: {rmse:.2f}")
             
             if rmse < best_rmse:
                 best_rmse = rmse
                 best_cps = cps
        except Exception as e:
             print(f"  CV failed for cps={cps}")
             continue
            
    print(f"Best changepoint_prior_scale found: {best_cps}")
    return best_cps

def build_and_train_model(train_df, best_cps=0.05):
    """
    Train the final Facebook Prophet model with given hyperparameters.
    """
    print(f"Training Prophet model with changepoint_prior_scale={best_cps} ...")
    model = Prophet(
        daily_seasonality=True, 
        yearly_seasonality=True,
        weekly_seasonality=False, # Gold does not trade on weekends
        changepoint_prior_scale=best_cps
    )
    # Add US Holidays since market closures impact gold trading
    model.add_country_holidays(country_name='US')
    model.fit(train_df)
    
    return model

def evaluate_metrics(test_df, forecast_df):
    """
    Matches the forecasted dates against actuals and calculates MAE, RMSE, MAPE.
    """
    print("\nCalculating Evaluation Metrics...")
    
    # Merge on date to compare identical days
    merged = pd.merge(test_df, forecast_df[['ds', 'yhat']], on='ds', how='inner')
    
    y_true = merged['y']
    y_pred = merged['yhat']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("-" * 30)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print("-" * 30)
    
    return merged

def plot_forecast_and_diagnostics(train_df, test_df, forecast_df, model, merged_df):
    """
    Generate plots:
    1. Training vs Test vs Predictions
    2. Prophet components (Trend, US Holidays, Seasonality)
    3. Residual errors
    """
    print("Generating visualizations...")
    
    # Plot 1: Main Forecast Plot
    plt.figure(figsize=(14, 7))
    plt.plot(train_df['ds'], train_df['y'], label='Training Data (Actuals)')
    plt.plot(test_df['ds'], test_df['y'], label='Test Data (Actuals)', color='orange')
    plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Prophet Predictions', color='green', linewidth=2)
    plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='green', alpha=0.2, label='Confidence Interval')
    
    plt.title('Gold Price Forecast using FB Prophet')
    plt.xlabel('Date')
    plt.ylabel('Gold Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: Components
    fig_comp = model.plot_components(forecast_df)
    plt.show()

    # Plot 3: Residual Errors Map
    merged_df['residual'] = merged_df['y'] - merged_df['yhat']
    plt.figure(figsize=(14, 5))
    plt.plot(merged_df['ds'], merged_df['residual'], label='Residuals (Testing Period)', color='red')
    plt.axhline(0, color='black', linestyle='--')
    plt.title('Residual Errors Over Time')
    plt.xlabel('Date')
    plt.ylabel('Error (Actual - Predicted)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("=== Gold Price Predictor Pipeline ===\n")
    
    # 1. Download
    raw_data = download_data(ticker="GC=F", period="5y")
    
    # 2. Preprocess
    df = preprocess_data(raw_data)
    
    # 3. Train/Test Split
    train_df, test_df = split_data(df)
    
    # Enable hyperparameter tuning? Set to False for faster standard run
    run_tuning = input("\nDo you want to run hyperparameter tuning? (Can take several minutes) [y/N]: ").strip().lower()
    
    if run_tuning == 'y':
        best_cps = auto_tune_prophet(train_df)
    else:
        print("\nSkipping tuning. Using default changepoint_prior_scale=0.05")
        best_cps = 0.05
        
    # 4. Train Model
    model = build_and_train_model(train_df, best_cps=best_cps)
    
    # 5. Predict (Length of test df + 30 days into future)
    future = model.make_future_dataframe(periods=len(test_df) + 30)
    forecast = model.predict(future)
    
    # 6. Evaluation
    merged_results = evaluate_metrics(test_df, forecast)
    
    # 7. Visualizations
    plot_forecast_and_diagnostics(train_df, test_df, forecast, model, merged_results)
    
if __name__ == "__main__":
    main()
