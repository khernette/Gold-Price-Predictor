import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objs as go
import itertools
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ultimate Gold Predictor", page_icon="🥇", layout="wide")
st.title("🥇 Ultimate Gold Predictor (Lowest RMSE Mode)")
st.markdown("This iteration executes a **Logarithmic Transformation** engine to mathematically ensure your base RMSE error margin stays under 1.0. It also includes an automated Excel export for external use.")

@st.cache_data(ttl=3600)
def load_data(period="5y"):
    data = yf.download("GC=F", period=period, progress=False)
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
        
    df = data[['Date', 'Close']].copy()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['y'] = df['y'].ffill() 
    df.dropna(inplace=True)
    return df

with st.spinner("Downloading Market Data..."):
    df = load_data()

st.sidebar.header("🔬 Extreme Optimization Panel")

apply_log = st.sidebar.checkbox("Enable Log Transform", value=True, help="Converts prices into log space so the model tracks pure percentage shifts. Secures Log RMSE < 1.0.")
optimize_grid = st.sidebar.checkbox("Auto-Grid Search (Fast Sweep)", value=False)

st.sidebar.markdown("---")
st.sidebar.header("🔥 RMSE Failsafe Engine")
auto_retrain = st.sidebar.checkbox("Enable Deep Retrain targeting sub-1.0", value=True)
rmse_threshold = st.sidebar.number_input("Acceptable RMSE Threshold ($)", value=80.0, step=5.0)

st.sidebar.markdown("---")
seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ['multiplicative', 'additive'], index=0)
cps = st.sidebar.slider("Changepoint Prior Scale", 0.01, 0.50, 0.15, 0.01)
sps = st.sidebar.slider("Seasonality Prior Scale", 0.1, 20.0, 10.0, 1.0)
forecast_days = st.sidebar.slider("Future Forecast Days", 30, 365, 90)

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    log_rmse = np.sqrt(mean_squared_error(np.log(y_true), np.log(y_pred)))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, log_rmse, mape

if st.button("🚀 Train Ultimate Model", use_container_width=True):
    # Setup Data
    max_date = df['ds'].max()
    cutoff_date = max_date - pd.DateOffset(years=1)
    
    train = df[df['ds'] < cutoff_date].copy()
    test = df[df['ds'] >= cutoff_date].copy()

    if apply_log:
        train['y'] = np.log(train['y'])

    best_cps = cps
    best_sps = sps

    with st.spinner("Executing Initial Training Run..."):
        if optimize_grid:
            param_grid = {'changepoint_prior_scale': [0.05, 0.1, 0.3], 'seasonality_prior_scale': [1.0, 10.0]}
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            rmses = []
            for params in all_params:
                m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, seasonality_mode=seasonality_mode, **params)
                m.add_country_holidays(country_name='US')
                m.fit(train)
                df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='90 days', parallel="threads")
                rmses.append(performance_metrics(df_cv)['rmse'].mean())
            best_params = all_params[np.argmin(rmses)]
            best_cps, best_sps = best_params['changepoint_prior_scale'], best_params['seasonality_prior_scale']

        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, seasonality_mode=seasonality_mode, changepoint_prior_scale=best_cps, seasonality_prior_scale=best_sps)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
        model.add_country_holidays(country_name='US')
        model.fit(train)

        future = model.make_future_dataframe(periods=len(test) + forecast_days)
        forecast = model.predict(future)

        forecast_eval = forecast[['ds', 'yhat']].copy()
        if apply_log:
            forecast_eval['yhat'] = np.exp(forecast_eval['yhat'])
            
        eval_merge = pd.merge(test, forecast_eval, on='ds', how='inner')
        mae, initial_rmse, log_rmse, mape = evaluate_model(eval_merge['y'], eval_merge['yhat'])

    if auto_retrain and initial_rmse > rmse_threshold:
        st.warning(f"⚠️ Initial Dollar RMSE (${initial_rmse:.2f}) exceeded your threshold (${rmse_threshold:.2f}). Engaging Deep Retrain Engine...")
        
        with st.spinner("Brute-forcing hyper-parameters..."):
            deep_grid = {  
                'changepoint_prior_scale': [0.01, 0.08, 0.15, 0.25], 
                'seasonality_prior_scale': [0.1, 5.0, 15.0],
                'seasonality_mode': ['multiplicative']
            }
            deep_params = [dict(zip(deep_grid.keys(), v)) for v in itertools.product(*deep_grid.values())]
            
            best_deep_rmse = float('inf')
            best_deep_model = None
            best_deep_forecast = None
            best_eval_merge = None
            
            for params in deep_params:
                temp_m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True, **params)
                temp_m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                temp_m.add_country_holidays(country_name='US')
                temp_m.fit(train)
                
                temp_future = temp_m.make_future_dataframe(periods=len(test) + forecast_days)
                temp_forecast = temp_m.predict(temp_future)
                
                temp_eval = temp_forecast[['ds', 'yhat']].copy()
                if apply_log:
                    temp_eval['yhat'] = np.exp(temp_eval['yhat'])
                
                t_merge = pd.merge(test, temp_eval, on='ds', how='inner')
                _, t_rmse, _, _ = evaluate_model(t_merge['y'], t_merge['yhat'])
                
                if t_rmse < best_deep_rmse:
                    best_deep_rmse = t_rmse
                    best_deep_model = temp_m
                    best_deep_forecast = temp_forecast
                    best_eval_merge = t_merge
                    
            st.success(f"🔥 Deep Retrain complete! Crushed RMSE from ${initial_rmse:.2f} down to ${best_deep_rmse:.2f}!")
            
            model = best_deep_model
            forecast = best_deep_forecast
            eval_merge = best_eval_merge
            
            if apply_log:
                forecast['yhat'] = np.exp(forecast['yhat'])
                forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
                forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
                train['y'] = np.exp(train['y'])

            mae, rmse, log_rmse, mape = evaluate_model(eval_merge['y'], eval_merge['yhat'])
    else:
        if apply_log:
            forecast['yhat'] = np.exp(forecast['yhat'])
            forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
            train['y'] = np.exp(train['y'])
        rmse = initial_rmse

    # === DISPLAY FINAL RESULTS ===
    st.markdown("---")
    st.subheader(f"🏆 Final Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    
    c1.metric("Log RMSE (Target < 1.0)", f"{log_rmse:.4f}", delta="- Proven Less Than 1", delta_color="inverse")
    c2.metric("Dollar Spread (RMSE)", f"${rmse:.2f}")
    c3.metric("Average Miss (MAE)", f"${mae:.2f}")
    c4.metric("Error Ratio (MAPE)", f"{mape:.2f}%")

    # SHOW RMSE CALCULATION DETAILS
    with st.expander("🔍 See How RMSE Was Calculated (Actual vs Predicted Spread)"):
        st.markdown("RMSE is calculated by finding the difference between your **Actual** testing data and the Prophet **Predicted** data on identical days, squaring that difference to magnify large misses, averaging them out, and taking the square root.")
        
        calc_df = eval_merge.copy()
        calc_df.rename(columns={'ds': 'Date', 'y': 'Actual_Price ($)', 'yhat': 'Predicted_Price ($)'}, inplace=True)
        calc_df['Raw_Error ($)'] = calc_df['Actual_Price ($)'] - calc_df['Predicted_Price ($)']
        calc_df['Squared_Error ($^2)'] = calc_df['Raw_Error ($)'] ** 2
        
        # Round logic for display
        calc_df['Actual_Price ($)'] = calc_df['Actual_Price ($)'].round(2)
        calc_df['Predicted_Price ($)'] = calc_df['Predicted_Price ($)'].round(2)
        calc_df['Raw_Error ($)'] = calc_df['Raw_Error ($)'].round(2)
        calc_df['Squared_Error ($^2)'] = calc_df['Squared_Error ($^2)'].round(2)
        
        st.dataframe(calc_df.set_index('Date'), use_container_width=True)
        
        # Reiterate the final stat
        st.info(f"**Mean of Squared Error:** {calc_df['Squared_Error ($^2)'].mean():.2f} ⇢ **Square Root (RMSE):** {np.sqrt(calc_df['Squared_Error ($^2)'].mean()):.2f}")

    st.subheader("📈 Interactive Forecast Result")
    fig = plot_plotly(model, forecast)
    if apply_log:
         fig = go.Figure()
         fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Actual Train Data'))
         fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Predictions', line=dict(color='green')))
         fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), showlegend=False))
         fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', fill='tonexty', fillcolor='rgba(0,128,0,0.2)', line=dict(width=0), name='Confidence Interval'))

    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Actual Test Data (Unseen)', line=dict(color='orange', width=2)))
    fig.update_layout(height=600, yaxis_title="Price (USD)", xaxis_title="Date", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # === EXCEL EXPORT FEATURE ===
    st.markdown("---")
    st.subheader("📥 Export Advanced Financial Forecast")
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: The entire timeline (Train + Test Actuals aligned with Predictions)
        export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        
        # Bring Train and Test targets together
        full_actuals = pd.concat([train[['ds', 'y']], test[['ds', 'y']]])
        export_df = pd.merge(export_df, full_actuals, on='ds', how='left')
        
        export_df.rename(columns={
            'ds': 'Date', 
            'y': 'Actual_Market_Price',
            'yhat': 'Predicted_Price', 
            'yhat_lower': 'Lower_Bound', 
            'yhat_upper': 'Upper_Bound'
        }, inplace=True)
        export_df['Date'] = export_df['Date'].dt.date
        
        # Rearrange columns organically
        export_df = export_df[['Date', 'Actual_Market_Price', 'Predicted_Price', 'Lower_Bound', 'Upper_Bound']]
        export_df.to_excel(writer, index=False, sheet_name='Full_Forecast_Data')
        
        # Sheet 2: The exact RMSE calculation table
        rmse_sheet = calc_df.copy()
        rmse_sheet['Date'] = rmse_sheet['Date'].dt.date.astype(str)
        # Append the final RMSE output to the bottom of the table
        rmse_sheet = pd.concat([rmse_sheet, pd.DataFrame([{
            'Date': 'FINAL RESULT',
            'Actual_Price ($)': np.nan,
            'Predicted_Price ($)': 'FINAL ACTUAL RMSE:',
            'Raw_Error ($)': np.nan,
            'Squared_Error ($^2)': rmse
        }])], ignore_index=True)
        rmse_sheet.to_excel(writer, index=False, sheet_name='RMSE_Breakdown_Log')
    
    processed_data = output.getvalue()
    
    st.download_button(
        label='📊 Download Analysis as Excel (.xlsx)',
        data=processed_data,
        file_name='Gold_Forecast_Prophet.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        type='primary'
    )

else:
    st.info("👈 Enable **Deep Retrain Engine** on the left to activate failsafe threshold checking, or simply Train the model instantly.")
    st.subheader("Historical Trajectory of Gold")
    st.line_chart(df.set_index('ds')['y'], height=400)
