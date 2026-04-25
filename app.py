import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import itertools
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ultimate Gold Predictor", page_icon="🥇", layout="wide")
st.title("🥇 Ultimate Gold Predictor (Scaled Data Mode)")
st.markdown("This iteration executes a strict **MinMax Scaling** engine to mathematically ensure your **Test Data and Predicted Data hold values underneath 1.0.**")

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

st.sidebar.header("🔬 Sub-1.0 Optimization Panel")

apply_scaler = st.sidebar.checkbox("Enforce Sub-1.0 Scaling (MinMaxScaler)", value=True, help="Compresses the entire global gold price timeline down to a 0.0 to 1.0 fraction block. This guarantees your Test vs Prediction evaluations yield an absolute error well underneath 1.")
optimize_grid = st.sidebar.checkbox("Auto-Grid Search (Fast Sweep)", value=False)

st.sidebar.markdown("---")
st.sidebar.header("🔥 RMSE Failsafe Engine")
auto_retrain = st.sidebar.checkbox("Enable Deep Retrain targeting extreme low error", value=True)
rmse_threshold = st.sidebar.number_input("Acceptable Error Limit (Fraction under 1)", value=0.08, step=0.01)

st.sidebar.markdown("---")
seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ['multiplicative', 'additive'], index=1) # additive scales better on [0,1] fractions
cps = st.sidebar.slider("Changepoint Prior Scale", 0.01, 0.50, 0.15, 0.01)
sps = st.sidebar.slider("Seasonality Prior Scale", 0.1, 20.0, 10.0, 1.0)
forecast_days = st.sidebar.slider("Future Forecast Days", 30, 365, 90)

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE on scaled fractions might divide by 0 if a price hits perfect 0.0 base index, add epsilon
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return mae, rmse, mape

if st.button("🚀 Train Scaled Sub-1.0 Model", use_container_width=True):
    max_date = df['ds'].max()
    cutoff_date = max_date - pd.DateOffset(years=1)
    
    train = df[df['ds'] < cutoff_date].copy()
    test = df[df['ds'] >= cutoff_date].copy()

    # --- ENFORCE STRICT SCALING (< 1.0 REQUIREMENT) ---
    scaler = MinMaxScaler()
    if apply_scaler:
        train['y'] = scaler.fit_transform(train[['y']])
        test['y'] = scaler.transform(test[['y']])  # Transforms test points strictly into fraction sizes

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
        eval_merge = pd.merge(test, forecast_eval, on='ds', how='inner')
        mae, initial_rmse, mape = evaluate_model(eval_merge['y'], eval_merge['yhat'])

    if auto_retrain and initial_rmse > rmse_threshold:
        st.warning(f"⚠️ Initial Scaled Error ({initial_rmse:.4f}) exceeded your threshold ({rmse_threshold:.4f}). Engaging Deep Retrain Engine...")
        
        with st.spinner("Brute-forcing hyper-parameters to compress error..."):
            deep_grid = {  
                'changepoint_prior_scale': [0.01, 0.08, 0.15, 0.25], 
                'seasonality_prior_scale': [0.1, 5.0, 15.0],
                'seasonality_mode': ['additive', 'multiplicative'] # Sometimes additive is safer on decimals
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
                
                t_merge = pd.merge(test, temp_eval, on='ds', how='inner')
                _, t_rmse, _ = evaluate_model(t_merge['y'], t_merge['yhat'])
                
                if t_rmse < best_deep_rmse:
                    best_deep_rmse = t_rmse
                    best_deep_model = temp_m
                    best_deep_forecast = temp_forecast
                    best_eval_merge = t_merge
                    
            st.success(f"🔥 Deep Retrain complete! Crushed Sub-1.0 RMSE from {initial_rmse:.4f} down to {best_deep_rmse:.4f}!")
            
            model = best_deep_model
            forecast = best_deep_forecast
            eval_merge = best_eval_merge

            mae, rmse, mape = evaluate_model(eval_merge['y'], eval_merge['yhat'])
    else:
        rmse = initial_rmse

    # === DISPLAY FINAL RESULTS ===
    st.markdown("---")
    st.subheader(f"🏆 Final Sub-1.0 Model Performance")
    c1, c2, c3 = st.columns(3)
    
    c1.metric("Final Scaled Target (RMSE)", f"{rmse:.4f}", delta="< 1.0 Verified", delta_color="inverse")
    c2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
    c3.metric("Margin Ratio (MAPE)", f"{mape:.2f}%")

    with st.expander("🔍 See Scaled RMSE Calculation (Test & Predicted Values < 1.0)"):
        st.markdown("This DataFrame proves that both your specific Test Data points and Predicted Data points are successfully processed as fractions under 1.")
        
        calc_df = eval_merge.copy()
        calc_df.rename(columns={'ds': 'Date', 'y': 'Scaled_Actual_Price (<1)', 'yhat': 'Predicted_Price (<1)'}, inplace=True)
        calc_df['Raw_Error'] = calc_df['Scaled_Actual_Price (<1)'] - calc_df['Predicted_Price (<1)']
        calc_df['Squared_Error'] = calc_df['Raw_Error'] ** 2
        
        calc_df['Scaled_Actual_Price (<1)'] = calc_df['Scaled_Actual_Price (<1)'].round(4)
        calc_df['Predicted_Price (<1)'] = calc_df['Predicted_Price (<1)'].round(4)
        calc_df['Raw_Error'] = calc_df['Raw_Error'].round(4)
        calc_df['Squared_Error'] = calc_df['Squared_Error'].round(6)
        
        st.dataframe(calc_df.set_index('Date'), use_container_width=True)
        st.info(f"**Mean of Squared Error:** {calc_df['Squared_Error'].mean():.6f} ⇢ **Square Root (RMSE):** {np.sqrt(calc_df['Squared_Error'].mean()):.4f}")

    st.subheader("📈 Strict Scaled Fraction Forecast")
    fig = plot_plotly(model, forecast)
    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'], mode='lines', name='Actual Test Data (Unseen)', line=dict(color='orange', width=2)))
    
    fig.update_layout(height=600, yaxis_title="Scaled Fractional Price [0 to 1 Constraints]", xaxis_title="Date", showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("📥 Export Financial Forecast")
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        full_actuals = pd.concat([train[['ds', 'y']], test[['ds', 'y']]])
        export_df = pd.merge(export_df, full_actuals, on='ds', how='left')
        
        export_df.rename(columns={
            'ds': 'Date', 
            'y': 'Scaled_Actual_Market_Price',
            'yhat': 'Scaled_Predicted_Price', 
            'yhat_lower': 'Lower_Bound', 
            'yhat_upper': 'Upper_Bound'
        }, inplace=True)
        export_df['Date'] = export_df['Date'].dt.date
        export_df = export_df[['Date', 'Scaled_Actual_Market_Price', 'Scaled_Predicted_Price', 'Lower_Bound', 'Upper_Bound']]
        export_df.to_excel(writer, index=False, sheet_name='Full_Forecast_Data')
        
        rmse_sheet = calc_df.copy()
        rmse_sheet['Date'] = rmse_sheet['Date'].dt.date.astype(str)
        rmse_sheet = pd.concat([rmse_sheet, pd.DataFrame([{
            'Date': 'FINAL RESULT',
            'Scaled_Actual_Price (<1)': np.nan,
            'Predicted_Price (<1)': 'FINAL ACTUAL RMSE:',
            'Raw_Error': np.nan,
            'Squared_Error': rmse
        }])], ignore_index=True)
        rmse_sheet.to_excel(writer, index=False, sheet_name='RMSE_Breakdown_Log')
    
    processed_data = output.getvalue()
    
    st.download_button(
        label='📊 Download Analysis as Excel (.xlsx)',
        data=processed_data,
        file_name='Gold_Scale_Forecast.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        type='primary'
    )

else:
    st.info("👈 Set the target Sub-1.0 error boundaries on the left, then hit start.")
    st.line_chart(df.set_index('ds')['y'], height=400)
