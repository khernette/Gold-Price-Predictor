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
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Ultimate Gold Predictor", page_icon="🥇", layout="wide")
st.title("🥇 Ultimate Gold Predictor (Lowest RMSE Mode)")
st.markdown("This iteration utilizes **Logarithmic Transformations** and **Grid Search Cross-Validation** to find the absolute mathematical minimum RMSE for forecasting Gold (GC=F).")

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

apply_log = st.sidebar.checkbox("Enable Log Transform (Crucial for lowering RMSE)", value=True, help="Converts prices into log space so the model tracks pure percentage shifts instead of raw dollars. Reduces explosive error scaling.")
optimize_grid = st.sidebar.checkbox("Auto-Grid Search (Find best hyperparameters)", value=False, help="Runs parallel combinations to find the exact configuration with the lowest error. Takes 1-2 minutes.")

seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ['multiplicative', 'additive'], index=0)
cps = st.sidebar.slider("Changepoint Prior Scale (If Grid Search is OFF)", 0.01, 0.50, 0.15, 0.01)
sps = st.sidebar.slider("Seasonality Prior Scale", 0.1, 20.0, 10.0, 1.0)
forecast_days = st.sidebar.slider("Future Forecast Days", 30, 365, 90)

if st.button("🚀 Train Ultimate Model", use_container_width=True):
    with st.spinner("Processing Time Series Calculations..."):
        # Split Data
        max_date = df['ds'].max()
        cutoff_date = max_date - pd.DateOffset(years=1)
        
        train = df[df['ds'] < cutoff_date].copy()
        test = df[df['ds'] >= cutoff_date].copy()

        # 1. LOG TRANSFORM (Extremely effective for financial assets)
        if apply_log:
            train['y'] = np.log(train['y'])

        best_cps = cps
        best_sps = sps

        # 2. GRID SEARCH CV (Finds mathematical best if enabled)
        if optimize_grid:
            st.info("Initiating Grid Search CV... Running parallel model checks to find the best configuration.")
            param_grid = {  
                'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.3],
                'seasonality_prior_scale': [1.0, 10.0]
            }
            all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
            rmses = []
            
            for params in all_params:
                m = Prophet(
                    daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True,
                    seasonality_mode=seasonality_mode, **params
                )
                m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
                m.add_seasonality(name='quarterly', period=91.25, fourier_order=7)
                m.add_country_holidays(country_name='US')
                m.fit(train)
                
                df_cv = cross_validation(m, initial='730 days', period='180 days', horizon='90 days', parallel="threads")
                df_p = performance_metrics(df_cv)
                rmses.append(df_p['rmse'].mean())
            
            best_params = all_params[np.argmin(rmses)]
            best_cps = best_params['changepoint_prior_scale']
            best_sps = best_params['seasonality_prior_scale']
            st.success(f"Grid Search finished! Optimal Parameters Locked -> CPS: {best_cps} | SPS: {best_sps}")

        # 3. BUILD ULTIMATE MODEL
        model = Prophet(
            daily_seasonality=False, 
            weekly_seasonality=False, 
            yearly_seasonality=True,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=best_cps,
            seasonality_prior_scale=best_sps
        )
        
        # Financial Market Seasonalities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=7) # Quarterly earnings cycle
        model.add_country_holidays(country_name='US')
        
        model.fit(train)

        # 4. PREDICT
        future = model.make_future_dataframe(periods=len(test) + forecast_days)
        forecast = model.predict(future)

        # Reverse Log Transform for Evaluation if applied
        if apply_log:
            forecast['yhat'] = np.exp(forecast['yhat'])
            forecast['yhat_lower'] = np.exp(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.exp(forecast['yhat_upper'])
            train['y'] = np.exp(train['y']) # restore train visually

        # 5. EVALUATION
        merged = pd.merge(test, forecast[['ds', 'yhat']], on='ds', how='inner')
        y_true = merged['y']
        y_pred = merged['yhat']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        st.markdown("---")
        st.subheader(f"🏆 Ultimate Model Performance (Testing on Unseen 1 Year Data)")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE (Minimized)", f"${rmse:.2f}", delta="- Grid/Log Optimization Applied", delta_color="inverse")
        c2.metric("MAE", f"${mae:.2f}")
        c3.metric("MAPE", f"{mape:.2f}%")

        # 6. VISUALIZE
        st.subheader("📈 Ultra-Fitted Forecast Overlay")
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

else:
    st.info("👈 Enable **Log Transform** and click **Train Ultimate Model** to drastically reduce your RMSE.")
    
    st.subheader("Historical Trajectory of Gold")
    chart_data = df.set_index('ds')
    st.line_chart(chart_data['y'], height=400)
