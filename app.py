import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objs as go
import warnings

warnings.filterwarnings('ignore')

# ----------------- UI Config ----------------- #
st.set_page_config(page_title="Gold Price Predictor", page_icon="🥇", layout="wide")
st.title("🥇 Advanced Gold Price Predictor")
st.markdown("A time-series forecasting web application utilizing Facebook Prophet, rebuilt to **minimize RMSE** via multiplicative seasonality and optimized flexibility.")

# ----------------- Data Loading ----------------- #
@st.cache_data(ttl=3600)
def load_data(period="5y"):
    """Downloads gold data directly from Yahoo Finance."""
    data = yf.download("GC=F", period=period, progress=False)
    data.reset_index(inplace=True)
    
    # Flatten columns if multi-level
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
        
    df = data[['Date', 'Close']].copy()
    df.columns = ['ds', 'y']
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['y'] = df['y'].ffill() # Forward fill gaps
    df.dropna(inplace=True)
    return df

with st.spinner("Downloading 5-yr market data for Gold (GC=F)..."):
    df = load_data()

# ----------------- Modeling ----------------- #
st.sidebar.header("⚙️ Optimization Parameters")
st.sidebar.markdown("Adjust these to further **reduce RMSE** if needed.")

# For highly volatile assets like Gold, multiplicative is usually better than additive
seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ['multiplicative', 'additive'], index=0)

# Higher CPS = reacts to market trend changes faster (lowers RMSE on test data, but beware overfitting)
cps = st.sidebar.slider("Changepoint Prior Scale (Flexibility)", min_value=0.01, max_value=0.50, value=0.15, step=0.01)

# Larger Seasonality Scale allows deeper reaction to annual/monthly cycles
sps = st.sidebar.slider("Seasonality Prior Scale", 0.1, 20.0, 10.0, 1.0)

# Forecast Length
forecast_days = st.sidebar.slider("Future Forecast Days", 30, 365, 90)

if st.button("🚀 Train Model & Generate Forecast", use_container_width=True):
    with st.spinner("Training Optimized Prophet Model... "):
        max_date = df['ds'].max()
        cutoff_date = max_date - pd.DateOffset(years=1)
        
        train = df[df['ds'] < cutoff_date].copy()
        test = df[df['ds'] >= cutoff_date].copy()

        # Optimized Configuration to significantly reduce RMSE
        model = Prophet(
            daily_seasonality=False,    # Daily noise increases RMSE
            weekly_seasonality=False,   # Markets closed weekends
            yearly_seasonality=True,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=cps,
            seasonality_prior_scale=sps 
        )
        
        # Adding custom monthly seasonality captures mid-term cyclical behaviors in commodities
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        model.add_country_holidays(country_name='US')
        
        model.fit(train)

        # Create Predictions
        future = model.make_future_dataframe(periods=len(test) + forecast_days)
        forecast = model.predict(future)

        # ----------------- Evaluation ----------------- #
        merged = pd.merge(test, forecast[['ds', 'yhat']], on='ds', how='inner')
        y_true = merged['y']
        y_pred = merged['yhat']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        st.markdown("---")
        st.subheader("📊 Model Performance (Test Year)")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE (Root Mean Square Error)", f"${rmse:.2f}", delta="- Optimized", delta_color="inverse")
        c2.metric("MAE (Mean Absolute Error)", f"${mae:.2f}")
        c3.metric("MAPE (Error %)", f"{mape:.2f}%")

        # ----------------- Visualizations ----------------- #
        st.subheader("📈 Interactive Forecast")
        
        fig = plot_plotly(model, forecast)
        # Manually plot the real Test Dataset to see the mapping
        fig.add_trace(go.Scatter(
            x=test['ds'], y=test['y'], 
            mode='lines', name='Actual Test Data (Unseen)', 
            line=dict(color='orange', width=2)
        ))
        # Update layout aesthetic
        fig.update_layout(height=600, yaxis_title="Price (USD)", xaxis_title="Date", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🧩 Under-the-hood Components")
        fig_comp = plot_components_plotly(model, forecast)
        st.plotly_chart(fig_comp, use_container_width=True)

else:
    st.info("👈 Click **Train Model & Generate Forecast** to run the calculations and reveal interactive graphs!")
    
    st.subheader("Historical Data Price Curve")
    chart_data = df.set_index('ds')
    st.line_chart(chart_data['y'], height=400)
    
