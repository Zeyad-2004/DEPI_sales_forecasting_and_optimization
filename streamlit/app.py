import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Set Streamlit page configurations
st.set_page_config(page_title="Sales Forecasting", layout="centered")
st.title("📈 Sales Forecasting with ARIMA (pmdarima)")

# Load pre-trained model
model = joblib.load("model.pkl")

# User input for number of weeks to predict
future_days = st.slider("Choose the number of Weeks you want to predict", min_value=1, max_value=60, value=10)

# Generate forecast for the selected number of weeks
forecast = model.predict(n_periods=future_days)

# Set the last date manually (as per your code snippet)
last_date = '2015-01-05'
future_dates = pd.date_range(start=last_date, periods=len(forecast), freq='W')

# Create DataFrame for forecasted data
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})
forecast_df.set_index('Date', inplace=True)

# Display forecast plot
st.subheader("📅 التوقع")

# Create plot for forecasted sales only
fig, ax = plt.subplots(figsize=(10, 5))

# Plot forecasted sales (in red dashed line)
ax.plot(forecast_df.index, forecast_df['Forecast'], label="Forecasted Sales", color='red', linestyle='--', linewidth=2)

# Add labels, title, and legend
ax.set_title("Sales Forecast", fontsize=16)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Sales", fontsize=12)
ax.legend(loc='upper left')

# Rotate the x-axis labels to make them vertical
plt.xticks(rotation=45)

# Add gridlines for better readability
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Display the plot in Streamlit
st.pyplot(fig, use_container_width=True)
