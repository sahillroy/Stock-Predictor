import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load model
model = load_model(r"C:\Users\HP\Documents\Project 1\Stock Prediction Model.keras")

# App Header
st.title('📈 Stock Market Predictor')

# Sidebar with stock input and date range
stock = st.sidebar.text_input('Enter Stock Symbol', 'NVDA')
start = st.sidebar.date_input('Start Date', pd.to_datetime('2014-01-01'))
end = st.sidebar.date_input('End Date', pd.to_datetime('2024-12-31'))

# Load data
data = yf.download(stock, start, end)
st.subheader(f'Stock Data for {stock}')
st.write(data)

# Train-test split
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_scale = scaler.fit_transform(data_train)

pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.transform(data_test)

# Moving Averages
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

st.subheader('Price vs Moving Avg. 50')
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50, label='MA50')
plt.plot(data.Close, label='Close Price')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs Moving Avg. 50 vs Moving Avg. 100')
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50, label='MA50')
plt.plot(ma_100, label='MA100')
plt.plot(data.Close, label='Close Price')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs Moving Avg. 100 vs Moving Avg. 200')
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100, label='MA100')
plt.plot(ma_200, label='MA200')
plt.plot(data.Close, label='Close Price')
plt.legend()
st.pyplot(fig3)

# Predict section
x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])
x, y = np.array(x), np.array(y)

predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict * scale
y = y * scale

# Plot predictions
st.subheader('Original vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(y, 'r', label='Original Price')
plt.plot(predict, 'b', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Error Metrics
st.write("**Model Error Metrics:**")
st.write(f"Mean Squared Error: {mean_squared_error(y, predict):.4f}")
st.write(f"Mean Absolute Error: {mean_absolute_error(y, predict):.4f}")

# Future prediction
st.subheader("Future Price Prediction")
future_days = st.slider('Predict next N days:', 1, 30, 5)
future_input = data_test_scale[-100:].tolist()
for _ in range(future_days):
    pred = model.predict(np.array([future_input[-100:]]))
    future_input.append(pred[0])

future_output = np.array(future_input[-future_days:]) * scale
st.line_chart(future_output)

# RSI Indicator
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
rsi = 100 - (100 / (1 + rs))

st.subheader('RSI Indicator')
fig_rsi = plt.figure(figsize=(8, 4))
plt.plot(rsi, label='RSI')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.legend()
st.pyplot(fig_rsi)

# Download data
csv = data.to_csv().encode('utf-8')
st.download_button(
    label="Download Data as CSV",
    data=csv,
    file_name=f'{stock}_data.csv',
    mime='text/csv',
)

# Model info
with st.expander("ℹ️ About the Model"):
    st.write("""
        This is an LSTM-based neural network model trained on historical stock closing prices.
        It uses the past 100 days of stock prices to predict the next day's closing price.
        Note: This tool is for educational purposes and not financial advice.
    """)
