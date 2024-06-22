import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
import datetime

st.write("""
# Simple Stock Price App

Shown are the stock **closing price** and **volume**.
""")

def user_input_features():
    stock_symbol = st.sidebar.selectbox('Symbol',('BMRI','APLN', 'MNCN', 'BFIN', 'CSAP'))
    date_start = st.sidebar.date_input("Start Date", datetime.date(2015, 5, 31))
    date_end = st.sidebar.date_input("End Date", datetime.date.today())

    tickerData = yf.Ticker(stock_symbol+'.JK')
    tickerDf = tickerData.history(period='1d', start=date_start, end=date_end)
    return tickerDf, stock_symbol

input_df, stock_symbol = user_input_features()

st.line_chart(input_df.Close)
st.line_chart(input_df.Volume)

st.write("""
# Stock Price Prediction

Shown are the stock predictions for the next 20 days.
""")

# Normalisasi data untuk LSTM
min_close = input_df['Close'].min()
max_close = input_df['Close'].max()
input_df['Close_normalized'] = (input_df['Close'] - min_close) / (max_close - min_close)

# Persiapan data untuk LSTM
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Ubah dataframe menjadi numpy array untuk model LSTM
data_for_lstm = input_df['Close_normalized'].values
X, y = prepare_data(data_for_lstm, n_steps)

# Reshape X untuk sesuai dengan format input LSTM (jumlah sampel, jumlah time steps, jumlah fitur)
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Bangun model LSTM
model = Sequential()
model.add(Bidirectional(LSTM(300, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Muat berat model yang telah disimpan sebelumnya
model.load_weights(stock_symbol + ".h5")

# Lakukan prediksi untuk 20 hari ke depan
predictions = []
current_sequence = X[-1]

for _ in range(20):
    # Prediksi harga berikutnya
    next_pred = model.predict(current_sequence.reshape(1, n_steps, n_features), verbose=0)
    # Tambahkan prediksi ke daftar prediksi
    predictions.append(next_pred[0])
    # Update urutan saat ini dengan menambahkan prediksi terbaru
    current_sequence = np.append(current_sequence[1:], next_pred)

# Denormalisasi prediksi untuk mendapatkan harga yang sebenarnya
predicted_prices = (predictions * (max_close - min_close)) + min_close

# Plot hasil prediksi dan harga sebelumnya
plt.figure(figsize=(20, 10))
plt.plot(input_df.index[-80:], input_df['Close'][-80:], label='Previous')
plt.plot(pd.date_range(start=input_df.index[-1], periods=20), predicted_prices, label='Prediction')
plt.ylabel('Price (Rp)', fontsize=15)
plt.xlabel('Days', fontsize=15)
plt.title(stock_symbol + " Stock Prediction", fontsize=20)
plt.legend()
plt.grid()

st.pyplot(plt)
