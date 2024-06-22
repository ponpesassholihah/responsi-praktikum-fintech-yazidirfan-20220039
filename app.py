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
