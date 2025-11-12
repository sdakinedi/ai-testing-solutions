# filename: fetch_nvda_prices.py

import yfinance as yf
import pandas as pd

# Define the stock ticker and dates
ticker = 'NVDA'
start_date = '2024-03-23'
end_date = '2024-04-23'

# Fetch historical stock price data
data = yf.download(ticker, start=start_date, end=end_date)

# Select only the closing prices
closing_prices = data['Close']

# Print the closing prices
print(closing_data)