import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import base64

# Function to load the image and convert it to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to the locally stored QR code image
qr_code_path = "qrcode.png"  # Ensure the image is in your app directory

# Convert image to base64
qr_code_base64 = get_base64_of_bin_file(qr_code_path)

# Custom CSS to position the QR code close to the top-right corner under the "Deploy" area
st.markdown(
    f"""
    <style>
    .qr-code {{
        position: fixed;  /* Keeps the QR code fixed in the viewport */
        top: 10px;       /* Sets the distance from the top of the viewport */
        right: 10px;     /* Sets the distance from the right of the viewport */
        width: 200px;    /* Adjusts the width of the QR code */
        z-index: 100;    /* Ensures the QR code stays above other elements */
    }}
    </style>
    <img src="data:image/png;base64,{qr_code_base64}" class="qr-code">
    """,
    unsafe_allow_html=True
)


# Define top 50 stocks list
top_50_stocks = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA',
    # ... add other stocks here
]

# Function to calculate VROC
def calculate_vroc(data, period=5):
    data['VROC'] = data['Volume'].pct_change(periods=period) * 100
    return data

# Function to calculate RSI
def calculate_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Function for fetching and cleaning stock data
def get_clean_financial_data(ticker, start_date, end_date):
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Clean structure
    data.columns = data.columns.get_level_values(0)

    # Handle missing values
    data = data.ffill()

    # Standardize timezone
    data.index = data.index.tz_localize(None)

    return data

# Backtesting function
def backtest_strategy(data, vroc_buy, vroc_sell, rsi_buy, rsi_sell, initial_investment=10000, transaction_cost_percentage=0.001):
    data['Signal'] = 0
    data.loc[(data['VROC'] > vroc_buy) & (data['RSI'] < rsi_buy), 'Signal'] = 1  # Buy signal
    data.loc[(data['VROC'] < vroc_sell) & (data['RSI'] > rsi_sell), 'Signal'] = -1  # Sell signal

    # Initialize portfolio values and share count
    total_cash = initial_investment  # Initialize total cash
    shares = 0

    # Transactions list
    transactions = []

    # Loop through data to calculate transactions and capital change
    for index, row in data.iterrows():
        if pd.isna(row['Signal']):  # Skip if Signal is NaN
            continue

        if row['Signal'] == 1:  # Buy signal
            if shares == 0:  # Only buy if no shares are held
                shares_to_buy = total_cash // row['Close']
                transaction_cost = shares_to_buy * row['Close'] * transaction_cost_percentage  # Calculate transaction cost
                total_cost = shares_to_buy * row['Close'] + transaction_cost  # Total cost including transaction cost
                
                if total_cost <= total_cash:
                    total_cash -= total_cost
                    shares += shares_to_buy
                    transactions.append({
                        'Date': index,
                        'Transaction': 'Buy',
                        'Shares': shares_to_buy,
                        'Price': row['Close'],
                        'Transaction Cost': -transaction_cost,  # Ensure transaction cost is negative
                        'Cash Left': total_cash,  # Total cash after the transaction
                    })

        elif row['Signal'] == -1 and shares > 0:  # Sell signal
            selling_price = row['Close']
            transaction_cost = shares * selling_price * transaction_cost_percentage  # Calculate transaction cost
            total_income = shares * selling_price - transaction_cost  # Total income after transaction cost
            
            total_cash += total_income
            transactions.append({
                'Date': index,
                'Transaction': 'Sell',
                'Shares': shares,
                'Price': selling_price,
                'Transaction Cost': -transaction_cost,  # Ensure transaction cost is negative
                'Cash Left': total_cash,  # Total cash after the transaction
            })
            shares = 0  # Reset shares after selling

    # Convert transactions list to DataFrame
    transactions_df = pd.DataFrame(transactions)

    # Ensure last transaction is a sell transaction
    if not transactions_df.empty and transactions_df.iloc[-1]['Transaction'] == 'Buy':
        last_buy_price = transactions_df.iloc[-1]['Price']
        last_buy_date = transactions_df.iloc[-1]['Date']
        # Sell at the last price in the dataset
        last_selling_price = data['Close'].iloc[-1]
        transaction_cost = shares * last_selling_price * transaction_cost_percentage
        total_income = shares * last_selling_price - transaction_cost

        total_cash += total_income
        last_sell_transaction = pd.DataFrame([{
            'Date': last_buy_date,  # Use the same date as the last buy
            'Transaction': 'Sell',
            'Shares': shares,
            'Price': last_selling_price,
            'Transaction Cost': -transaction_cost,  # Ensure transaction cost is negative
            'Cash Left': total_cash,  # Total cash after the transaction
        }])

        transactions_df = pd.concat([transactions_df, last_sell_transaction], ignore_index=True)

    # Filter completed trades
    completed_trades = transactions_df[transactions_df['Transaction'].isin(['Buy', 'Sell'])]
    
    # Calculate total strategy return based on last sell transaction
    total_return_percentage = 0
    if not completed_trades.empty and completed_trades.iloc[-1]['Transaction'] == 'Sell':
        last_sell_value = completed_trades.iloc[-1]['Cash Left']
        total_return_percentage = ((last_sell_value - initial_investment) / initial_investment) * 100

    # Calculate winning and losing trades
    winning_trades = 0
    losing_trades = 0

    for i in range(1, len(completed_trades)):
        if completed_trades.iloc[i]['Transaction'] == 'Sell':
            buy_price = completed_trades.iloc[i - 1]['Price']
            sell_price = completed_trades.iloc[i]['Price']
            if sell_price > buy_price:
                winning_trades += 1
            else:
                losing_trades += 1

    total_trades = winning_trades + losing_trades
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    return data, total_return_percentage, win_rate, transactions_df

# Streamlit App
st.title('VROC and RSI Trading Dashboard with Backtesting')

selected_stock = st.selectbox('Select a stock:', top_50_stocks)

# Set date range for historical data
start_date = st.date_input('Start Date', datetime.today() - timedelta(days=365))
end_date = st.date_input('End Date', datetime.today())

# Fetch and process stock data
data = get_clean_financial_data(selected_stock, start_date, end_date)
data = calculate_vroc(data)
data = calculate_rsi(data)

# Set trading thresholds
vroc_buy_threshold = st.slider('VROC Buy Threshold', -100, 100, 20)
vroc_sell_threshold = st.slider('VROC Sell Threshold', -100, 100, -20)
rsi_buy_threshold = st.slider('RSI Buy Threshold', 0, 100, 30)
rsi_sell_threshold = st.slider('RSI Sell Threshold', 0, 100, 70)

# Perform backtesting
data, total_return, win_rate, transactions = backtest_strategy(
    data, vroc_buy_threshold, vroc_sell_threshold, rsi_buy_threshold, rsi_sell_threshold
)

# Display backtesting results
st.write(f"Total Strategy Return: {total_return:.2f}%")
st.write(f"Win Rate: {win_rate:.2f}%")

# Display completed trades
st.subheader("Completed Trades Summary")
st.write(transactions[['Date', 'Transaction', 'Shares', 'Price', 'Transaction Cost', 'Cash Left']])

# Plotting
fig, ax1 = plt.subplots(figsize=(14, 6))

# Price and signals plot
ax1.plot(data['Close'], label='Close Price', color='blue')
ax1.set_title(f'{selected_stock} Price and Strategy Signals')

# Buy signals
buy_signals = data[data['Signal'] == 1]
ax1.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', marker='^', color='green', s=100)

# Sell signals
sell_signals = data[data['Signal'] == -1]
ax1.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', marker='v', color='red', s=100)

ax1.legend()

st.pyplot(fig)
