# Install TA-Lib (if not installed)
!pip install ta

# Import libraries
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import ta  # For technical indicators
import matplotlib.pyplot as plt

# List of Features
features = [
    'Close', 'Volume', 'Daily_Returns', 'RSI', 'MACD',
    'SMA_10', 'SMA_50', 'EMA_10',
    'Bollinger_Upper', 'Bollinger_Lower',
    'ATR', 'OBV', 'Historical_Volatility'
]

# Download Data and Create Features
def download_data(stock):
    df = yf.download(stock, period="5y")
    df.dropna(inplace=True)

    # Extract Series
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    volume = df['Volume'].squeeze()

    # Technical Indicators
    df['Daily_Returns'] = close.pct_change()
    df['RSI'] = ta.momentum.RSIIndicator(close=close).rsi()
    df['MACD'] = ta.trend.MACD(close=close).macd()
    df['SMA_10'] = close.rolling(window=10).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['EMA_10'] = close.ewm(span=10, adjust=False).mean()
    bb = ta.volatility.BollingerBands(close=close)
    df['Bollinger_Upper'] = bb.bollinger_hband()
    df['Bollinger_Lower'] = bb.bollinger_lband()
    df['ATR'] = ta.volatility.AverageTrueRange(high=high, low=low, close=close).average_true_range()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()
    df['Historical_Volatility'] = df['Daily_Returns'].rolling(window=21).std() * np.sqrt(252)

    df.dropna(inplace=True)
    return df

# Scale Data
def scale_data(df, features):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])
    return scaler, scaled

# Prepare Sequences
def prepare_sequences(scaled_data, sequence_length=100):
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i][0])  # Target is 'Close'
    return np.array(X), np.array(y)

# Build LSTM Model
def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# List of Indian Stocks
indian_stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
    "ICICIBANK.NS", "LT.NS", "KOTAKBANK.NS", "HINDUNILVR.NS", 
    "SBIN.NS", "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", 
    "AXISBANK.NS", "ITC.NS"
]

# Train model on each stock
for stock in indian_stocks:
    print(f"\nTraining for {stock}")

    # Download and preprocess
    df = download_data(stock)
    scaler, scaled_data = scale_data(df, features)
    X, y = prepare_sequences(scaled_data)

    # Train/Test Split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = build_model((X_train.shape[1], X_train.shape[2]))

    # Train
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Predict
    predictions = model.predict(X_test)

    # Inverse Transform for actual price comparison
    close_scaler = MinMaxScaler()
    close_scaler.min_, close_scaler.scale_ = scaler.min_[0], scaler.scale_[0]
    y_test_actual = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_actual = close_scaler.inverse_transform(predictions)

    # Plot results
    plt.figure(figsize=(12,6))
    plt.plot(y_test_actual, label='Actual Price')
    plt.plot(predictions_actual, label='Predicted Price')
    plt.title(f"{stock} - Actual vs Predicted Closing Price")
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

