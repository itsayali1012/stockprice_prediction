# Stock Price Prediction using LSTM and Technical Indicators

## 📈 Overview
This project predicts future stock closing prices for selected Indian stocks using an LSTM (Long Short-Term Memory) model, enhanced with technical indicators like RSI, MACD, SMA, EMA, Bollinger Bands, ATR, OBV, and Historical Volatility.

---

##  Project Objectives
- Collect historical stock data using Yahoo Finance API.
- Engineer multiple technical indicators to improve feature richness.
- Train an LSTM deep learning model for time-series prediction.
- Predict future closing prices.
- Visualize actual vs predicted stock prices for performance evaluation.

---

##  Dataset
- **Source:** Yahoo Finance (`yfinance` library)
- **Stocks Used:**
  - RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS
  - LT.NS, KOTAKBANK.NS, HINDUNILVR.NS, SBIN.NS, BAJFINANCE.NS
  - WIPRO.NS, HCLTECH.NS, AXISBANK.NS, ITC.NS
- **Period:** Last 5 years

---

##  Technical Indicators Used
- **RSI (Relative Strength Index)**
- **MACD (Moving Average Convergence Divergence)**
- **SMA (Simple Moving Average)** (10 and 50 days)
- **EMA (Exponential Moving Average)** (10 days)
- **Bollinger Bands** (Upper and Lower)
- **ATR (Average True Range)**
- **OBV (On Balance Volume)**
- **Historical Volatility** (Annualized)

---

##  Model Architecture
- Two LSTM layers (64 units each)
- Dropout layers (rate: 20%) to reduce overfitting
- Dense output layer (1 neuron for regression)
- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam

---

##  Project Workflow
1. Download historical stock data.
2. Calculate technical indicators.
3. Scale features using MinMaxScaler.
4. Create sequences using a 100-day sliding window.
5. Split data into training and testing sets (80/20 split).
6. Train the LSTM model.
7. Predict and plot actual vs predicted closing prices.

---

##  Results
The model effectively predicts closing prices with a strong alignment between actual and predicted values across various selected stocks.

---

##  Future Improvements
1. Hyperparameter tuning for improved accuracy.
2. Integration of news sentiment analysis.
3. Adoption of attention mechanisms for enhanced time-series prediction.

---

##  Requirements
Install required libraries:
```bash
pip install tensorflow scikit-learn pandas numpy yfinance ta matplotlib

