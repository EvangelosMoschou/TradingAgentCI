import pandas as pd
import yfinance as yf
import pandas_ta as ta

def get_stock_data(ticker="AAPL", start="2010-01-01", end="2024-01-01"):
    """
    Download historical stock data using yfinance.
    """
    print(f"Downloading data for {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    
    # yfinance often returns MultiIndex columns when downloading, let's flatten them if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Rename columns to ensure standard capitalized names
    df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }, inplace=True)
    
    return df

def add_technical_indicators(df):
    """
    Add SMA, RSI, MACD, and Bollinger Bands to the DataFrame.
    """
    print("Calculating technical indicators...")
    df = df.copy()
    
    # 1. Simple Moving Averages
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=50, append=True)
    
    # 2. Relative Strength Index (RSI)
    df.ta.rsi(length=14, append=True)
    
    # 3. MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # 4. Bollinger Bands
    df.ta.bbands(length=20, std=2, append=True)
    
    # Drop rows with NaN values created by moving windows (e.g., first 50 rows)
    df.dropna(inplace=True)
    print(f"Data shape after adding indicators and dropping NaNs: {df.shape}")
    
    return df

if __name__ == "__main__":
    df = get_stock_data()
    df = add_technical_indicators(df)
    df.to_csv("data/AAPL_with_indicators.csv")
    print("Data saved to data/AAPL_with_indicators.csv")
