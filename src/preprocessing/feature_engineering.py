import pandas as pd


def calculate_roc(df, periods=1):
    """Calculate Rate of Change (RoC) for the dataframe."""
    return df.pct_change(periods=periods)

def calculate_moving_average(df, window=5):
    """Calculate Moving Average (MA) for the dataframe."""
    return df.rolling(window=window).mean()

def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """Calculate Moving Average Convergence Divergence (MACD) and Signal line."""
    short_ema = df.ewm(span=short_window, adjust=False).mean()
    long_ema = df.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.rolling(window=signal_window).mean()
    return macd, signal
