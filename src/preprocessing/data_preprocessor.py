import pandas as pd
import numpy as np


def preprocess_market_data(df):
    # Convert publish_time to datetime format
    df['publish_time'] = pd.to_datetime(df['publish_time'], unit='ms')
    
    # Set the datetime as index for resampling
    df.set_index('publish_time', inplace=True)
    
    # Create a pivot table for each selection_id's last traded price and resample it
    df_pivot = df.pivot_table(index='publish_time', columns='selection_id', values='last_price_traded', aggfunc='last')
    
    # Resample at 30-second intervals using forward-fill for missing values
    df_resampled = df_pivot.resample('10S').ffill()
    
    return df_resampled

def drop_na_rows(df):
    """Drop rows with NaN values."""
    return df.dropna()

def remove_outliers(df):
    """Remove rows containing outliers based on the IQR method."""
    # Calculate IQR for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Define bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter rows without outliers
    return df[((df >= lower_bound) & (df <= upper_bound)).all(axis=1)]
