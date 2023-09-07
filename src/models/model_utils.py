import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from preprocessing.feature_engineering import calculate_macd, calculate_moving_average, calculate_roc
from preprocessing.data_preprocessor import preprocess_market_data, drop_na_rows, remove_outliers
from .lstm_model import LSTMModel
from sklearn.preprocessing import MinMaxScaler


def normalize_data(X_train, X_val, X_test):
    """
    Normalize features of train, validation, and test data using MinMaxScaler fitted on training data.
    
    Parameters:
    - X_train: Training data.
    - X_val: Validation data.
    - X_test: Test data.
    
    Returns:
    - X_train_normalized: Normalized training data.
    - X_val_normalized: Normalized validation data.
    - X_test_normalized: Normalized test data.
    - scaler: Fitted MinMaxScaler instance.
    """
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Fit the scaler on the training data
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    # Transform the training, validation, and test data
    X_train_normalized = scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_normalized = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_normalized = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    return X_train_normalized, X_val_normalized, X_test_normalized, scaler

def process_multiple_markets(dfs, sequence_length=10, val_ratio=0.15, test_ratio=0.1):
    """
    Process multiple market dataframes to create train, validation, and test sequences and labels for LSTM training.
    
    Parameters:
    - dfs: List of DataFrames, each containing data for a different market.
    - sequence_length: Number of intervals to consider for each sequence.
    - val_ratio: Proportion of data to be used for validation.
    - test_ratio: Proportion of data to be used for testing.
    
    Returns:
    - Train, validation, and test sequences and labels.
    """
    
    train_sequences, val_sequences, test_sequences = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    
    # Process each dataframe individually
    for df in dfs:
        processed_df = preprocess_market_data(df)
        processed_cleaned_df = drop_na_rows(processed_df)

        # Calculate synthetic features
        roc = calculate_roc(processed_cleaned_df)
        ma = calculate_moving_average(processed_cleaned_df)
        macd, signal = calculate_macd(processed_cleaned_df)
        processed_cleaned_df = remove_outliers(processed_cleaned_df)

        # Combine the features into a single dataframe
        features_df = pd.concat([processed_cleaned_df, roc.add_suffix('_roc'), ma.add_suffix('_ma'), macd.add_suffix('_macd'), signal.add_suffix('_signal')], axis=1)
        features_df = drop_na_rows(features_df)
        
        sequences, labels = create_sequences(features_df, sequence_length)

        # Train-Val-Test split for the current market
        # First, separate out the test set
        sequences_train_val, sequences_test, labels_train_val, labels_test = train_test_split(sequences, labels, test_size=test_ratio, shuffle=False)
        
        # Now, split the remaining data into train and validation sets
        sequences_train, sequences_val, labels_train, labels_val = train_test_split(sequences_train_val, labels_train_val, test_size=val_ratio / (1 - test_ratio), shuffle=False)
        
        train_sequences.append(sequences_train)
        val_sequences.append(sequences_val)
        test_sequences.append(sequences_test)
        train_labels.append(labels_train)
        val_labels.append(labels_val)
        test_labels.append(labels_test)
    
    # Concatenate sequences and labels from all dataframes
    X_train = np.concatenate(train_sequences, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    X_val = np.concatenate(val_sequences, axis=0)
    y_val = np.concatenate(val_labels, axis=0)
    X_test = np.concatenate(test_sequences, axis=0)
    y_test = np.concatenate(test_labels, axis=0)
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def create_sequences(data, sequence_length=10):
    """
    Create input sequences and corresponding labels for LSTM training.
    
    Parameters:
    - data: DataFrame containing features and odds for each selection.
    - sequence_length: Number of intervals to consider for each sequence.
    
    Returns:
    - sequences: Numpy array containing input sequences.
    - labels: Numpy array containing corresponding labels.
    """
    
    sequences = []
    labels = []
    
    # Loop through the data to create sequences
    for i in range(sequence_length, len(data)):
        # Extract sequences and corresponding labels
        seq = data.iloc[i-sequence_length:i].values
        label = data.iloc[i, :3].values  # Directly access the odds columns by their index
        sequences.append(seq)
        labels.append(label)
        
    return np.array(sequences), np.array(labels)

def save_model(model, output_path):
    """
    Save the given PyTorch model to the specified output path.

    Parameters:
    - model: PyTorch model to be saved.
    - output_path: Path (including filename) where the model will be saved.
    """
    torch.save(model.state_dict(), output_path)

def load_model(model_path, model_class, *model_args, **model_kwargs):
    """
    Load a PyTorch model from the specified path.

    Parameters:
    - model_path: Path to the saved model weights.
    - model_class: The class of the PyTorch model (a subclass of nn.Module).
    - *model_args and **model_kwargs: Positional and keyword arguments to pass to the model_class when instantiating.

    Returns:
    - model: PyTorch model with loaded weights.
    
    Usage:
    1. For LSTMModel:
    loaded_lstm = load_model("path_to_lstm_weights.pth", LSTMModel, input_dim=15, hidden_dim=50, output_dim=3, num_layers=2)
    
    2. For TransformerModel:
    loaded_transformer = load_model("path_to_transformer_weights.pth", TransformerModel, input_dim=15, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, output_dim=3, dropout=0.1)
    
    3. For any other model derived from nn.Module:
    loaded_model = load_model("path_to_model_weights.pth", YourModelClass, arg1, arg2, ..., kwarg1=val1, kwarg2=val2)
    """
    # Instantiate the model architecture
    model = model_class(*model_args, **model_kwargs)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path))
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    return model

