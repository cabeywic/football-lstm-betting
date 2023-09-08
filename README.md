# Football Inplay Betting Predictor

This project aims to predict the inplay betting odds for football matches using deep learning models. The models take the historical odds data from the Betfair exchange as input and predict the last traded prices for the next time interval.

## Features:
- Data preprocessing and feature engineering: Convert raw Betfair data into a format suitable for LSTM and other deep learning models.
- Various deep learning models: LSTM, BiLSTM, GRU, Transformer.
- Evaluation tools: Functions to visualize true vs. predicted values, residual plots, etc.
- Model persistence: Save and load trained models for future use.

## Repository Structure:
```
football-lstm-betting/
│
├── data/
│   ├── raw/                 # For storing raw data files
│   ├── processed/           # For storing processed data like CSVs
│   └── models/              # For storing trained models
│
├── src/
│   ├── utils/              # Utility functions for betfair (like the given 'betfairutil')
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_preprocessor.py  # Contains 'preprocess_market_data' and related functions
│   │   └── feature_engineering.py # Contains functions like 'calculate_roc', 'calculate_macd' etc.
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py   # Contains LSTM model definition
│   │   ├── training.py     # Contains training functions
│   │   └── evaluation.py   # Contains evaluation functions and metrics
│   │
│   └── main.py             # Main script to tie everything together
│
├── notebooks/              # For Jupyter notebooks if you use them for analysis or prototyping
│
├── requirements.txt        # Required libraries and their versions
│
└── README.md               # Project description, setup instructions, etc.

```

## Getting Started:

### 1. Data Extraction:
Use the `BetfairFileHandler` class to extract and convert raw Betfair `.brz` files into `.csv` format.

### 2. Data Preprocessing:
Utilize the provided preprocessing functions to clean and engineer features from the raw data. The main steps include:
- Resampling the data to fixed time intervals.
- Calculating synthetic features like Rate of Change, Moving Average, and MACD.
- Removing outliers.

### 3. Model Training:
Train one of the provided deep learning models using the training loop functions. Models include LSTM, BiLSTM, GRU, and Transformer.

### 4. Evaluation:
Evaluate the model's predictions using provided visualization tools. This project offers functions to plot true vs. predicted values, residuals, and histograms of residuals.

### 5. Model Persistence:
Save trained models using the model persistence utility and load them back when needed.

## Dependencies:
List all the main libraries and versions you used, e.g.,
- PyTorch (1.8.0)
- Pandas (1.2.4)
- NumPy (1.20.1)
- Seaborn (0.11.1)
- TQDM (4.59.0)

## Future Work:
- Incorporate more data sources.
- Explore other deep learning architectures and ensemble methods.
- Integrate with real-time betting platforms for automated betting.
