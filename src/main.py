import argparse
import os 
import sys
import betfairutil
from utils.file_handler import BetfairFileHandler


def preprocess_raw_data(input_dir, output_dir):
    handler = BetfairFileHandler()
    handler.extract_files(
        input_dir, 
        output_dir, 
        should_restrict_to_inplay=True, 
        _format=betfairutil.DataFrameFormatEnum.LAST_PRICE_TRADED
    )

def train_model(model_data_path: str, model_path: str):
    raise NotImplementedError()

def evaluate_model(model_data_path: str, model_path: str):
    raise NotImplementedError()

def main():
    parser = argparse.ArgumentParser(description='Football Inplay Betting Model CLI')
    parser.add_argument("--mode", type=str, required=True, choices=['preprocess', 'train'], help="Which mode to run: preprocess or train")
    parser.add_argument("--raw_data_path", type=str, required=True, help='Path to the raw data')
    parser.add_argument('--model_data_path', type=str, required=True, help='Path to the model data')
    parser.add_argument('--model_path', type=str, default=None, help='Path to save/load the model (for training or prediction)')
    args = parser.parse_args()

    if args.mode == 'preprocess':
        # Load and preprocess data
        preprocess_raw_data(args.raw_data_path, args.model_data_path)
    
    elif args.mode == 'train':
        # Load data and train model
        train_model(args.model_data_path, args.model_path)
        # Evaluate model
        evaluate_model(args.model_data_path, args.model_path)

    else:
        print(f"Unknown stage: {args.mode}")
        sys.exit(1)

if __name__ == '__main__':
    main()
