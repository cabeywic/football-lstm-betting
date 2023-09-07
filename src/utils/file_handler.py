import os
import betfairutil

class BetfairFileHandler:

    @staticmethod
    def find_brz_files_in_directory(directory_path):
        """
        Recursively find all *.brz files in a directory.
        """
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.bz2'):
                    yield os.path.join(root, file)

    @staticmethod
    def extract_files(input_directory, output_directory, **kwargs):
        """
        Convert .brz price files to .csv.
        """
        for prices_file in BetfairFileHandler.find_brz_files_in_directory(input_directory):
            try:
                market_id = betfairutil.get_market_id_from_string(prices_file)
                path_to_csv_file = os.path.join(output_directory, f"{market_id}.csv")
                
                print(f"Market ID: {market_id} | Path: {prices_file} {path_to_csv_file}")
                betfairutil.prices_file_to_csv_file(prices_file, path_to_csv_file, **kwargs)
            except ValueError as e:
                print(f"Error processing file {prices_file}: {e}")
                continue