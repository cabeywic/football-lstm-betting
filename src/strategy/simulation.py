import glob
import os
import math
from utils.stratergy_logging_control import BacktestLoggingControl
from strategy.lstm_stratergy import ModelFlatBetting
from flumine import FlumineSimulation, BaseStrategy, utils, clients
from concurrent import futures


def run_sim_process(strategy: BaseStrategy, file_name: str = "sim_strategy.csv"):
    """Replays a Betfair historic data. Places bets according to the user defined strategy and tries to accurately simulate matching by replaying the historic data.

    Args:
        strategy (BaseStrategy): Our strategy to run on the simulated markets
    """    
    # Set Flumine to simulation mode
    client = clients.SimulatedClient()
    framework = FlumineSimulation(client=client)
    
    framework.add_strategy(strategy)
    framework.add_logging_control(
        BacktestLoggingControl(
            context = {
                "file_name": file_name
            }
        )
    )
    framework.run()

def run_simulation(data_folder_loc: str):
    data_files = os.listdir(data_folder_loc,)
    data_files = [f'{data_folder_loc}/{path}' for path in data_files]

    all_markets = data_files  # All the markets we want to simulate
    processes = os.cpu_count()  # Returns the number of CPUs in the system.
    markets_per_process = 8   # 8 is optimal as it prevents data leakage.

    _process_jobs = []
    with futures.ProcessPoolExecutor(max_workers=processes) as p:
        # Number of chunks to split the process into depends on the number of markets we want to process and number of CPUs we have.
        chunk = min(
            markets_per_process, math.ceil(len(all_markets) / processes)
        )
        # Split all the markets we want to process into chunks to run on separate CPUs and then run them on the separate CPUs
        for markets in (utils.chunks(all_markets, chunk)):
            # Set parameters for our strategy
            strategy = ModelFlatBetting(
                market_filter={
                    "markets": markets,  
                    'market_types':['WIN'],
                    "listener_kwargs": {"inplay": False, "seconds_to_start": 80},  
                    },
                max_order_exposure=1000,
                max_selection_exposure=1000,
                max_live_trade_count=1,
                max_trade_count=1,
            )

            _process_jobs.append(
                p.submit(
                    run_sim_process,
                    strategy=strategy,
                )
            )
        for job in futures.as_completed(_process_jobs):
            job.result()  # wait for result