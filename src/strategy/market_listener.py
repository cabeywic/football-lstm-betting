import os
import json
import time
import logging
import gzip
import queue
import threading
from typing import List, Type, Union
from flumine.markets.market import Market
from flumine import BaseStrategy
from betfairlightweight.resources import MarketBook, RunnerBook
from flumine.streams.marketstream import BaseStream, MarketStream
from flumine.utils import create_short_uuid, file_line_count
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class MarketListener(BaseStrategy):

    """
    Raw streaming market listener, context:

        market_expiration: int, Seconds to wait after market closure before removing files
        remove_file: bool, Remove txt file during cleanup
        remove_gz_file: bool, Remove gz file during cleanup
        force_update: bool, Update zip/closure if update received after closure
        load_market_catalogue: bool, Store marketCatalogue as {marketId}.json
        local_dir: str, Dir to store data
        recorder_id: str, Directory name (defaults to random uuid)
    """

    MARKET_ID_LOOKUP = "id"

    def __init__(self, *args, **kwargs):
        BaseStrategy.__init__(self, *args, **kwargs)
        self._market_expiration = self.context.get("market_expiration", 3600)  # seconds
        self._remove_file = self.context.get("remove_file", False)
        self._remove_gz_file = self.context.get("remove_gz_file", False)
        self._force_update = self.context.get("force_update", True)
        self._load_market_catalogue = self.context.get("load_market_catalogue", True)
        self.local_dir = self.context.get("local_dir", "/tmp")
        self.recorder_id = self.context.get("recorder_id", create_short_uuid())
        self._loaded_markets = []  # list of marketIds
        self._queue = queue.Queue()

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)

    def add(self) -> None:
        self._logger.info("Adding strategy %s with id %s" % (self.name, self.recorder_id))
        # check local dir
        if not os.path.isdir(self.local_dir):
            raise OSError("File dir %s does not exist" % self.local_dir)
        # create sub dir
        directory = os.path.join(self.local_dir, self.recorder_id)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def start(self) -> None:
        # start load processor thread
        threading.Thread(
            name="{0}_load_processor".format(self.name),
            target=self._load_processor,
            daemon=True,
        ).start()

    def process_raw_data(self, clk: str, publish_time: int, data: dict):
        market_id = data.get(self.MARKET_ID_LOOKUP)
        file_directory = os.path.join(self.local_dir, self.recorder_id, market_id)
        with open(file_directory, "a") as f:
            f.write(
                json.dumps(
                    {"op": "mcm", "clk": clk, "pt": publish_time, "mc": [data]},
                    separators=(",", ":"),
                )
                + "\n"
            )

    def process_closed_market(self, market, data: dict) -> None:
        market_id = data.get(self.MARKET_ID_LOOKUP)
        if market_id in self._loaded_markets:
            if self._force_update:
                self._logger.warning(
                    "File: /{0}/{1}/{2} has already been loaded, updating..".format(
                        self.local_dir, self.recorder_id, market_id
                    )
                )
            else:
                return
        else:
            self._loaded_markets.append(market_id)
        self._logger.info("Closing market %s" % market_id)

        file_dir = os.path.join(self.local_dir, self.recorder_id, market_id)
        market_definition = data.get("marketDefinition")

        # check that file actually exists
        if not os.path.isfile(file_dir):
            self._logger.error(
                "File: %s does not exist in /%s/%s/"
                % (self.local_dir, market_id, self.recorder_id)
            )
            return

        # check that file is not empty / 1 line (i.e. the market had already closed on startup)
        line_count = file_line_count(file_dir)
        if line_count == 1:
            self._logger.warning(
                "File: %s contains one line only and will not be loaded (already closed on startup)"
                % file_dir
            )
            return

        self._queue.put((market, file_dir, market_definition))

    def _load_processor(self):
        # process compression/load in thread
        while True:
            market, file_dir, market_definition = self._queue.get(block=True)
            # check file still exists (potential race condition)
            if not os.path.isfile(file_dir):
                self._logger.warning(
                    "File: %s does not exist in %s" % (market.market_id, file_dir)
                )
                continue
            # compress file
            compress_file_dir = self._compress_file(file_dir)
            # core load code
            self._load(market, compress_file_dir, market_definition)
            # clean up
            self._clean_up()

    def _compress_file(self, file_dir: str) -> str:
        """compresses txt file into filename.gz"""
        compressed_file_dir = "{0}.gz".format(file_dir)
        with open(file_dir, "rb") as f:
            with gzip.open(compressed_file_dir, "wb") as compressed_file:
                compressed_file.writelines(f)
        return compressed_file_dir

    def _load(self, market, compress_file_dir: str, market_definition: dict) -> None:
        # store marketCatalogue data `{marketId}.json.gz`
        if market and self._load_market_catalogue:
            if market.market_catalogue is None:
                self._logger.warning(
                    "No marketCatalogue data available for %s" % market.market_id
                )
                return
            market_catalogue_compressed = self._compress_catalogue(
                market.market_catalogue
            )
            # save to file
            file_dir = os.path.join(
                self.local_dir, self.recorder_id, "{0}.json.gz".format(market.market_id)
            )
            with open(file_dir, "wb") as f:
                f.write(market_catalogue_compressed)

    @staticmethod
    def _compress_catalogue(market_catalogue) -> bytes:
        market_catalogue_dumped = market_catalogue.json()
        if isinstance(market_catalogue_dumped, str):
            market_catalogue_dumped = market_catalogue_dumped.encode("utf-8")
        return gzip.compress(market_catalogue_dumped)

    def _clean_up(self) -> None:
        """If gz > market_expiration old remove
        gz and txt file
        """
        directory = os.path.join(self.local_dir, self.recorder_id)
        for file in os.listdir(directory):
            if file.endswith(".gz"):
                gz_path = os.path.join(directory, file)
                file_stats = os.stat(gz_path)
                seconds_since = time.time() - file_stats.st_mtime
                if seconds_since > self._market_expiration:
                    if self._remove_gz_file:
                        self._logger.info(
                            "Removing: %s, age: %ss"
                            % (gz_path, round(seconds_since, 2))
                        )
                        os.remove(gz_path)
                    txt_path = os.path.join(directory, file.split(".gz")[0])
                    if os.path.exists(txt_path) and self._remove_file:
                        file_stats = os.stat(txt_path)
                        seconds_since = time.time() - file_stats.st_mtime
                        if seconds_since > self._market_expiration:
                            self._logger.info(
                                "Removing: %s, age: %ss"
                                % (txt_path, round(seconds_since, 2))
                            )
                            os.remove(txt_path)

    @staticmethod
    def _create_metadata(market_definition: dict) -> dict:
        try:
            del market_definition["runners"]
        except KeyError:
            pass
        return dict([a, str(x)] for a, x in market_definition.items())
    

class LadderView(BaseStrategy):

    def __init__(self, *args, **kwargs):
        BaseStrategy.__init__(self, *args, **kwargs)
        self._ladder_limit = self.context.get("ladder_limit", 7)

    def start(self) -> None:
        fig = plt.figure(figsize=(10, 5))

        self._axes = []
        self._max_depth = []
        self._fig = fig

        print("Starting Ladder View")

    # Prevent looking at markets that are closed
    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        # process_market_book only executed if this returns True
        if market_book.status != "CLOSED":
            return True
    
    def plot_market_depth(self, market: Market, market_book: MarketBook) -> None:
        runners = market_book.runners

        while len(self._axes) > len(runners):
            ax_to_remove = self._axes.pop()
            _ = self._max_depth.pop()
            self._fig.delaxes(ax_to_remove)

        while len(self._axes) < len(runners):
            self._max_depth.append(1000)
            self._axes.append(self._fig.add_subplot(1, len(runners), len(self._axes) + 1))

        print(f"\nMarket: {market_book.market_id}[{market_book.number_of_runners}]")
        print("-" * 30)

        for idx, ax in enumerate(self._axes):
            runner = runners[idx]
            print(f"\nRunner {runner.selection_id} \t | LTP: {runner.last_price_traded}")
            atb = pd.DataFrame(runner.ex.available_to_back).sort_values(by='price', ascending=False).head(self._ladder_limit)
            atb = atb.sort_values(by='price', ascending=True)
            atl = pd.DataFrame(runner.ex.available_to_lay).sort_values(by='price', ascending=True).head(self._ladder_limit)

            self._generate_market_depth_plot(ax, atb, atl, title=runner.selection_id, max_idx=idx)

            atb_df = pd.DataFrame({
                f'B_{price}': [size] for price, size in zip(atb['price'], atb['size'])
            })
            atl_df = pd.DataFrame({
                f'L_{price}': [size] for price, size in zip(atl['price'], atl['size'])
            })
            
            # Display Ladder on Console
            print(pd.concat([atb_df, atl_df], axis=1).to_string(index=False))

        plt.draw()
        plt.pause(0.1)

    def _generate_market_depth_plot(self, ax, bids, asks, title="Market Depth", max_idx=None):       
        bids = bids.sort_values(by='price', ascending=False) 

        ax.clear()
        bid_prices, bid_sizes = bids["price"].values, bids["size"].values  # bids should be sorted in descending order
        ask_prices, ask_sizes = asks["price"].values, asks["size"].values  # asks should be sorted in ascending order

        # Calculate the cumulative size
        bid_sizes_cum = np.cumsum(bid_sizes)
        ask_sizes_cum = np.cumsum(ask_sizes)

        if max_idx is None:
            max_depth = 1000
        else:
            max_depth = max(self._max_depth[max_idx], np.max(bid_sizes_cum), np.max(ask_sizes_cum))
            self._max_depth[max_idx] = max_depth

        # Plot
        ax.step(bid_prices, bid_sizes_cum, label="Bids", color='green')
        ax.step(ask_prices, ask_sizes_cum, label="Asks", color='red')

        ax.set_xlabel('Price')
        ax.set_ylabel('Size')
        ax.set_title(title)
        ax.legend()
        ax.set_ylim([0, max_depth])

    # If check_market_book returns true i.e. the market is open and not closed then we will run process_market_book once initially
    #  After the first initial time, process_market_book runs every single time someone places, updates or cancels a bet
    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        self.plot_market_depth(market, market_book)
