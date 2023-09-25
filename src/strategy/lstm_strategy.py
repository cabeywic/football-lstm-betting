import glob
import logging
from typing import List, Type, Union
from flumine.streams.marketstream import BaseStream, MarketStream
import pandas as pd
from flumine import FlumineSimulation, BaseStrategy, utils, clients
from flumine.order.trade import Trade
from flumine.order.order import LimitOrder, OrderStatus
from flumine.markets.market import Market
from betfairlightweight.resources import MarketBook, RunnerBook


class ModelFlatBetting(BaseStrategy):
    def __init__(self, *args, **kwargs):
        BaseStrategy.__init__(self, *args, **kwargs)
        self._order_size = self.context.get("order_size", 5.00)

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
    
    def start(self) -> None:
        self._logger.info("Adding strategy %s with id %s" % (self.name, self.recorder_id))

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        if market_book.status != "CLOSED":
            return True
        
    def _get_price(self, market_book: MarketBook, selected_runner: RunnerBook) -> float:
        raise NotImplementedError()

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        # At the 60 second mark:
        if market.seconds_to_start < 60 and market_book.inplay == False:
            # Can't simulate polling API
            # Only use streaming API:
            for runner in market_book.runners:
                model_price = self._get_price(market_book, runner)
                # If best available to back price is > rated price then flat $5 back
                if runner.status == "ACTIVE" and runner.ex.available_to_back[0]['price'] > model_price:
                    trade = Trade(
                    market_id=market_book.market_id,
                    selection_id=runner.selection_id,
                    handicap=runner.handicap,
                    strategy=self,
                    )
                    order = trade.create_order(
                        side="BACK", order_type=LimitOrder(price=runner.ex.available_to_back[0]['price'], size=self._order_size)
                    )
                    market.place_order(order)
                # If best available to lay price is < rated price then flat $5 lay
                if runner.status == "ACTIVE" and runner.ex.available_to_lay[0]['price'] < model_price:
                    trade = Trade(
                    market_id=market_book.market_id,
                    selection_id=runner.selection_id,
                    handicap=runner.handicap,
                    strategy=self,
                    )
                    order = trade.create_order(
                        side="LAY", order_type=LimitOrder(price=runner.ex.available_to_lay[0]['price'], size=self._order_size)
                    )
                    market.place_order(order)
                    