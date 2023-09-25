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
from enum import Enum


class TradeSignal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BaseFlatBetting(BaseStrategy):
    def __init__(self, *args, **kwargs):
        BaseStrategy.__init__(self, *args, **kwargs)
        self._order_size = self.context.get("order_size", 5.00)
        self.price_history = {}

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
    
    def start(self) -> None:
        self._logger.info("Adding BaseFlatBetting strategy")

    def check_market_book(self, market: Market, market_book: MarketBook) -> bool:
        if market_book.status != "CLOSED":
            return True
        
    def calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        # Calculate MACD and Signal Line
        short_ema = prices.ewm(span=12, adjust=False).mean()
        long_ema = prices.ewm(span=26, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': histogram
        })
    
    def _get_signal(self, runner: RunnerBook) -> str:
         # Update the price history
        if runner.selection_id not in self.price_history:
            self.price_history[runner.selection_id] = []
        self.price_history[runner.selection_id].append(runner.last_price_traded)

        # Convert to pandas Series for easier calculations
        prices = pd.Series(self.price_history[runner.selection_id])

        if prices.size < 26:
            return TradeSignal.HOLD
        
        # Calculate MACD values
        macd_df = self.calculate_macd(prices)
        
        # Get the last value
        latest_macd = macd_df['MACD'].iloc[-1]
        latest_signal = macd_df['Signal'].iloc[-1]
        
        if latest_macd > latest_signal:
            return TradeSignal.BUY
        elif latest_macd < latest_signal:
            return TradeSignal.SELL
        else:
            return TradeSignal.HOLD

    def process_market_book(self, market: Market, market_book: MarketBook) -> None:
        if market.seconds_to_start < 60 and market_book.inplay == False:
            for runner in market_book.runners:
                signal = self._get_signal(runner)
                if runner.status == "ACTIVE":
                    if signal == TradeSignal.BUY:
                        trade = Trade(
                            market_id=market_book.market_id,
                            selection_id=runner.selection_id,
                            handicap=runner.handicap,
                            strategy=self,
                        )
                        runner.last_price_traded
                        order = trade.create_order(
                            side="BACK", order_type=LimitOrder(price=runner.ex.available_to_back[0]['price'], size=self._order_size)
                        )
                        market.place_order(order)
                    elif signal == TradeSignal.SELL:
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

                    