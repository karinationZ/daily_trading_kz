"""
Actual ATR calculates as:
TrueRange = MAX(abs(High - Low), abs(High - Previous Close), abs(Low - Previous Close)) 
TrueRange.mean()
"""
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf

from settings import logger

CURR_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))
CACHE_FILE_PATH = CURR_DIR + "/exchanges_cache.json"

FORBIDDEN_BROKERS = ['alpaca', 'binance', 'binancecoinm', 'binanceusdm', 'bit2c', 'bitbank', 'bitflyer', 'bithumb',
                     'bitpanda', 'bitvavo', 'bl3p', 'blofin', 'btcbox', 'bybit', 'coinbaseinternational', 'coincheck',
                     'coinmetro', 'coinone', 'coinspot', 'currencycom', 'defx', 'ellipx', 'huobijp', 'hyperliquid',
                     'idex', 'independentreserve', 'krakenfutures', 'kucoinfutures', 'kuna', 'latoken', 'lykke',
                     'mercado', 'okcoin', 'onetrading', 'oxfun', 'paradex', 'paymium', 'poloniexfutures', 'tokocrypto',
                     'vertex', 'wavesexchange', 'woo', 'woofipro', 'zaif', 'zonda']


@dataclass
class TradingResults:
    potential_profit: float
    potential_loss: float
    position: float


class Calculator:
    def __init__(self, asset: str, level_price: float, direction: Literal["long", "short"] = "long",
                 risk_reward_ratio: int = 3, risk_amount: float = 3, perc_of_atr=None, entry_point=None,
                 perc_of_atr_stop=None, custom_sl_point=None,
                 data_source: Literal["yfinance", "ccxt"] = "ccxt"):
        self.position_size = None
        self.instrument = asset.upper() + "/USDT"
        self.data_source = data_source
        self.exchange = self.select_exchange()
        self.level_price = level_price
        self.direction = direction
        self.risk_reward_ratio = risk_reward_ratio
        self.risk_amount = risk_amount
        self.perc_of_atr = perc_of_atr
        self.perc_of_atr_stop = perc_of_atr_stop
        self.custom_sl_point = custom_sl_point
        self.selected_atr_candles = None

        # Calculate these once during initialization
        self.five_min_tr = self.five_min_atr()
        self.atr = self.calculate_atr()
        self.stop = self.calculate_stop()
        self.luft = self.stop * 0.2

        # calculate TBX
        if entry_point is None:
            self.entry_point = (self.level_price + self.luft) if direction == "long" else (self.level_price - self.luft)
        else:
            self.entry_point = entry_point

        # Calculate SL
        if self.perc_of_atr_stop:
            _stop = (self.atr * self.perc_of_atr_stop) / 100
            if direction == "long":
                self.stop_loss_point = self.entry_point - _stop
            else:
                self.stop_loss_point = self.entry_point + _stop

        elif self.custom_sl_point:
            self.stop_loss_point = custom_sl_point
            if direction == "long":
                _stop = self.entry_point - self.custom_sl_point
            else:
                _stop = self.custom_sl_point - self.entry_point
        else:
            logger.info("Using Two 5_min sizes for the stop loss! ")
            two_five_min_sizes = self.get_last_five_min_sizes()
            if direction == "long":
                self.custom_sl_point = self.entry_point - two_five_min_sizes
                _stop = self.entry_point - self.custom_sl_point
            else:
                self.custom_sl_point = self.entry_point + two_five_min_sizes
                _stop = self.custom_sl_point - self.entry_point
            self.stop_loss_point = self.custom_sl_point

        # Calculate TP, TL, Position Size
        self.take_profit = self.stop * self.risk_reward_ratio
        self.take_limit = (self.entry_point + self.take_profit) if direction == "long" else (
                self.entry_point - self.take_profit)

        self.position_size = self.risk_amount / _stop

        # Calculate 2XTP
        take_profit_min = self.risk_amount * 2 / self.position_size
        self.take_limit_min = (self.entry_point + take_profit_min) if direction == "long" else (
                self.entry_point - take_profit_min)

    @staticmethod
    def find_forbidden_brokers():
        forbidden_brokers = []
        exchanges = {exchange_name: getattr(ccxt, exchange_name)() for exchange_name in ccxt.exchanges}
        for exchange_name, exchange in exchanges.items():
            try:
                exchange.fetch_ticker("BTC/USDT")
            except Exception as e:
                logger.info(e)
                forbidden_brokers.append(exchange_name)
        return forbidden_brokers

    def lookup_broker_cache(self):
        with open(CACHE_FILE_PATH, "r") as file:
            cache = json.load(file)
        for broker, instruments in cache.items():
            if self.instrument in instruments:
                logger.info(f"Instrument {self.instrument} is in the cache.\nBroker: {broker}")
                return broker
        return None

    def cache_instrument(self, broker_name):
        with open(CACHE_FILE_PATH, "r") as file:
            cache = json.load(file)
        cache[broker_name].append(self.instrument)
        # Save the updated dictionary back to the file
        with open(CACHE_FILE_PATH, 'w') as file:
            json.dump(cache, indent=2, fp=file)
        logger.info(f"Cached {self.instrument}")

    def select_exchange(self):
        broker = self.lookup_broker_cache()
        exchanges = {exchange_name: getattr(ccxt, exchange_name)() for exchange_name in ccxt.exchanges}

        if broker is None:
            if self.data_source == "ccxt":
                for exchange_name, exchange in exchanges.items():
                    if exchange_name not in FORBIDDEN_BROKERS:
                        try:
                            # Modify to use futures market instead of spot
                            exchange.has["swap"] = True  # Set to swap/futures market
                            exchange.has["spot"] = False
                            exchange.fetch_ohlcv(
                                self.instrument,
                                timeframe="5m",
                                limit=1
                            )
                            logger.info(f"Broker: {exchange_name}")
                            self.cache_instrument(exchange_name)
                            return exchange
                        except Exception as e:
                            print(f"Error fetching markets from {exchange_name}: {e}")
                            continue
                raise Exception(f"Instrument '{self.instrument}' is not supported on any exchange")
        return exchanges[broker]

    def get_ccxt_data(self, time_frame: str, time_delta_days: int = None, limit: int = None) -> pd.DataFrame:
        """Fetch and process historical data more efficiently"""
        since = None
        if time_delta_days:
            end = datetime.today()
            start = end - timedelta(days=time_delta_days)
            since = int(start.timestamp() * 1000)
        bars = self.exchange.fetch_ohlcv(
            self.instrument,
            timeframe=time_frame,
            since=since,
            limit=limit
        )
        assert bars, "There is no data! exchange.fetch_ohlcv() returned []"
        if len(bars) == 1:
            df = pd.DataFrame(
                bars,
                columns=["time", "open", "high", "low", "close", "volume"]
            )
        else:
            # Create DataFrame directly with required columns
            df = pd.DataFrame(
                bars[:-1],
                columns=["time", "open", "high", "low", "close", "volume"]
            )
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df

    def get_yfinance_data(self, time_frame: str, time_delta_days: int, limit: int = None) -> pd.DataFrame:
        """Fetch historical data using Yahoo Finance."""
        end_date = datetime.today()
        start_date = end_date - timedelta(days=time_delta_days) if time_delta_days else None

        ticker = yf.Ticker(self.instrument)
        df = ticker.history(start=start_date, end=end_date, interval=time_frame)
        if limit:
            df = df.tail(limit)
        return df

    def get_historical_data(self, time_frame: str, time_delta_days: int = None, limit: int = None) -> pd.DataFrame:
        """Fetch and process historical data"""
        if self.data_source == "yfinance":
            df = self.get_yfinance_data(time_frame, time_delta_days, limit)
        elif self.data_source == "ccxt":
            df = self.get_ccxt_data(time_frame, time_delta_days, limit)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")

        # Standardize column names to lowercase
        df.columns = df.columns.str.lower()
        return df

    def calculate_atr(self, time_frame="1d", time_delta=10):
        df = self.get_historical_data(time_frame, time_delta + 1, limit=25)
        # Reset the index to convert the DatetimeIndex into a column
        df = df.reset_index()
        # Rename the index column to "time"
        df = df.rename(columns={"Date": "time"})

        ranges = df["high"] - df["low"]  # TR calculation

        # Sort the values and keep track of original indices
        idx_val_pairs = sorted(enumerate(ranges), key=lambda x: x[1])
        sorted_ranges = np.array([x[1] for x in idx_val_pairs])
        sorted_indices = np.array([x[0] for x in idx_val_pairs])

        # Find the window with minimum range
        min_range = float('inf')
        best_group_idx = 0
        n = 5
        for i in range(len(sorted_ranges) - n + 1):
            window = sorted_ranges[i:i + n]
            range_window = window[-1] - window[0]

            if range_window < min_range:
                min_range = range_window
                best_group_idx = i

        # Get the indices of the selected ranges
        df = df.copy()

        # Map integer indices to DataFrame index
        selected_ranges = sorted_ranges[best_group_idx:best_group_idx + n]
        selected_indices = sorted_indices[best_group_idx:best_group_idx + n]
        df.loc[df.index[selected_indices], "true_range"] = selected_ranges

        # Create a new column marking the selected ranges
        df["selected_range"] = False
        df.loc[df.index[selected_indices], "selected_range"] = True

        # Log the time values of the selected range
        selected_times = df.loc[selected_indices, "time"].tolist()

        # Format the timestamps to display only the date (YYYY-MM-DD)
        formatted_times = "\n".join([time.strftime("%Y-%m-%d") for time in selected_times])
        logger.info(f"\nSelected Candles for ATR calculation:\n{formatted_times}")
        self.selected_atr_candles = formatted_times

        # Calculate Average True Range (ATR)
        atr = float(np.mean(selected_ranges))
        formatted_atr = format(atr, '.10f')
        formatted_atr_float = float(formatted_atr)
        if formatted_atr_float < self.five_min_tr:
            logger.info(f"ATR is too small -> {atr} | Using Five MIN ATR -> {self.five_min_tr}")
            return round(self.five_min_tr, 10)
        logger.info(f"Average True Range (ATR): {atr}")
        return formatted_atr_float

    def get_last_five_min_sizes(self) -> float:
        """
        Calculate the size of the last two 5-minute candles and return their sum.
        For long positions, uses (high - low) size; for short positions, uses (low - high) size.

        Returns:
            float: Sum of the sizes of the last two 5-minute candles
        """
        # Fetch the last two 5-minute candles
        df = self.get_historical_data(time_frame="5m", limit=3)  # Fetch 3 to ensure we have at least 2 complete candles

        # Make sure we have at least 2 candles
        if len(df) < 2:
            logger.warning(f"Not enough data: only {len(df)} candles retrieved")
            return 0.0

        # Take just the last 2 candles
        df = df.tail(2)

        # Calculate sizes based on direction
        sizes = []
        for index, row in df.iterrows():
            if self.direction == "long":
                size = row["high"] - row["low"]
            else:  # short direction
                size = row["low"] - row["high"]
            sizes.append(float(format(size, ".10f")))

        # Sum the sizes
        total_size = sum(sizes)
        logger.info(f"Last two 5-min candle sizes: {sizes}, Sum: {total_size}")

        return float(format(total_size, ".10f"))

    def five_min_atr(self) -> float:
        """Calculate 5-minute average bar ATR"""
        df = self.get_historical_data(time_frame="5m", time_delta_days=1)
        return float(df["high"].iloc[-1] - df["low"].iloc[-1])

    def calculate_stop(self) -> float:
        """Calculate stop value"""
        stop = (self.atr * self.perc_of_atr) / 100
        logger.info(f"STOP: {stop}")
        return float(format(stop, ".10f"))

    def calculate_potential_profit_and_loss(self) -> TradingResults:
        """Calculate potential profit and loss more efficiently"""
        if self.direction == "long":
            potential_loss = (self.entry_point - self.stop_loss_point) * self.position_size
            potential_profit = (self.take_limit - self.entry_point) * self.position_size
        else:
            potential_loss = (self.stop_loss_point - self.entry_point) * self.position_size
            potential_profit = (self.entry_point - self.take_limit) * self.position_size

        return TradingResults(
            potential_profit=round(potential_profit, 2),
            potential_loss=round(potential_loss, 2),
            position=self.position_size
        )

    def current_daily_bar_price_movement(self):
        if self.data_source == "ccxt":
            ticker = self.exchange.fetch_ticker(self.instrument)
            ohlcv = self.exchange.fetch_ohlcv(self.instrument, '1d', limit=1)[0]
            current_price = ticker['last']
            open_price = ohlcv[1]
        else:
            ticker = yf.Ticker(self.instrument)
            df = ticker.history(period="1d", interval="1d")
            current_price = df["Close"].iloc[-1]
            open_price = df["Open"].iloc[-1]

        # Calculate absolute price movement
        price_movement = abs(current_price - open_price)

        # Calculate what percentage of ATR has been moved
        atr_percentage = round((price_movement * 100) / self.atr, 2)

        # Determine if we should enter the trade (>= 75% of ATR)
        should_enter = atr_percentage <= 75

        return atr_percentage, should_enter

    def print_calculations(self) -> None:
        """Print trading calculations"""
        results = self.calculate_potential_profit_and_loss()
        current_bar_atr_percentage, should_enter = self.current_daily_bar_price_movement()
        logger.info(
            f"***\n"
            f"Asset: {self.instrument}\n"
            f"Level Price: {self.level_price}\n"
            f"Direction: {self.direction}\n"
            f"Risk-Reward Ratio: {self.risk_reward_ratio}\n"
            f"***\n"
            f"ATR: {self.atr}\n"
            f"Stop: {self.stop}\n"
            f"Luft: {self.luft}\n"
            f"Take Profit: {self.take_profit}\n"
            f"***\n"
            f"{self.direction.upper()}\n"
            f"Level: {self.level_price}\n"
            f"***\n"
            f"{self.instrument}\n"
            f"TBX: {round(self.entry_point, 10)}\n"
            f"TP: {round(self.take_limit, 10)}\n"
            f"SL: {round(self.stop_loss_point, 5)}\n"
            f"Position Size: {round(results.position, 2)}\n"
            f"2x1 TP: {round(self.take_limit_min, 10)}\n"
            f"***\n"
            f"Potential Loss: ${results.potential_loss:.2f}\n"
            f"Potential Profit: ${results.potential_profit:.2f}\n"
            f"Position Investment: {self.position_size * self.entry_point}"
            f"\n***"
            f"\nCurrent Bar Price Movement is: {current_bar_atr_percentage}% of ATR"
            f"\nShould Enter: {should_enter}"
            f"\nSelected Candles for ATR calculation:\n{self.selected_atr_candles}"

        )


if __name__ == "__main__":
    c = Calculator(asset="bera",
                   risk_reward_ratio=3,
                   risk_amount=3,
                   perc_of_atr=10,
                   # perc_of_atr_stop=15,
                   direction="short",
                   level_price=6.464,
                   custom_sl_point=6.486,
                   entry_point=None)
    c.print_calculations()
