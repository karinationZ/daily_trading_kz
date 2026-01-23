"""
Actual ATR calculates as:
TrueRange = MAX(abs(High - Low), abs(High - Previous Close), abs(Low - Previous Close)) 
TrueRange.mean()
"""

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import pandas as pd

from daily_trading.settings import logger
from daily_trading.brokers.BitUnixClient import BitUnixClient

CURR_DIR = os.path.dirname(os.path.realpath(sys.argv[0]))


@dataclass
class TradingResults:
    potential_profit: float
    potential_loss: float
    position: float


class Calculator:
    def __init__(
        self,
        asset: str,
        level_price: float,
        direction: Literal["long", "short"] = "long",
        risk_reward_ratio: int = 3,
        risk_amount: float = 3,
        perc_of_atr=None,
        entry_point=None,
        perc_of_atr_stop=None,
        custom_sl_point=None,
        data_source: Literal["yfinance", "ccxt"] = "ccxt",
    ):
        self.position_size = None
        self.instrument = asset.upper() + "USDT"
        self.data_source = data_source
        self.cl = BitUnixClient()
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
            _stop_calculated = (self.atr * 10) / 100
            if direction == "long":
                default_calculated_stop_loss_point = self.entry_point - _stop_calculated
            else:
                default_calculated_stop_loss_point = self.entry_point + _stop_calculated

            # 5 MIN candles stop loss
            two_five_min_sizes = self.get_last_five_min_sizes()
            if direction == "long":
                self.custom_sl_point = self.entry_point - two_five_min_sizes
                _stop_five_min = self.entry_point - self.custom_sl_point
            else:
                self.custom_sl_point = self.entry_point + two_five_min_sizes
                _stop_five_min = self.custom_sl_point - self.entry_point

            # Choose between default 10% of ATR and 5 min candles sizes SL
            if self.custom_sl_point < _stop_calculated:
                self.stop_loss_point = self.custom_sl_point
                _stop = _stop_five_min
                logger.info("Using Two 5_min sizes for the stop loss!")
            else:
                self.stop_loss_point = default_calculated_stop_loss_point
                _stop = _stop_calculated
                logger.info("Using Default 10% of ATR calculated stop loss ")

        # Calculate TP, TL, Position Size
        self.take_profit = self.stop * self.risk_reward_ratio
        self.take_limit = (self.entry_point + self.take_profit) if direction == "long" else (self.entry_point - self.take_profit)

        self.position_size = self.risk_amount / _stop

        # Calculate 2XTP
        take_profit_min = self.risk_amount * 2 / self.position_size
        self.take_limit_min = (self.entry_point + take_profit_min) if direction == "long" else (self.entry_point - take_profit_min)

    def get_historical_data(self, time_frame: str, time_delta_days: int = None, limit: int = 500) -> pd.DataFrame:
        """
        Fetch and process historical mgc using BitUnixClient

        Args:
            time_frame (str): Timeframe for the klines (e.g., "1m", "5m", "1h", "1d")
            time_delta_days (int, optional): Number of days to look back
            limit (int, optional): Maximum number of candles to fetch, default 500

        Returns:
            pd.DataFrame: DataFrame with OHLCV mgc
        """
        # Convert time_frame from ccxt format to BitUnix format if needed
        # BitUnix already uses the same format (1m, 5m, 1h, 1d, etc.)
        interval = time_frame

        # Calculate end_time and start_time if time_delta_days is provided
        end_time = None
        if time_delta_days:
            now = datetime.now(timezone.utc)
            start = now - timedelta(days=time_delta_days)
            start_time = int(start.timestamp() * 1000)
        else:
            start_time = None

        # Fetch klines from BitUnix API
        bars = self.cl.get_klines(
            symbol=self.instrument,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
        )

        # Verify mgc is available
        assert bars, f"There is no mgc! BitUnixClient.get_klines() returned empty list for {self.instrument}"

        # Create a list to hold the OHLCV mgc
        ohlcv_data = []

        # Format mgc - BitUnix returns dictionary format whereas ccxt returns arrays
        for bar in bars:
            ohlcv_data.append(
                [
                    int(bar["time"]),  # timestamp
                    float(bar["open"]),
                    float(bar["high"]),
                    float(bar["low"]),
                    float(bar["close"]),
                    float(bar["quoteVol"]),
                ]
            )

        # Skip the last bar if we have more than one (it might be incomplete)
        if len(ohlcv_data) > 1:
            df = pd.DataFrame(
                ohlcv_data[:-1],
                columns=["time", "open", "high", "low", "close", "volume"],
            )
        else:
            df = pd.DataFrame(ohlcv_data, columns=["time", "open", "high", "low", "close", "volume"])

        # Convert timestamp to datetime
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        logger.info(f"Fetched {len(df)} {interval} candles for {self.instrument}")

        return df

    # def calculate_atr(self, time_frame="1d", time_delta=10):
    #     df = self.get_historical_data(time_frame, time_delta + 2, limit=25)
    #     # Reset the index to convert the DatetimeIndex into a column
    #     df = df.reset_index()
    #     # Rename the index column to "time"
    #     df = df.rename(columns={"Date": "time"})
    #     # Exclude the row at index 0
    #     df = df.iloc[1:]
    #
    #     ranges = df["high"] - df["low"]  # TR calculation
    #
    #     # Sort the values and keep track of original indices
    #     idx_val_pairs = sorted(enumerate(ranges), key=lambda x: x[1])
    #     sorted_ranges = np.array([x[1] for x in idx_val_pairs])
    #     sorted_indices = np.array([x[0] for x in idx_val_pairs])
    #
    #     # Find the window with minimum range
    #     min_range = float('inf')
    #     best_group_idx = 0
    #     n = 5
    #     for i in range(len(sorted_ranges) - n + 1):
    #         window = sorted_ranges[i:i + n]
    #         range_window = window[-1] - window[0]
    #
    #         if range_window < min_range:
    #             min_range = range_window
    #             best_group_idx = i
    #
    #     # Get the indices of the selected ranges
    #     df = df.copy()
    #
    #     # Map positional indices to DataFrame index
    #     selected_ranges = sorted_ranges[best_group_idx:best_group_idx + n]
    #     selected_indices = sorted_indices[best_group_idx:best_group_idx + n]
    #     if df.empty:
    #         msg = "NEW Coin? There is NO DATA!"
    #         logger.error(msg)
    #         raise RuntimeError(msg)
    #     actual_indices = df.index[selected_indices]  # Convert positional indices to actual index labels
    #     df.loc[actual_indices, "true_range"] = selected_ranges
    #
    #     # Create a new column marking the selected ranges
    #     df["selected_range"] = False
    #     df.loc[actual_indices, "selected_range"] = True
    #
    #     # Log the time values of the selected range
    #     selected_times = df.loc[actual_indices, "time"].tolist()
    #
    #     # Format the timestamps to display only the date (YYYY-MM-DD)
    #     selected_times_sorted = sorted(selected_times, reverse=True)
    #     formatted_times = "\n".join([time.strftime("%Y-%m-%d") for time in selected_times_sorted])
    #     logger.info(f"\nSelected Candles for ATR calculation:\n{formatted_times}")
    #     self.selected_atr_candles = formatted_times
    #
    #     # Calculate Average True Range (ATR)
    #     atr = float(np.mean(selected_ranges))
    #     formatted_atr = format(atr, '.10f')
    #     formatted_atr_float = float(formatted_atr)
    #     if formatted_atr_float < self.five_min_tr:
    #         logger.info(f"ATR is too small -> {atr} | Using Five MIN ATR -> {self.five_min_tr}")
    #         return round(self.five_min_tr, 10)
    #     logger.info(f"Average True Range (ATR): {atr}")
    #     return formatted_atr_float
    def calculate_atr(self, time_frame="4h", time_delta=5):
        # Get historical mgc
        df = self.get_historical_data(time_frame, time_delta + 20, limit=50)

        # Reset the index to convert the DatetimeIndex into a column
        df = df.reset_index()
        df = df.rename(columns={"Date": "time"})

        # Exclude the current bar (only use completed bars)
        df = df.iloc[1:].reset_index(drop=True)

        # Define parameters (matching Pine Script)
        days1 = time_delta  # Number of bars to use for ATR calculation
        chg1 = 2.0  # Upper deviation threshold (200%)
        chg2 = 0.5  # Lower deviation threshold (50%)

        # Initialize arrays to store selected bars
        selected_highs = []
        selected_lows = []
        selected_ranges = []
        selected_times = []

        # Step 1: Calculate initial average range to establish baseline
        initial_ranges = df["high"].iloc[:days1] - df["low"].iloc[:days1]
        if len(initial_ranges) < days1:
            msg = "Not enough historical mgc for ATR calculation"
            logger.error(msg)
            raise RuntimeError(msg)

        initial_atr = initial_ranges.mean()

        # Step 2: Filter bars based on volatility thresholds
        df["range"] = df["high"] - df["low"]
        df["is_normal"] = (df["range"] >= initial_atr * chg2) & (df["range"] <= initial_atr * chg1)

        # Step 3: Collect exactly days1 "normal" bars, looking back as far as needed
        bars_collected = 0
        i = 0

        while bars_collected < days1 and i < len(df):
            if df["is_normal"].iloc[i]:
                selected_highs.append(df["high"].iloc[i])
                selected_lows.append(df["low"].iloc[i])
                selected_ranges.append(df["range"].iloc[i])
                selected_times.append(df["time"].iloc[i])
                bars_collected += 1
            i += 1

            if i >= len(df) and bars_collected < days1:
                # If we've gone through all bars and still don't have enough,
                # use what we have or fall back to initial calculation
                logger.warning(f"Could only find {bars_collected} 'normal' bars out of requested {days1}")
                if bars_collected == 0:
                    # If no "normal" bars found, use initial calculation
                    selected_ranges = initial_ranges.tolist()[:days1]
                    selected_times = df["time"].iloc[:days1].tolist()
                    logger.warning(f"No 'normal' bars found, using initial {days1} bars")
                    break

        # Step 4: Calculate ATR from selected bars
        atr = sum(selected_ranges) / len(selected_ranges)

        # Format and display selected candles
        selected_times_sorted = sorted(selected_times, reverse=True)
        formatted_times = "\n".join([time.strftime("%Y-%m-%d") if hasattr(time, "strftime") else str(time) for time in selected_times_sorted])
        logger.info(f"\nSelected Candles for ATR calculation:\n{formatted_times}")
        self.selected_atr_candles = formatted_times

        # Format ATR value
        formatted_atr = format(atr, ".10f")
        formatted_atr_float = float(formatted_atr)

        # Check if ATR is too small
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
            logger.warning(f"Not enough mgc: only {len(df)} candles retrieved")
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
            position=self.position_size,
        )

    def current_daily_bar_price_movement(self):
        """
        Calculate the daily price movement as a percentage of ATR
        and determine if we should enter a trade based on that movement.

        Returns:
            tuple: (atr_percentage, should_enter)
        """
        # Get the daily candle mgc
        daily_data = self.cl.get_klines(symbol=self.instrument, interval="1d", limit=1)
        if not daily_data:
            raise Exception(f"{self.instrument} Does Not Have enough Day Data - New Coin??")

        # Get open price from daily candle
        open_price = float(daily_data[0]["open"])

        # Get current price from latest minute candle
        last_min_data = self.cl.get_klines(symbol=self.instrument, interval="1m", limit=1)[0]
        current_price = float(last_min_data["close"])

        # Calculate absolute price movement
        price_movement = abs(current_price - open_price)

        # Calculate what percentage of ATR has been moved
        atr_percentage = round((price_movement * 100) / self.atr, 2)

        # Determine if we should enter the trade (<= 75% of ATR)
        should_enter = atr_percentage <= 75

        logger.info(f"{self.instrument} Daily Movement: {atr_percentage}% of ATR, Enter Trade: {should_enter}")

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
            f"\nATR: {self.atr}"
            f"\nCurrent Bar Price Movement is: {current_bar_atr_percentage}% of ATR"
            f"\nShould Enter: {should_enter}"
            f"\nSelected Candles for ATR calculation:\n{self.selected_atr_candles}"
        )


if __name__ == "__main__":
    c = Calculator(
        asset="haedal",
        direction="long",
        level_price=0.155623,
        risk_reward_ratio=3,
        risk_amount=7,
        perc_of_atr=10,
        perc_of_atr_stop=10,
        custom_sl_point=None,
        entry_point=None,
    )
    c.print_calculations()
