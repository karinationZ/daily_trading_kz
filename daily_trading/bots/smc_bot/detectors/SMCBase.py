from enum import Enum

import numpy as np
import pandas as pd

from daily_trading.settings import logger

GET_READY_ALERT = "ALERT - Get Ready"
BULLISH = 1
BEARISH = 0
NEUTRAL = -1


class BIAS(Enum):
    """Trend enumeration with integer values"""
    SHORT = "SHORT"
    LONG = "LONG"
    NEUTRAL = "NEUTRAL"


class Trend(Enum):
    """Trend enumeration with integer values"""
    BULLISH = 1
    BEARISH = 0
    NEUTRAL = -1


class SMCBase:
    def __init__(self, main_tf, left_bars=None, right_bars=None, plot_tail=100):
        self.main_tf = main_tf
        self.swing_length_map = {
            "15m": 8,
            "1h": 6,
            "4h": 4
        }
        self.length = self.swing_length_map.get(main_tf.lower(), 10)
        self.left_bars = left_bars if left_bars is not None else self.length
        self.right_bars = right_bars if right_bars is not None else self.length
        self.ArrayType = []
        self.ArrayValue = []
        self.ArrayIndex = []
        self.Major_HighLevel = None
        self.Major_LowLevel = None
        self.Major_HighIndex = None
        self.Major_LowIndex = None
        self.Major_HighType = None
        self.Major_LowType = None
        logger.debug(f"Initialized SMCBase with lower_tf={main_tf}, length={self.length}, "
                     f"left_bars={left_bars}, right_bars={right_bars}")
        self.plot_tail = plot_tail

    def calculate_pivots(self, ohlc):
        logger.debug(
            f"Starting pivot calculation with "
            f"left_bars={self.left_bars}, "
            f"right_bars={self.right_bars} on OHLC data with {len(ohlc)} rows")

        if len(ohlc) < self.left_bars + self.right_bars + 1:
            logger.error(f"OHLC data too short ({len(ohlc)} rows) for pivot calculation")
            return [], []

        ohlc = self.validate_data(ohlc)

        # Track all detected pivots before sequencing them
        raw_highs = []
        raw_lows = []

        # First pass: detect all potential pivots
        for i in range(self.left_bars, len(ohlc) - self.right_bars):
            high_value = ohlc['high'].iloc[i]
            low_value = ohlc['low'].iloc[i]
            bar_index = i
            time = ohlc.index[i]
            volume = ohlc['volume'].iloc[i]

            window = ohlc.iloc[i - self.left_bars:i + self.right_bars + 1]
            is_high_pivot = high_value == window['high'].max()
            is_low_pivot = low_value == window['low'].min()

            if is_high_pivot:
                raw_highs.append({
                    "level": high_value,
                    "time": time,
                    "index": bar_index,
                    "volume": volume
                })

            if is_low_pivot:
                raw_lows.append({
                    "level": low_value,
                    "time": time,
                    "index": bar_index,
                    "volume": volume
                })

        # Second pass: sequence the pivots and determine types
        all_pivots = []
        high_ptr = 0
        low_ptr = 0

        while high_ptr < len(raw_highs) and low_ptr < len(raw_lows):
            if raw_highs[high_ptr]['index'] < raw_lows[low_ptr]['index']:
                # Process high pivot
                current_high = raw_highs[high_ptr]

                if not all_pivots:
                    pivot_type = "H"
                else:
                    last_high = next((p for p in reversed(all_pivots) if p['type'] in ['H', 'HH', 'LH']), None)
                    if last_high:
                        pivot_type = "HH" if current_high['level'] > last_high['level'] else "LH"
                    else:
                        last_low = next((p for p in reversed(all_pivots) if p['type'] in ['L', 'LL', 'HL']), None)
                        pivot_type = "H" if last_low else "H"  # First high after series of lows

                current_high['type'] = pivot_type
                current_high['iscrossed'] = False
                all_pivots.append(current_high)
                high_ptr += 1
            else:
                # Process low pivot
                current_low = raw_lows[low_ptr]

                if not all_pivots:
                    pivot_type = "L"
                else:
                    last_low = next((p for p in reversed(all_pivots) if p['type'] in ['L', 'LL', 'HL']), None)
                    if last_low:
                        pivot_type = "LL" if current_low['level'] < last_low['level'] else "HL"
                    else:
                        last_high = next((p for p in reversed(all_pivots) if p['type'] in ['H', 'HH', 'LH']), None)
                        pivot_type = "L" if last_high else "L"  # First low after series of highs

                current_low['type'] = pivot_type
                current_low['iscrossed'] = False
                all_pivots.append(current_low)
                low_ptr += 1

        # Add remaining pivots if any
        while high_ptr < len(raw_highs):
            current_high = raw_highs[high_ptr]
            last_high = next((p for p in reversed(all_pivots) if p['type'] in ['H', 'HH', 'LH']), None)
            pivot_type = "HH" if last_high and current_high['level'] > last_high[
                'level'] else "LH" if last_high else "H"
            current_high['type'] = pivot_type
            current_high['iscrossed'] = False
            all_pivots.append(current_high)
            high_ptr += 1

        while low_ptr < len(raw_lows):
            current_low = raw_lows[low_ptr]
            last_low = next((p for p in reversed(all_pivots) if p['type'] in ['L', 'LL', 'HL']), None)
            pivot_type = "LL" if last_low and current_low['level'] < last_low['level'] else "HL" if last_low else "L"
            current_low['type'] = pivot_type
            current_low['iscrossed'] = False
            all_pivots.append(current_low)
            low_ptr += 1

        # Separate into high and low points
        pivot_high_points = [p for p in all_pivots if p['type'] in ['H', 'HH', 'LH']]
        pivot_low_points = [p for p in all_pivots if p['type'] in ['L', 'LL', 'HL']]

        # Update class arrays
        self.ArrayType = [p['type'] for p in all_pivots]
        self.ArrayValue = [p['level'] for p in all_pivots]
        self.ArrayIndex = [p['index'] for p in all_pivots]

        # Initialize major levels if we have enough pivots
        if len(all_pivots) >= 2:
            # Find the highest high and lowest low among the first few pivots
            initial_highs = [p for p in all_pivots[:10] if p['type'] in ['H', 'HH', 'LH']]
            initial_lows = [p for p in all_pivots[:10] if p['type'] in ['L', 'LL', 'HL']]

            if initial_highs and initial_lows:
                major_high = max(initial_highs, key=lambda x: x['level'])
                major_low = min(initial_lows, key=lambda x: x['level'])

                self.Major_HighLevel = major_high['level']
                self.Major_LowLevel = major_low['level']
                self.Major_HighIndex = major_high['index']
                self.Major_LowIndex = major_low['index']
                self.Major_HighType = major_high['type']
                self.Major_LowType = major_low['type']

        logger.debug(f"Detected {len(pivot_high_points)} swing highs and {len(pivot_low_points)} swing lows")
        return pivot_high_points, pivot_low_points

    def reset_pivot_arrays(self):
        self.ArrayType = []
        self.ArrayValue = []
        self.ArrayIndex = []
        self.Major_HighLevel = None
        self.Major_LowLevel = None
        self.Major_HighIndex = None
        self.Major_LowIndex = None
        self.Major_HighType = None
        self.Major_LowType = None

    @staticmethod
    def validate_data(ohlc):
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in ohlc.columns for col in required_columns):
            raise ValueError(f"OHLC DataFrame must contain columns: {required_columns}")
        if ohlc.empty:
            raise ValueError("OHLC DataFrame is empty")
        return ohlc

    @staticmethod
    def calculate_ema(series, length):
        """
        Calculate EMA to match Pine Script's ta.ema behavior.

        Parameters:
        - series (pd.Series): Input series (e.g., volume).
        - length (int): EMA period.

        Returns:
        - pd.Series: EMA of the input series.
        """
        alpha = 2 / (length + 1)
        ema = np.zeros(len(series))
        ema[0] = series.iloc[0]  # Initialize with first value
        for i in range(1, len(series)):
            ema[i] = alpha * series.iloc[i] + (1 - alpha) * ema[i - 1]
        return pd.Series(ema, index=series.index)

    @staticmethod
    def detect_trend(
            ohlc_data: pd.DataFrame,
            short_ma: int = 5,
            long_ma: int = 20,
            slope_lookback: int = 20,
            slope_threshold_pct: float = 0.001,
            min_confirmation: int = 2
    ) -> str:
        """
        Improved Trend Detection: SMA Crossover + Linear Regression Slope.
        Includes NaN protection and dynamic thresholding.
        """
        if len(ohlc_data) < max(long_ma, slope_lookback) + min_confirmation:
            return Trend.NEUTRAL.name

        # Calculate MAs
        short_sma = ohlc_data['close'].rolling(window=short_ma).mean()
        long_sma = ohlc_data['close'].rolling(window=long_ma).mean()

        # Slope Calculation with NaN Protection
        recent_data = ohlc_data['close'].tail(slope_lookback)
        if not np.isfinite(recent_data).all():
            return Trend.NEUTRAL.name

        x = np.arange(len(recent_data))
        slope, _ = np.polyfit(x, recent_data, 1)

        # Dynamic slope threshold
        mean_close = recent_data.mean()
        slope_threshold = mean_close * slope_threshold_pct

        # MA Alignment Check
        ma_bullish = all(short_sma.iloc[-i] > long_sma.iloc[-i] for i in range(1, min_confirmation + 1))
        ma_bearish = all(short_sma.iloc[-i] < long_sma.iloc[-i] for i in range(1, min_confirmation + 1))

        if ma_bullish and slope > slope_threshold:
            return Trend.BULLISH.name
        elif ma_bearish and slope < -slope_threshold:
            return Trend.BEARISH.name

        return Trend.NEUTRAL.name

    # @staticmethod
    # def detect_trend(
    #         ohlc_data: pd.DataFrame,
    #         short_ma: int = 5,
    #         long_ma: int = 20,
    #         slope_lookback: int = 20,
    #         slope_threshold_pct: float = 0.001,
    #         min_confirmation: int = 2
    # ) -> str:
    #     """
    #     Fixed version of original function with better validation and parameters.
    #     """
    #     # Input validation
    #     if len(ohlc_data) < max(long_ma, slope_lookback) + min_confirmation:
    #         return Trend.NEUTRAL.name
    #
    #     # Calculate moving averages
    #     short_sma = ohlc_data['close'].rolling(window=short_ma).mean()
    #     long_sma = ohlc_data['close'].rolling(window=long_ma).mean()
    #
    #     # Calculate slope
    #     recent_data = ohlc_data['close'].tail(slope_lookback)
    #     x = np.arange(len(recent_data))
    #     slope, _ = np.polyfit(x, recent_data, 1)
    #
    #     # Dynamic slope threshold
    #     mean_close = recent_data.mean()
    #     slope_threshold = mean_close * slope_threshold_pct
    #
    #     # Trend conditions with confirmation
    #     ma_bullish = all(
    #         short_sma.iloc[-i] > long_sma.iloc[-i]
    #         for i in range(1, min_confirmation + 1)
    #     )
    #     ma_bearish = all(
    #         short_sma.iloc[-i] < long_sma.iloc[-i]
    #         for i in range(1, min_confirmation + 1)
    #     )
    #
    #     if ma_bullish and slope > slope_threshold:
    #         return Trend.BULLISH.name
    #     elif ma_bearish and slope < -slope_threshold:
    #         return Trend.BEARISH.name
    #     return Trend.NEUTRAL.name

    @staticmethod
    def get_time_frame_name(ohlc: pd.DataFrame) -> str:
        if not isinstance(ohlc.index, pd.DatetimeIndex):
            raise ValueError("OHLC index must be DatetimeIndex")

        # Determine TF using the most common delta
        deltas = ohlc.index.to_series().diff().dropna()
        seconds = int(deltas.dt.total_seconds().mode().iloc[0])

        if seconds % 86400 == 0:
            return f"{seconds // 86400}D"
        if seconds % 3600 == 0:
            return f"{seconds // 3600}H"
        if seconds % 60 == 0:
            return f"{seconds // 60}M"

        return f"{seconds}s"

    @staticmethod
    def calculate_atr_stop_loss(
            ohlc: pd.DataFrame,
            entry_price: float,
            direction: BIAS,
            atr_length: int = 14,
            atr_multiplier: float = 1.5,
    ) -> float:
        """
        Smart ATR-based Stop Loss calculation.

        Args:
            ohlc: DataFrame with 'high', 'low', 'close' columns.
            entry_price: The price where you intend to enter.
            direction: 'LONG' or 'SHORT'.
            atr_length: Lookback period for ATR (default 14).
            atr_multiplier: Volatility multiplier (default 2.0).
            tick_size: Minimum price variation (e.g., 0.25 for MNQ).

        Returns:
            float: A tick-conformed Stop Loss price.
        """
        # 1. Efficient Vectorized True Range (TR) Calculation
        high_low = ohlc['high'] - ohlc['low']
        high_cp = (ohlc['high'] - ohlc['close'].shift(1)).abs()
        low_cp = (ohlc['low'] - ohlc['close'].shift(1)).abs()

        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)

        # 2. Calculate ATR using RMA (Running Moving Average - TradingView Standard)
        # alpha = 1/length matches Pine Script's ta.atr()
        atr = tr.ewm(alpha=1 / atr_length, min_periods=atr_length, adjust=False).mean().iloc[-1]

        # 3. Calculate Raw Stop Loss
        atr_distance = atr * atr_multiplier

        if direction == BIAS.LONG:
            sl = entry_price - atr_distance
        else:
            sl = entry_price + atr_distance

        return round(sl, 2)
