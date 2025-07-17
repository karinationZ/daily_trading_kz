import plotly.graph_objects as go
import talib
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.clients.SMCBase import BEARISH, BULLISH, SMCBase, Trend
from daily_trading.settings import logger


class CandlePatternDetector(SMCBase):
    def __init__(self, main_tf, volume_window=200, volume_factor=1.0, min_pattern_length=5, length=5,
                 left_bars=5, right_bars=5, pin_body_ratio=0.15, pin_wick_ratio=0.60,
                 pin_opposite_wick_ratio=0.05):
        """
        Initialize the CandleDetector with timeframe-dependent configuration.

        Parameters:
        - lower_tf (str): Lower timeframe for analysis (e.g., '15m', '1h', '4h').
        - volume_window (int): Window for rolling mean volume calculation (default=14).
        - volume_factor (float): Scaling factor for mean volume threshold (default=1.0).
        - min_pattern_length (int): Minimum number of bars for pattern detection.
        - length (int): Length for pivot calculation (inherited from SMCBase).
        - left_bars (int): Number of left bars for pivot detection.
        - right_bars (int): Number of right bars for pivot detection.
        - pin_body_ratio (float): Maximum body size as a fraction of candle range for Pin Bars (default=0.15).
        - pin_wick_ratio (float): Minimum wick size as a fraction of candle range for Pin Bars (default=0.60).
        - pin_opposite_wick_ratio (float): Maximum opposite wick size as a fraction of candle range for Pin Bars (default=0.05).
        """
        super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
        self.volume_window = volume_window
        self.volume_factor = volume_factor
        self.min_pattern_length = min_pattern_length
        self.length = length
        self.pin_body_ratio = pin_body_ratio
        self.pin_wick_ratio = pin_wick_ratio
        self.pin_opposite_wick_ratio = pin_opposite_wick_ratio
        if talib is None:
            logger.warning(
                "TA-Lib not installed. Standard candlestick patterns disabled, but Pin Bar detection will proceed.")
        logger.info(f"Initialized CandleDetector with lower_tf={main_tf}, volume_window={volume_window}, "
                    f"volume_factor={volume_factor}, min_pattern_length={min_pattern_length}, length={length}, "
                    f"left_bars={left_bars}, right_bars={right_bars}, pin_body_ratio={pin_body_ratio}, "
                    f"pin_wick_ratio={pin_wick_ratio}, pin_opposite_wick_ratio={pin_opposite_wick_ratio}")

    def detect_pin_bars(self, ohlc):
        """
        Detect Pin Bar patterns (long wick, small body, no/minimal opposite wick).

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].

        Returns:
        - pd.DataFrame: DataFrame with columns 'Bullish Pin Bar' and 'Bearish Pin Bar' (1 for bullish, -1 for bearish, 0 otherwise).
        """
        logger.info(f"Detecting Pin Bar patterns on OHLC data with {len(ohlc)} rows")
        ohlc = self.validate_data(ohlc)
        if len(ohlc) < 1:
            logger.error(f"OHLC data too short ({len(ohlc)} rows) for Pin Bar detection")
            return ohlc

        result = ohlc.copy()
        result['Bullish Pin Bar'] = 0
        result['Bearish Pin Bar'] = 0

        # Calculate rolling mean volume
        mean_volume = ohlc['volume'].rolling(window=self.volume_window, min_periods=1).mean()
        logger.info(f"Mean volume threshold at last candle: {mean_volume.iloc[-1]:.2f} * {self.volume_factor}")

        for i in range(len(ohlc)):
            open_price = ohlc['open'].iloc[i]
            high_price = ohlc['high'].iloc[i]
            low_price = ohlc['low'].iloc[i]
            close_price = ohlc['close'].iloc[i]
            volume = ohlc['volume'].iloc[i]
            volume_threshold = mean_volume.iloc[i] * self.volume_factor

            candle_range = high_price - low_price
            if candle_range == 0:  # Avoid division by zero
                continue

            body = abs(close_price - open_price)
            lower_wick = min(open_price, close_price) - low_price
            upper_wick = high_price - max(open_price, close_price)

            body_ratio = body / candle_range
            lower_wick_ratio = lower_wick / candle_range
            upper_wick_ratio = upper_wick / candle_range

            # Bullish Pin Bar: Long lower wick, small body, minimal/no upper wick
            if (body_ratio <= self.pin_body_ratio and
                    lower_wick_ratio >= self.pin_wick_ratio and
                    upper_wick_ratio <= self.pin_opposite_wick_ratio and
                    volume >= volume_threshold):
                result['Bullish Pin Bar'].iloc[i] = 1
                logger.debug(f"Bullish Pin Bar at {ohlc.index[i]}: body_ratio={body_ratio:.3f}, "
                             f"lower_wick_ratio={lower_wick_ratio:.3f}, upper_wick_ratio={upper_wick_ratio:.3f}, "
                             f"volume={volume:.2f}, threshold={volume_threshold:.2f}")

            # Bearish Pin Bar: Long upper wick, small body, minimal/no lower wick
            if (body_ratio <= self.pin_body_ratio and
                    upper_wick_ratio >= self.pin_wick_ratio and
                    lower_wick_ratio <= self.pin_opposite_wick_ratio and
                    volume >= volume_threshold):
                result['Bearish Pin Bar'].iloc[i] = -1
                logger.debug(f"Bearish Pin Bar at {ohlc.index[i]}: body_ratio={body_ratio:.3f}, "
                             f"upper_wick_ratio={upper_wick_ratio:.3f}, lower_wick_ratio={lower_wick_ratio:.3f}, "
                             f"volume={volume:.2f}, threshold={volume_threshold:.2f}")

        return result

    def detect_candlestick_patterns(self, ohlc):
        """
        Detect candlestick patterns using TA-Lib and custom Pin Bar detection, filtered by mean volume.

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].

        Returns:
        - pd.DataFrame: DataFrame with additional columns for detected patterns.
        """
        logger.info(f"Detecting candlestick patterns on OHLC data with {len(ohlc)} rows")
        ohlc = self.validate_data(ohlc)
        if len(ohlc) < self.min_pattern_length:
            logger.error(
                f"OHLC data too short ({len(ohlc)} rows) for pattern detection (min={self.min_pattern_length})")
            return ohlc

        result = ohlc.copy()

        # Detect Pin Bars
        result = self.detect_pin_bars(result)

        # Detect TA-Lib patterns if available
        if talib:
            open_prices = result['open'].values
            high_prices = result['high'].values
            low_prices = result['low'].values
            close_prices = result['close'].values

            # Calculate rolling mean volume
            mean_volume = result['volume'].rolling(window=self.volume_window, min_periods=1).mean()
            logger.info(f"Mean volume threshold at last candle: {mean_volume.iloc[-1]:.2f} * {self.volume_factor}")

            patterns = {
                'Bullish Engulfing': talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices),
                'Morning Star': talib.CDLMORNINGSTAR(open_prices, high_prices, low_prices, close_prices,
                                                     penetration=0.3),
                'Hammer': talib.CDLHAMMER(open_prices, high_prices, low_prices, close_prices),
                'Inverted Hammer': talib.CDLINVERTEDHAMMER(open_prices, high_prices, low_prices, close_prices),
                'Piercing Pattern': talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices),
                'Three White Soldiers': talib.CDL3WHITESOLDIERS(open_prices, high_prices, low_prices, close_prices),
                'Rising Three Methods': talib.CDLRISEFALL3METHODS(open_prices, high_prices, low_prices, close_prices),
                'Bearish Engulfing': talib.CDLENGULFING(open_prices, high_prices, low_prices, close_prices) * -1,
                'Evening Star': talib.CDLEVENINGSTAR(open_prices, high_prices, low_prices, close_prices,
                                                     penetration=0.3),
                'Shooting Star': talib.CDLSHOOTINGSTAR(open_prices, high_prices, low_prices, close_prices),
                'Dark Pool Cover': talib.CDLDARKCLOUDCOVER(open_prices, high_prices, low_prices, close_prices,
                                                           penetration=0.5),
                'Three Black Crows': talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices),
                'Falling Three Methods': talib.CDLRISEFALL3METHODS(open_prices, high_prices, low_prices,
                                                                   close_prices) * -1
            }

            for pattern_name, pattern_values in patterns.items():
                result[pattern_name] = pattern_values
                result[pattern_name] = result.apply(
                    lambda row: row[pattern_name] if row['volume'] >= mean_volume[row.name] * self.volume_factor else 0,
                    axis=1
                )
                logger.debug(f"Detected {sum(row != 0 for row in result[pattern_name])} {pattern_name} patterns")

            # Fix: Prevent conflicting Bullish and Bearish Engulfing patterns
            for idx, row in result.iterrows():
                if row.get('Bullish Engulfing', 0) != 0 and row.get('Bearish Engulfing', 0) != 0:
                    logger.error(
                        f"Conflicting patterns at {row.name}: Bullish and Bearish Engulfing detected simultaneously")
                    result.at[idx, 'Bearish Engulfing'] = 0  # Prioritize Bullish Engulfing

        return result

    def filter_candlestick_signals(self, df):
        """
        Filter DataFrame to include only rows with detected candlestick patterns and assign bias.

        Parameters:
        - df (pd.DataFrame): DataFrame with pattern columns from detect_candlestick_patterns.

        Returns:
        - list: List of dictionaries with pattern signals (time, pattern, bias, volume).
        """
        logger.info("Filtering candlestick patterns to generate signals")
        signals = []
        pattern_columns = [
            'Bullish Pin Bar', 'Bearish Pin Bar',
            'Bullish Engulfing', 'Morning Star', 'Hammer', 'Inverted Hammer',
            'Piercing Pattern', 'Three White Soldiers', 'Rising Three Methods',
            'Bearish Engulfing', 'Evening Star', 'Shooting Star', 'Dark Pool Cover',
            'Three Black Crows', 'Falling Three Methods'
        ]

        for idx, row in df.iterrows():
            for pattern in pattern_columns:
                if row.get(pattern, 0) != 0:
                    bias = BULLISH if pattern in [
                        'Bullish Pin Bar', 'Bullish Engulfing', 'Morning Star', 'Hammer', 'Inverted Hammer',
                        'Piercing Pattern', 'Three White Soldiers', 'Rising Three Methods'
                    ] else BEARISH
                    signals.append({
                        'type': 'Candlestick',
                        'pattern': pattern,
                        'bias': bias,
                        'time': row.name,
                        'level': row['close'],
                        'volume': row['volume']
                    })
                    logger.info(
                        f"{pattern} detected at {row.name}, bias={'Bullish' if bias == BULLISH else 'Bearish'}, "
                        f"level={row['close']:.6f}, volume={row['volume']:.2f}")

        logger.info(f"Generated {len(signals)} candlestick signals")
        return signals

    def calculate_candlestick_signals(self, ohlc, plot=True):
        """
        Calculate candlestick patterns and generate trading signals.

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
        - plot (bool): If True, generate a Plotly figure with candlesticks and pattern signals.

        Returns:
        - dict: Dictionary containing:
            - fig (go.Figure): Plotly figure with candlesticks and pattern signals (if plot=True).
            - signals (list): List of detected candlestick signals.
        """
        logger.info(
            f"Starting candlestick signal calculation with volume_window={self.volume_window}, volume_factor={self.volume_factor}")
        ohlc = self.validate_data(ohlc)
        if len(ohlc) < self.min_pattern_length:
            logger.error(f"OHLC data too short ({len(ohlc)} rows) for candlestick pattern detection")
            return {"fig": None, "signals": []}

        df_with_patterns = self.detect_candlestick_patterns(ohlc)
        signals = self.filter_candlestick_signals(df_with_patterns)

        fig = None
        if plot:
            fig = self._plot_candlestick_signals(ohlc, signals)

        return {"fig": fig, "signals": signals}

    def _plot_candlestick_signals(self, ohlc, signals):
        """
        Plot candlesticks and candlestick pattern signals.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame.
        - signals (list): List of candlestick pattern signals.

        Returns:
        - go.Figure: Plotly figure with candlesticks and pattern signals.
        """
        logger.info("Generating Plotly figure for candlestick patterns")
        ohlc = self.validate_data(ohlc)
        ohlc = ohlc.tail(self.plot_tail)

        fig = make_subplots(rows=1, cols=1, subplot_titles=["Price Chart with Candlestick Patterns"])

        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ), row=1, col=1)

        for signal in signals:
            marker_color = 'blue' if signal['bias'] == BULLISH else 'red'
            marker_symbol = 'triangle-up' if signal['bias'] == BULLISH else 'triangle-down'
            fig.add_trace(go.Scatter(
                x=[signal['time']],
                y=[signal['level'] * (1.002 if signal['bias'] == BULLISH else 0.998)],
                mode='markers+text',
                text=[signal['pattern']],
                textposition='top center' if signal['bias'] == BULLISH else 'bottom center',
                marker=dict(symbol=marker_symbol, size=10, color=marker_color),
                name=signal['pattern'],
                showlegend=True
            ), row=1, col=1)

        fig.update_layout(
            title="Candlestick Patterns [SMC]",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            height=800,
            xaxis=dict(type='date', tickformat='%Y-%m-%d %H:%M', showticklabels=True),
            yaxis=dict(title="Price"),
            margin=dict(t=100, b=100, l=50, r=50)
        )
        fig.update_xaxes(title_text="Time", row=1, col=1)
        logger.info("Plotly figure generated successfully")
        return fig

    def check_pattern(self, open, high, low, close, pattern, config):
        """
        Custom detection for single-candle patterns with configurable thresholds.

        Parameters:
        - open, high, low, close (float): Candle OHLC values.
        - pattern (str): Pattern to check (e.g., 'Belt Hold', 'Hammer').
        - config (dict): Thresholds for shadow, body, and wick ratios.

        Returns:
        - int: 100 (bullish), -100 (bearish), or 0 (no pattern).
        """
        range_ = high - low
        if range_ == 0:
            return 0
        body = abs(close - open)
        lower_shadow = min(open, close) - low
        upper_shadow = high - max(open, close)
        body_ratio = body / range_ if range_ > 0 else 0
        lower_wick_ratio = lower_shadow / range_ if range_ > 0 else 0
        upper_wick_ratio = upper_shadow / range_ if range_ > 0 else 0

        shadow_threshold = config.get('shadow_threshold', 0.1)
        body_threshold = config.get('body_threshold', 0.7)
        small_body_threshold = config.get('small_body_threshold', 0.3)
        long_wick_threshold = config.get('long_wick_threshold', 0.6)

        if pattern == 'Belt Hold':
            if (close > open and
                    lower_wick_ratio <= shadow_threshold and
                    upper_wick_ratio <= shadow_threshold and
                    body_ratio >= body_threshold):
                return 100
            elif (close < open and
                  upper_wick_ratio <= shadow_threshold and
                  lower_wick_ratio <= shadow_threshold and
                  body_ratio >= body_threshold):
                return -100
        elif pattern == 'Closing Marubozu':
            if (close > open and
                    lower_wick_ratio <= shadow_threshold and
                    upper_wick_ratio <= shadow_threshold and
                    body_ratio >= body_threshold):
                return 100
            elif (close < open and
                  upper_wick_ratio <= shadow_threshold and
                  lower_wick_ratio <= shadow_threshold and
                  body_ratio >= body_threshold):
                return -100
        elif pattern == 'Doji':
            if body_ratio <= small_body_threshold:
                return 100  # Neutral, treated as bullish for consistency
        elif pattern == 'Dragonfly Doji':
            if (body_ratio <= small_body_threshold and
                    lower_wick_ratio >= long_wick_threshold and
                    upper_wick_ratio <= shadow_threshold):
                return 100
        elif pattern == 'Gravestone Doji':
            if (body_ratio <= small_body_threshold and
                    upper_wick_ratio >= long_wick_threshold and
                    lower_wick_ratio <= shadow_threshold):
                return -100
        elif pattern == 'Hammer':
            if (close >= open and
                    lower_wick_ratio >= long_wick_threshold and
                    upper_wick_ratio <= shadow_threshold and
                    body_ratio <= small_body_threshold):
                return 100
        elif pattern == 'Hanging Man':
            if (close <= open and
                    lower_wick_ratio >= long_wick_threshold and
                    upper_wick_ratio <= shadow_threshold and
                    body_ratio <= small_body_threshold):
                return -100
        elif pattern == 'High Wave':
            if (body_ratio <= small_body_threshold and
                    lower_wick_ratio >= long_wick_threshold and
                    upper_wick_ratio >= long_wick_threshold):
                return 100  # Neutral, treated as bullish
        elif pattern == 'Inverted Hammer':
            if (close >= open and
                    upper_wick_ratio >= long_wick_threshold and
                    lower_wick_ratio <= shadow_threshold and
                    body_ratio <= small_body_threshold):
                return 100
        elif pattern == 'Long Legged Doji':
            if (body_ratio <= small_body_threshold and
                    lower_wick_ratio >= long_wick_threshold and
                    upper_wick_ratio >= long_wick_threshold):
                return 100  # Neutral, treated as bullish
        elif pattern == 'Long Line':
            if (close > open and body_ratio >= body_threshold):
                return 100
            elif (close < open and body_ratio >= body_threshold):
                return -100
        elif pattern == 'Marubozu':
            if (close > open and
                    lower_wick_ratio <= shadow_threshold and
                    upper_wick_ratio <= shadow_threshold and
                    body_ratio >= body_threshold):
                return 100
            elif (close < open and
                  upper_wick_ratio <= shadow_threshold and
                  lower_wick_ratio <= shadow_threshold and
                  body_ratio >= body_threshold):
                return -100
        elif pattern == 'Rickshaw Man':
            if (body_ratio <= small_body_threshold and
                    lower_wick_ratio >= long_wick_threshold and
                    upper_wick_ratio >= long_wick_threshold and
                    abs(open - close) <= abs(high - low) * 0.1):
                return 100  # Neutral, treated as bullish
        elif pattern == 'Shooting Star':
            if (close <= open and
                    upper_wick_ratio >= long_wick_threshold and
                    lower_wick_ratio <= shadow_threshold and
                    body_ratio <= small_body_threshold):
                return -100
        elif pattern == 'Short Line':
            if body_ratio <= small_body_threshold:
                return 100  # Neutral, treated as bullish
        elif pattern == 'Spinning Top':
            if (body_ratio <= small_body_threshold and
                    lower_wick_ratio >= shadow_threshold and
                    upper_wick_ratio >= shadow_threshold):
                return 100  # Neutral, treated as bullish
        elif pattern == 'Takuri':
            if (close >= open and
                    lower_wick_ratio >= long_wick_threshold * 1.5 and
                    upper_wick_ratio <= shadow_threshold and
                    body_ratio <= small_body_threshold):
                return 100
        return 0

    def detect_single_candle_pattern(self, df, config=None):
        """
        Detect single-candle pattern with custom logic and configurable thresholds.

        Parameters:
        - df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close'].
        - config (dict): Optional thresholds {'shadow_threshold', 'body_threshold', 'small_body_threshold', 'long_wick_threshold'}.

        Returns:
        - tuple: (trend (int), pattern_name (str)) - Trend is 1 (Bullish), -1 (Bearish), or 0 (Neutral).
        """
        logger.info("Detecting single-candle pattern")
        if not all(col in df for col in ['open', 'high', 'low', 'close']):
            logger.error("Missing required columns in DataFrame")
            return Trend.NEUTRAL, "None"

        config = config or {
            'shadow_threshold': 0.1,
            'body_threshold': 0.7,
            'small_body_threshold': 0.3,
            'long_wick_threshold': 0.6
        }

        patterns = [
            'Belt Hold', 'Closing Marubozu', 'Doji', 'Dragonfly Doji', 'Gravestone Doji',
            'Hammer', 'Hanging Man', 'High Wave', 'Inverted Hammer', 'Long Legged Doji',
            'Long Line', 'Marubozu', 'Rickshaw Man', 'Shooting Star', 'Short Line',
            'Spinning Top', 'Takuri'
        ]

        for pattern in patterns:
            value = self.check_pattern(df['open'][0], df['high'][0], df['low'][0], df['close'][0], pattern, config)
            if value > 0:
                logger.info(f"Detected {pattern} (Bullish)")
                return Trend.BULLISH, pattern
            elif value < 0:
                logger.info(f"Detected {pattern} (Bearish)")
                return Trend.BEARISH, pattern

        logger.info("No pattern detected (Neutral)")
        return Trend.NEUTRAL, "None"
