from daily_trading.bots.smc_bot.clients.SMCBase import SMCBase, Trend
from daily_trading.settings import logger


class CandlePatternDetector(SMCBase):
    def __init__(self, main_tf):
        super().__init__(main_tf=main_tf)

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
