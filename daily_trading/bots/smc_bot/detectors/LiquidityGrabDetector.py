import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from daily_trading.bots.smc_bot.detectors.StrategyBase import BEARISH, BULLISH, StrategyBase
from daily_trading.settings import logger


class LiquidityGrabDetector(StrategyBase):
    def __init__(self, main_tf, pivot_length=25, wick_body_ratio=0.5, liq_zone_count=5, cooldown=3, volume_threshold=10,
                 left_bars=25, right_bars=25):
        """
        Liquidity Grab Detector
        Definition: Detects liquidity grabs based on rejection candles (hanging man, shooting star, hammer, inverted hammer)
        at pivot highs/lows, indicating institutional liquidity sweeps.
        Purpose: Signals potential reversal after liquidity is absorbed by institutional traders.

        Parameters:
        - main_tf (str): Timeframe for analysis (e.g., '15m', '1h').
        - pivot_length (int): Length for pivot high/low calculation.
        - wick_body_ratio (float): Minimum wick-to-body ratio for rejection candles.
        - liq_zone_count (int): Maximum number of pivot highs/lows to track.
        - cooldown (int): Minimum bars between consecutive signals.
        - volume_threshold (float): Minimum volume to confirm a grab.
        - left_bars (int): Left bars for pivot detection (set to pivot_length).
        - right_bars (int): Right bars for pivot detection (set to pivot_length).
        """
        super().__init__(main_tf=main_tf, left_bars=pivot_length, right_bars=pivot_length)
        self.pivot_length = pivot_length
        self.wick_body_ratio = wick_body_ratio
        self.liq_zone_count = liq_zone_count
        self.cooldown = cooldown
        self.volume_threshold = volume_threshold
        self.buyside_liqs = []
        self.sellside_liqs = []
        self.last_buyside_grab = -cooldown - 1
        self.last_sellside_grab = -cooldown - 1
        logger.debug(f"Initialized LiquidityGrabDetector with main_tf={main_tf}, pivot_length={pivot_length}, "
                     f"wick_body_ratio={wick_body_ratio}, liq_zone_count={liq_zone_count}, cooldown={cooldown}, "
                     f"volume_threshold={volume_threshold}, left_bars={left_bars}, right_bars={right_bars}")

    def calculate_pivots(self, ohlc, length=None):
        """Override to ensure pivot calculation matches Pine Script with equal left and right bars."""
        length = length if length is not None else self.pivot_length
        pivot_high_points = []
        pivot_low_points = []
        for i in range(length, len(ohlc) - length):
            is_high = all(ohlc['high'].iloc[i] > ohlc['high'].iloc[i - j] for j in range(1, length + 1)) and \
                      all(ohlc['high'].iloc[i] >= ohlc['high'].iloc[i + j] for j in range(1, length + 1))
            is_low = all(ohlc['low'].iloc[i] < ohlc['low'].iloc[i - j] for j in range(1, length + 1)) and \
                     all(ohlc['low'].iloc[i] <= ohlc['low'].iloc[i + j] for j in range(1, length + 1))
            if is_high:
                pivot_high_points.append({"index": i, "level": ohlc['high'].iloc[i], "time": ohlc.index[i]})
            if is_low:
                pivot_low_points.append({"index": i, "level": ohlc['low'].iloc[i], "time": ohlc.index[i]})
        return pivot_high_points, pivot_low_points

    def get_invalidation_point(self, ohlc, idx, is_top, is_close):
        """Calculate invalidation point for liquidity grab detection."""
        open_price = ohlc['open'].iloc[idx]
        close_price = ohlc['close'].iloc[idx]
        high_price = ohlc['high'].iloc[idx]
        low_price = ohlc['low'].iloc[idx]

        if is_top and is_close:
            return min(close_price, open_price)
        elif is_top:
            return high_price
        elif is_close:
            return max(close_price, open_price)
        else:
            return low_price

    def detect_liquidity_grabs(self, ohlc, i):
        """Detect liquidity grabs based on Pine Script logic."""
        signals = []
        liqs_to_remove = []
        grab_found = False

        # Update liquidity zones
        if i >= self.pivot_length:
            for pivot in self.pivot_high_points:
                if pivot["index"] == i - self.pivot_length and len(self.buyside_liqs) < self.liq_zone_count:
                    self.buyside_liqs.append(pivot["level"])
                    if len(self.buyside_liqs) > self.liq_zone_count:
                        self.buyside_liqs.pop(0)
            for pivot in self.pivot_low_points:
                if pivot["index"] == i - self.pivot_length and len(self.sellside_liqs) < self.liq_zone_count:
                    self.sellside_liqs.append(pivot["level"])
                    if len(self.sellside_liqs) > self.liq_zone_count:
                        self.sellside_liqs.pop(0)

        # Check for buyside liquidity grab
        for cur_liq in self.buyside_liqs:
            if (self.get_invalidation_point(ohlc, i, True, False) > cur_liq and
                    self.get_invalidation_point(ohlc, i, True, True) < cur_liq):
                body_size = abs(ohlc['close'].iloc[i] - ohlc['open'].iloc[i]) + 1e-6  # Avoid division by zero
                wick_size = ohlc['high'].iloc[i] - max(ohlc['close'].iloc[i], ohlc['open'].iloc[i])
                cur_wbr = wick_size / body_size
                if cur_wbr >= self.wick_body_ratio and ohlc['volume'].iloc[i] > self.volume_threshold:
                    grab_found = True
                    grab_size = min(int(cur_wbr / self.wick_body_ratio), 3)
                    if (i - self.last_buyside_grab) > self.cooldown:
                        signals.append({
                            "type": "LiquidityGrab",
                            "bias": BULLISH,
                            "time": ohlc.index[i],
                            "level": cur_liq,
                            "volume": ohlc['volume'].iloc[i],
                            "size": grab_size
                        })
                        self.last_buyside_grab = i
                        logger.debug(
                            f"Bullish Liquidity Grab detected: level={cur_liq:.6f}, time={ohlc.index[i]}, volume={ohlc['volume'].iloc[i]:.2f}, size={grab_size}")
                    liqs_to_remove.append(cur_liq)
                    break
            # Removed invalidation by close alone
            # elif self.get_invalidation_point(ohlc, i, True, True) > cur_liq:
            #     liqs_to_remove.append(cur_liq)
            #     break

        # Check for sellside liquidity grab if no buyside grab found
        if not grab_found:
            for cur_liq in self.sellside_liqs:
                if (self.get_invalidation_point(ohlc, i, False, False) < cur_liq and
                        self.get_invalidation_point(ohlc, i, False, True) > cur_liq):
                    body_size = abs(ohlc['close'].iloc[i] - ohlc['open'].iloc[i]) + 1e-6  # Avoid division by zero
                    wick_size = min(ohlc['close'].iloc[i], ohlc['open'].iloc[i]) - ohlc['low'].iloc[i]
                    cur_wbr = wick_size / body_size
                    if cur_wbr >= self.wick_body_ratio and ohlc['volume'].iloc[i] > self.volume_threshold:
                        grab_size = min(int(cur_wbr / self.wick_body_ratio), 3)
                        if (i - self.last_sellside_grab) > self.cooldown:
                            signals.append({
                                "type": "LiquidityGrab",
                                "bias": BEARISH,
                                "time": ohlc.index[i],
                                "level": cur_liq,
                                "volume": ohlc['volume'].iloc[i],
                                "size": grab_size
                            })
                            self.last_sellside_grab = i
                            logger.debug(
                                f"Bearish Liquidity Grab detected: level={cur_liq:.6f}, time={ohlc.index[i]}, volume={ohlc['volume'].iloc[i]:.2f}, size={grab_size}")
                        liqs_to_remove.append(cur_liq)
                        break
                # Removed invalidation by close alone
                # elif self.get_invalidation_point(ohlc, i, False, True) < cur_liq:
                #     liqs_to_remove.append(cur_liq)
                #     break

        # Remove invalidated liquidity levels
        for liq in liqs_to_remove:
            if liq in self.buyside_liqs:
                self.buyside_liqs.remove(liq)
            elif liq in self.sellside_liqs:
                self.sellside_liqs.remove(liq)

        return signals

    def calculate_liquidity_grabs(self, ohlc, plot=True):
        """
        Calculate liquidity grabs based on rejection candles at pivot points.

        Parameters:
        - ohlc (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
        - plot (bool): If True, generate a Plotly figure with candlesticks and grab signals.

        Returns:
        - dict: Dictionary containing:
            - fig (go.Figure): Plotly figure with candlesticks and grab signals (if plot=True).
            - signals (list): List of detected liquidity grab signals.
        """
        logger.debug(f"Starting liquidity grab calculation with pivot_length={self.pivot_length}, "
                     f"wick_body_ratio={self.wick_body_ratio}, liq_zone_count={self.liq_zone_count}, cooldown={self.cooldown}")
        if len(ohlc) < self.pivot_length * 2 + 1:
            logger.error(f"OHLC data too short ({len(ohlc)} rows) for pivot_length={self.pivot_length}")
            return {"fig": None, "signals": []}

        ohlc = self.validate_data(ohlc)
        self.pivot_high_points, self.pivot_low_points = self.calculate_pivots(ohlc, length=self.pivot_length)
        signals = []

        for i in range(self.pivot_length, len(ohlc)):
            grab_signals = self.detect_liquidity_grabs(ohlc, i)
            signals.extend(grab_signals)

        fig = None
        if plot:
            fig = self._plot_grabs(ohlc, signals)

        logger.debug(
            f"Detected {len(signals)} liquidity grabs: {sum(1 for s in signals if s['bias'] == BULLISH)} bullish, "
            f"{sum(1 for s in signals if s['bias'] == BEARISH)} bearish")
        return {"fig": fig, "signals": signals}

    def _plot_grabs(self, ohlc, signals):
        """
        Plot candlesticks and liquidity grab signals with improved visibility.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame.
        - signals (list): List of grab signals.

        Returns:
        - go.Figure: Plotly figure with candlesticks and grab signals.
        """
        ohlc = ohlc.sort_index()
        ohlc = ohlc.tail(self.plot_tail)
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'

        logger.debug("Generating Plotly figure for liquidity grabs")
        ohlc = self.validate_data(ohlc)
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} | Price Chart with Liquidity Grabs"])

        # Plot candlesticks
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ), row=1, col=1)

        # Plot grab signals with circles at the top of the rejection candle
        for signal in signals:
            marker_color = '#1E90FF' if signal['bias'] == BULLISH else '#DC143C'
            marker_size = 8 + 4 * signal['size']
            end_time = ohlc.index[min(ohlc.index.get_loc(signal['time']) + 5, len(ohlc) - 1)]
            signal_idx = ohlc.index.get_loc(signal['time'])
            fig.add_trace(go.Scatter(
                x=[signal['time']],
                y=[ohlc['high'].iloc[signal_idx]],  # Place at the high of the rejection candle
                mode='markers',
                marker=dict(symbol='circle', size=marker_size, color=marker_color, line=dict(width=2, color='black')),
                name=f"{'Bull' if signal['bias'] == BULLISH else 'Bear'} Grab (Size {signal['size']})",
                showlegend=True
            ), row=1, col=1)
            fig.add_shape(
                type="line",
                x0=signal['time'], x1=end_time,
                y0=signal['level'], y1=signal['level'],
                line=dict(color=marker_color, width=2, dash="dot"),
                opacity=0.7
            )

        last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | Liquidity Grabs | {candle_time_str} | TF: {self.main_tf}",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            xaxis=dict(type='date', tickformat='%Y-%m-%d %H:%M', showticklabels=True),
            yaxis=dict(title="Price"),
            margin=dict(t=100, b=100, l=50, r=50)
        )
        fig.update_xaxes(title_text="Time", row=1, col=1)
        logger.debug("Plotly figure generated successfully")
        return fig