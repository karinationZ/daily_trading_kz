import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.clients.SMCBase import BEARISH, BULLISH, SMCBase
from daily_trading.settings import logger


class BosChochDetector(SMCBase):
    def __init__(self, main_tf, pivot_period=None, left_bars=5, right_bars=1, show_bull=True, show_bear=True,
                 volume_thresh=5, min_price_move=0.001, min_signal_gap=1):
        super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
        self.pivot_period = pivot_period if pivot_period is not None else self.length
        self.show_bull = show_bull
        self.show_bear = show_bear
        self.volume_thresh = volume_thresh
        self.min_price_move = min_price_move
        self.min_signal_gap = min_signal_gap
        self.ExternalTrend = 'No Trend'
        self.LockBreak_M = None
        self.LastSignalIndex = None
        self.bull_ms_count = 0
        self.bear_ms_count = 0
        self.Major_HighLevel = None
        self.Major_LowLevel = None
        self.Major_HighIndex = None
        self.Major_LowIndex = None
        self.Major_HighType = None
        self.Major_LowType = None
        logger.info(f"Initialized BosChochDetector with lower_tf={main_tf}, pivot_period={self.pivot_period}, "
                    f"left_bars={left_bars}, right_bars={right_bars}, show_bull={show_bull}, "
                    f"show_bear={show_bear}, volume_thresh={volume_thresh}, "
                    f"min_price_move={min_price_move}, min_signal_gap={min_signal_gap}")

    def detect_signals(self, ohlc, i, signals, pivot_high_points, pivot_low_points):
        close = ohlc['close'].iloc[i]
        high = ohlc['high'].iloc[i]
        low = ohlc['low'].iloc[i]
        time = ohlc.index[i]
        volume = ohlc['volume'].iloc[i]

        if self.LastSignalIndex is not None and i - self.LastSignalIndex < self.min_signal_gap:
            logger.debug(f"Skipping BOS/CHoCH at index {i}, time {time}: Within min_signal_gap={self.min_signal_gap}")
            return signals

        logger.debug(f"Checking signals at index {i}, time {time}, close={close:.6f}, "
                     f"ExternalTrend={self.ExternalTrend}, Major_HighLevel={self.Major_HighLevel}, "
                     f"Major_LowLevel={self.Major_LowLevel}, LockBreak_M={self.LockBreak_M}, volume={volume:.2f}")

        # Update trend based on pivot sequence
        if len(self.ArrayType) >= 3:
            last_pivot = self.ArrayType[-1]
            prev_pivot = self.ArrayType[-2]
            prev_prev_pivot = self.ArrayType[-3]
            if last_pivot == 'LH' and prev_pivot in ['HH', 'H'] and prev_prev_pivot in ['L', 'LL', 'HL']:
                self.ExternalTrend = 'Down Trend'
            elif last_pivot == 'LL' and prev_pivot == 'LH' and prev_prev_pivot in ['H', 'HH', 'LH']:
                self.ExternalTrend = 'Down Trend'
            elif last_pivot == 'HH' and prev_pivot in ['L', 'LL', 'HL'] and prev_prev_pivot in ['H', 'HH', 'LH']:
                self.ExternalTrend = 'Up Trend'
            elif last_pivot == 'HL' and prev_pivot == 'HH' and prev_prev_pivot in ['L', 'LL', 'HL']:
                self.ExternalTrend = 'Up Trend'

        # Bullish Major BOS/CHoCH
        if (self.Major_HighLevel is not None and high > self.Major_HighLevel and
                self.show_bull and volume >= self.volume_thresh):
            logger.debug(f"Bullish BOS check: high={high:.6f}, Major_HighLevel={self.Major_HighLevel:.6f}, "
                         f"LockBreak_M={self.LockBreak_M}, Major_HighIndex={self.Major_HighIndex}, volume={volume:.2f}")
            next_close = ohlc['close'].iloc[i + 1] if i < len(ohlc) - 1 else close
            if next_close > self.Major_HighLevel or close > self.Major_HighLevel:
                signal_type = "BOS" if self.ExternalTrend in ['No Trend', 'Up Trend'] else "CHoCH"
                signals.append({
                    "type": f"{signal_type}",
                    "bias": BULLISH,
                    "time": time,
                    "level": self.Major_HighLevel,
                    "fractal_time": ohlc.index[self.Major_HighIndex] if self.Major_HighIndex is not None else time,
                    "index": i,
                    "volume": volume
                })
                self.bull_ms_count += 1
                self.ExternalTrend = 'Up Trend'
                self.LockBreak_M = self.Major_HighIndex
                self.LastSignalIndex = i
                logger.info(f"Bullish {signal_type} detected: level={self.Major_HighLevel:.6f}, time={time}")
                # Update Major_HighLevel after BOS
                self.Major_HighLevel = high
                self.Major_HighIndex = i
                self.Major_HighType = 'H'
                recent_lows = [p for p in pivot_low_points if p['index'] <= i]
                if recent_lows:
                    latest_low = max(recent_lows, key=lambda x: x['index'])
                    self.Major_LowLevel = latest_low['level']
                    self.Major_LowIndex = latest_low['index']
                    self.Major_LowType = latest_low['type']
                    logger.debug(f"Updated Major_LowLevel={self.Major_LowLevel:.6f} at index {self.Major_LowIndex}")
            else:
                logger.debug(f"Bullish BOS skipped: next_close={next_close:.6f}, close={close:.6f}, "
                             f"Major_HighLevel={self.Major_HighLevel:.6f}")

        # Bearish Major BOS/CHoCH
        if (self.Major_LowLevel is not None and low < self.Major_LowLevel and
                self.show_bear and volume >= self.volume_thresh):
            logger.debug(f"Bearish BOS check: low={low:.6f}, Major_LowLevel={self.Major_LowLevel:.6f}, "
                         f"LockBreak_M={self.LockBreak_M}, Major_LowIndex={self.Major_LowIndex}, volume={volume:.2f}")
            next_close = ohlc['close'].iloc[i + 1] if i < len(ohlc) - 1 else close
            if next_close < self.Major_LowLevel or close < self.Major_LowLevel:
                signal_type = "BOS" if self.ExternalTrend in ['No Trend', 'Down Trend'] else "CHoCH"
                signals.append({
                    "type": f"{signal_type}",
                    "bias": BEARISH,
                    "time": time,
                    "level": self.Major_LowLevel,
                    "fractal_time": ohlc.index[self.Major_LowIndex] if self.Major_LowIndex is not None else time,
                    "index": i,
                    "volume": volume
                })
                self.bear_ms_count += 1
                self.ExternalTrend = 'Down Trend'
                self.LockBreak_M = self.Major_LowIndex
                self.LastSignalIndex = i
                logger.info(f"Bearish {signal_type} detected: level={self.Major_LowLevel:.6f}, time={time}")
                # Update Major_LowLevel after BOS
                self.Major_LowLevel = low
                self.Major_LowIndex = i
                self.Major_LowType = 'L'
                recent_highs = [p for p in pivot_high_points if p['index'] <= i]
                if recent_highs:
                    latest_high = max(recent_highs, key=lambda x: x['index'])
                    self.Major_HighLevel = latest_high['level']
                    self.Major_HighIndex = latest_high['index']
                    self.Major_HighType = latest_high['type']
                    logger.debug(f"Updated Major_HighLevel={self.Major_HighLevel:.6f} at index {self.Major_HighIndex}")
            else:
                logger.debug(f"Bearish BOS skipped: next_close={next_close:.6f}, close={close:.6f}, "
                             f"Major_LowLevel={self.Major_LowLevel:.6f}")

        # Update major levels when new pivots are confirmed (after BOS check)
        if i in self.ArrayIndex:
            pivot_idx = self.ArrayIndex.index(i)
            pivot_type = self.ArrayType[pivot_idx]
            pivot_value = self.ArrayValue[pivot_idx]
            if pivot_type in ['H', 'HH', 'LH']:
                if (self.Major_HighLevel is None or
                        abs(pivot_value - self.Major_HighLevel) / self.Major_HighLevel >= self.min_price_move):
                    self.Major_HighLevel = pivot_value
                    self.Major_HighIndex = i
                    self.Major_HighType = pivot_type
                    self.LockBreak_M = None
                    logger.debug(f"Updated Major_HighLevel={self.Major_HighLevel:.6f} at index {i}, reset LockBreak_M")
            elif pivot_type in ['L', 'LL', 'HL']:
                if (self.Major_LowLevel is None or
                        abs(pivot_value - self.Major_LowLevel) / self.Major_LowLevel >= self.min_price_move):
                    self.Major_LowLevel = pivot_value
                    self.Major_LowIndex = i
                    self.Major_LowType = pivot_type
                    self.LockBreak_M = None
                    logger.debug(f"Updated Major_LowLevel={self.Major_LowLevel:.6f} at index {i}, reset LockBreak_M")

        return signals

    def calculate_bos_choch(self, ohlc, plot=True):
        logger.info(f"Starting BOS and CHoCH calculation with pivot_period={self.pivot_period}, "
                    f"left_bars={self.left_bars}, right_bars={self.right_bars}")
        if len(ohlc) < self.left_bars + self.right_bars + 1:
            logger.error(f"OHLC data too short ({len(ohlc)} rows)")
            return {"fig": None, "signals": [], "alert_conditions": {
                "BULLISH_BOS": [], "BEARISH_BOS": [], "BULLISH_CHoCH": [], "BEARISH_CHoCH": []
            }}
        ohlc = self.validate_data(ohlc)

        # Reset state
        self.ArrayType = []
        self.ArrayValue = []
        self.ArrayIndex = []
        self.Major_HighLevel = None
        self.Major_LowLevel = None
        self.Major_HighIndex = None
        self.Major_LowIndex = None
        self.Major_HighType = None
        self.Major_LowType = None
        self.ExternalTrend = 'No Trend'
        self.LockBreak_M = None
        self.LastSignalIndex = None
        self.bull_ms_count = 0
        self.bear_ms_count = 0

        signals = []
        pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)

        # Process up to the last candle
        for i in range(self.left_bars, len(ohlc)):
            signals = self.detect_signals(ohlc, i, signals, pivot_high_points, pivot_low_points)

        # Update Major_HighLevel or Major_LowLevel for the last candle if it's a potential pivot
        last_index = len(ohlc) - 1
        last_high = ohlc['high'].iloc[last_index]
        last_low = ohlc['low'].iloc[last_index]
        last_volume = ohlc['volume'].iloc[last_index]
        if (self.Major_HighLevel is not None and last_high > self.Major_HighLevel and
                self.show_bull and last_volume >= self.volume_thresh):
            logger.debug(
                f"Last candle bullish BOS check: high={last_high:.6f}, Major_HighLevel={self.Major_HighLevel:.6f}, "
                f"volume={last_volume:.2f}")
            last_close = ohlc['close'].iloc[last_index]
            if last_close > self.Major_HighLevel:
                signal_type = "BOS" if self.ExternalTrend in ['No Trend', 'Up Trend'] else "CHoCH"
                signals.append({
                    "type": f"{signal_type}",
                    "bias": BULLISH,
                    "time": ohlc.index[last_index],
                    "level": self.Major_HighLevel,
                    "fractal_time": ohlc.index[self.Major_HighIndex] if self.Major_HighIndex is not None else
                    ohlc.index[last_index],
                    "index": last_index,
                    "volume": last_volume
                })
                self.bull_ms_count += 1
                self.ExternalTrend = 'Up Trend'
                self.LockBreak_M = self.Major_HighIndex
                self.LastSignalIndex = last_index
                logger.info(
                    f"Bullish Major {signal_type} detected (last candle): level={self.Major_HighLevel:.6f}, time={ohlc.index[last_index]}")
                self.Major_HighLevel = last_high
                self.Major_HighIndex = last_index
                self.Major_HighType = 'H'
        elif (self.Major_LowLevel is not None and last_low < self.Major_LowLevel and
              self.show_bear and last_volume >= self.volume_thresh):
            logger.debug(
                f"Last candle bearish BOS check: low={last_low:.6f}, Major_LowLevel={self.Major_LowLevel:.6f}, "
                f"volume={last_volume:.2f}")
            last_close = ohlc['close'].iloc[last_index]
            if last_close < self.Major_LowLevel:
                signal_type = "BOS" if self.ExternalTrend in ['No Trend', 'Down Trend'] else "CHoCH"
                signals.append({
                    "type": f"{signal_type}",
                    "bias": BEARISH,
                    "time": ohlc.index[last_index],
                    "level": self.Major_LowLevel,
                    "fractal_time": ohlc.index[self.Major_LowIndex] if self.Major_LowIndex is not None else ohlc.index[
                        last_index],
                    "index": last_index,
                    "volume": last_volume
                })
                self.bear_ms_count += 1
                self.ExternalTrend = 'Down Trend'
                self.LockBreak_M = self.Major_LowIndex
                self.LastSignalIndex = last_index
                logger.info(
                    f"Bearish Major {signal_type} detected (last candle): level={self.Major_LowLevel:.6f}, time={ohlc.index[last_index]}")
                self.Major_LowLevel = last_low
                self.Major_LowIndex = last_index
                self.Major_LowType = 'L'

        logger.info(f"Detected {len(pivot_high_points)} swing highs and {len(pivot_low_points)} swing lows")
        logger.info(f"Detected {self.bull_ms_count} bullish signals and {self.bear_ms_count} bearish signals")

        fig = None
        if plot:
            fig = self._plot_signals(ohlc, signals, pivot_high_points, pivot_low_points)

        alert_conditions = {
            "BULLISH_BOS": [s for s in signals if s["type"] == "BOS" and s["bias"] == BULLISH],
            "BEARISH_BOS": [s for s in signals if s["type"] == "BOS" and s["bias"] == BEARISH],
            "BULLISH_CHoCH": [s for s in signals if s["type"] == "CHoCH" and s["bias"] == BULLISH],
            "BEARISH_CHoCH": [s for s in signals if s["type"] == "CHoCH" and s["bias"] == BEARISH]
        }

        logger.info(f"Completed BOS and CHoCH calculation: "
                    f"{len(alert_conditions['BULLISH_BOS'])} bullish BOS, "
                    f"{len(alert_conditions['BEARISH_BOS'])} bearish BOS, "
                    f"{len(alert_conditions['BULLISH_CHoCH'])} bullish CHoCH, "
                    f"{len(alert_conditions['BEARISH_CHoCH'])} bearish CHoCH signals")
        return {"fig": fig, "signals": signals, "alert_conditions": alert_conditions}

    def _plot_signals(self, ohlc, signals, pivot_high_points, pivot_low_points):
        """
        Generate Plotly figure for BOS and CHoCH signals.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
        - signals (list): List of BOS/CHoCH signals with bias, time, fractal_time, level, type.
        - pivot_high_points (list): List of pivot high points with time, level, type.
        - pivot_low_points (list): List of pivot low points with time, level, type.

        Returns:
        - go.Figure: Plotly figure with candlesticks, pivots, and BOS/CHoCH signals.
        """
        logger.info("Generating Plotly figure for BOS and CHoCH signals")
        ohlc = ohlc.sort_index()

        # Log sample of index values for debugging
        logger.info(f"Sample ohlc.index values: {ohlc.index[-5:].tolist()}")

        # Limit to last 100 rows
        ohlc = ohlc.tail(self.plot_tail)

        # Filter pivots to ensure timestamps are within ohlc.index
        pivot_high_points = [p for p in pivot_high_points if p['time'] in ohlc.index]
        pivot_low_points = [p for p in pivot_low_points if p['time'] in ohlc.index]

        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'
        fig = go.Figure()  # Use go.Figure for larger plot area

        # Add OHLC candlestick
        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ))

        # Add pivot high points
        for pivot in pivot_high_points:
            fig.add_trace(go.Scatter(
                x=[pivot['time']],
                y=[pivot['level'] * 1.001],
                mode='markers+text',
                text=[pivot['type']],
                textposition='top center',
                marker=dict(symbol='triangle-down', size=8, color='#8E2043'),
                name=pivot['type'],
                showlegend=False
            ))

        # Add pivot low points
        for pivot in pivot_low_points:
            fig.add_trace(go.Scatter(
                x=[pivot['time']],
                y=[pivot['level'] * 0.999],
                mode='markers+text',
                text=[pivot['type']],
                textposition='bottom center',
                marker=dict(symbol='triangle-up', size=8, color='#027C5E'),
                name=pivot['type'],
                showlegend=False
            ))

        # Add signals
        for signal in signals:
            if (signal['bias'] == BULLISH and self.show_bull) or (signal['bias'] == BEARISH and self.show_bear):
                color = '#01AADD' if signal['bias'] == BULLISH else '#EFB465'
                fig.add_trace(go.Scatter(
                    x=[signal['fractal_time'], signal['time']],
                    y=[signal['level'], signal['level']],
                    mode='lines+text',
                    text=[signal['type'], ''],
                    textposition='bottom center' if signal['bias'] == BULLISH else 'top center',
                    line=dict(color=color, dash='solid'),
                    name=signal['type'],
                    showlegend=False
                ))

        last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} - BOS and CHoCH Signals | {candle_time_str}",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            xaxis=dict(range=[ohlc.index[0], ohlc.index[-1] + pd.Timedelta(minutes=45)])
        )
        fig.update_xaxes(
            title_text="Time",
            type='date',  # Ensure datetime handling
            tickformat="%H:%M",  # Show only time (HH:MM)
            tickmode='linear',  # Enforce strict dtick intervals
            dtick=7200000,  # 2-hour intervals
            tickangle=45,  # Rotate for readability
        )
        logger.info("Plotly figure generated successfully")
        return fig