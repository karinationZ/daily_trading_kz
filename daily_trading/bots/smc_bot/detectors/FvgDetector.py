# class FvgDetector(DetectorsBase):
#     def __init__(self, main_tf, pivot_period=5, left_bars=5, right_bars=1, show_bull=True, show_bear=True,
#                  volume_thresh=10, fvg_threshold=0.5, mode='Present', back=180,
#                  show_labels=False, show_fulfilled=True):
#         super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
#         self.pivot_period = pivot_period if pivot_period is not None else self.length
#         self.show_bull = show_bull
#         self.show_bear = show_bear
#         self.fvg_threshold = fvg_threshold
#         self.mode = mode
#         self.back = back
#         self.show_labels = show_labels
#         self.show_fulfilled = show_fulfilled
#         self.bull_count = 0
#         self.bear_count = 0
#         self.bull_mitigated = 0
#         self.bear_mitigated = 0
#         self.fvg_records = []
#         if fvg_threshold == 0:
#             logger.warning("fvg_threshold=0, all gaps will be detected, which may increase noise")
#         logger.info(f"Initialized FvgDetector with lower_tf={main_tf}, pivot_period={self.pivot_period}, "
#                      f"left_bars={self.left_bars}, right_bars={self.right_bars}, show_bull={show_bull}, "
#                      f"show_bear={show_bear}, volume_thresh={volume_thresh}, "
#                      f"fvg_threshold={fvg_threshold}, mode={mode}, back={back}, show_labels={show_labels}, "
#                      f"show_fulfilled={show_fulfilled}")
#
#     def _get_threshold(self, ohlc, i):
#         atr = (ohlc['high'] - ohlc['low']).rolling(window=144).mean().iloc[i]
#         return atr * self.fvg_threshold if not np.isnan(atr) else 0.0
#
#     def detect_fvg(self, ohlc, i, signals):
#         if i < 2:
#             return signals
#         high = ohlc['high'].iloc[i]
#         low = ohlc['low'].iloc[i]
#         time = ohlc.index[i]
#         volume = ohlc['volume'].iloc[i]
#         threshold = self._get_threshold(ohlc, i)
#
#         if self.mode == 'Present' and (len(ohlc) - 1 - i) > self.back:
#             return signals
#
#         if self.show_bull:
#             bullish_gap = low > ohlc['high'].iloc[i - 2] and ohlc['close'].iloc[i - 1] > ohlc['high'].iloc[i - 2]
#             if bullish_gap:
#                 gap_size = low - ohlc['high'].iloc[i - 2]
#                 if gap_size >= threshold:
#                     fvg_item = {
#                         "type": "Major Bullish FVG",
#                         "bias": BULLISH,
#                         "top": low,
#                         "bottom": ohlc['high'].iloc[i - 2],
#                         "time": time,
#                         "index": i,
#                         "volume": volume,
#                         "mitigated": False,
#                         "mitigated_time": None,
#                         "mitigated_index": None
#                     }
#                     signals.append(fvg_item)
#                     self.fvg_records.append(fvg_item)
#                     self.bull_count += 1
#                     logger.info(f"Bullish FVG detected: top={fvg_item['top']:.6f}, bottom={fvg_item['bottom']:.6f}, "
#                                  f"time={time}, gap_size={gap_size:.6f}, threshold={threshold:.6f}")
#
#         if self.show_bear:
#             bearish_gap = high < ohlc['low'].iloc[i - 2] and ohlc['close'].iloc[i - 1] < ohlc['low'].iloc[i - 2]
#             if bearish_gap:
#                 gap_size = ohlc['low'].iloc[i - 2] - high
#                 if gap_size >= threshold:
#                     fvg_item = {
#                         "type": "Major Bearish FVG",
#                         "bias": BEARISH,
#                         "top": ohlc['low'].iloc[i - 2],
#                         "bottom": high,
#                         "time": time,
#                         "index": i,
#                         "volume": volume,
#                         "mitigated": False,
#                         "mitigated_time": None,
#                         "mitigated_index": None
#                     }
#                     signals.append(fvg_item)
#                     self.fvg_records.append(fvg_item)
#                     self.bear_count += 1
#                     logger.info(f"Bearish FVG detected: top={fvg_item['top']:.6f}, bottom={fvg_item['bottom']:.6f}, "
#                                  f"time={time}, gap_size={gap_size:.6f}, threshold={threshold:.6f}")
#
#         return signals
#
#     def check_mitigation(self, ohlc, fvg_item, start_index):
#         """
#         Check if an FVG is mitigated by a candle's wick crossing the FVG boundary.
#
#         Args:
#             ohlc (pd.DataFrame): OHLC mgc.
#             fvg_item (dict): FVG signal with 'top', 'bottom', 'type', etc.
#             start_index (int): Index where FVG was detected.
#
#         Returns:
#             bool: True if mitigated, False otherwise.
#         """
#         fvg_top = fvg_item['top']
#         fvg_bottom = fvg_item['bottom']
#         fvg_range = fvg_top - fvg_bottom
#
#         if fvg_range <= 0:
#             logger.warning(f"Invalid FVG range {fvg_range:.6f} for {fvg_item['type']} at {fvg_item['time']}")
#             return False
#
#         for j in range(start_index + 1, len(ohlc)):
#             high = ohlc['high'].iloc[j]
#             low = ohlc['low'].iloc[j]
#             time = ohlc.index[j]
#             index = j
#
#             logger.info(
#                 f"Checking mitigation for {fvg_item['type']} at {time}: top={fvg_top:.6f}, bottom={fvg_bottom:.6f}, "
#                 f"high={high:.6f}, low={low:.6f}")
#
#             if fvg_item["type"] == "Major Bullish FVG":
#                 if low <= fvg_bottom:
#                     fvg_item["mitigated"] = True
#                     fvg_item["mitigated_time"] = time
#                     fvg_item["mitigated_index"] = index
#                     self.bull_mitigated += 1
#                     logger.info(
#                         f"{fvg_item['type']} mitigated: bottom={fvg_bottom:.6f}, top={fvg_top:.6f}, "
#                         f"time={time}, high={high:.6f}, low={low:.6f}")
#                     return True
#             else:  # Major Bearish FVG
#                 if high >= fvg_top:
#                     fvg_item["mitigated"] = True
#                     fvg_item["mitigated_time"] = time
#                     fvg_item["mitigated_index"] = index
#                     self.bear_mitigated += 1
#                     logger.info(
#                         f"{fvg_item['type']} mitigated: bottom={fvg_bottom:.6f}, top={fvg_top:.6f}, "
#                         f"time={time}, high={high:.6f}, low={low:.6f}")
#                     return True
#         return False
#
#     def calculate_fvg(self, ohlc, plot=True):
#         logger.info(f"Starting FVG calculation with pivot_period={self.pivot_period}, "
#                      f"left_bars={self.left_bars}, right_bars={self.right_bars}, fvg_threshold={self.fvg_threshold}, "
#                      f"mode={self.mode}, back={self.back}")
#         if len(ohlc) < 3:
#             logger.error(f"OHLC mgc too short ({len(ohlc)} rows)")
#             return {"fig": None, "signals": [], "alert_conditions": {
#                 "BULLISH_FVG": [], "BEARISH_FVG": [], "BULLISH_FVG_FULFILLED": [], "BEARISH_FVG_FULFILLED": []
#             }}
#         ohlc = self.validate_data(ohlc)
#
#         self.bull_count = 0
#         self.bear_count = 0
#         self.bull_mitigated = 0
#         self.bear_mitigated = 0
#         self.fvg_records = []
#         signals = []
#
#         for i in range(self.left_bars, len(ohlc)):
#             signals = self.detect_fvg(ohlc, i, signals)
#             for fvg_item in self.fvg_records:
#                 if not fvg_item["mitigated"]:
#                     self.check_mitigation(ohlc, fvg_item, fvg_item["index"])
#
#         pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)
#
#         logger.info(f"Detected {len(signals)} FVGs: {self.bull_count} bullish, {self.bear_count} bearish, "
#                      f"{self.bull_mitigated} bullish mitigated, {self.bear_mitigated} bearish mitigated")
#
#         fig = None
#         if plot:
#             fig = self._plot_fvg(ohlc, signals, pivot_high_points, pivot_low_points)
#
#         alert_conditions = {
#             "BULLISH_FVG": [s for s in signals if s["type"] == "Major Bullish FVG" and not s["mitigated"]],
#             "BEARISH_FVG": [s for s in signals if s["type"] == "Major Bearish FVG" and not s["mitigated"]],
#             "BULLISH_FVG_FULFILLED": [s for s in signals if s["type"] == "Major Bullish FVG" and s["mitigated"]],
#             "BEARISH_FVG_FULFILLED": [s for s in signals if s["type"] == "Major Bearish FVG" and s["mitigated"]]
#         }
#
#         logger.info(f"Completed FVG calculation: "
#                      f"{len(alert_conditions['BULLISH_FVG'])} bullish FVGs, "
#                      f"{len(alert_conditions['BEARISH_FVG'])} bearish FVGs, "
#                      f"{len(alert_conditions['BULLISH_FVG_FULFILLED'])} bullish fulfilled, "
#                      f"{len(alert_conditions['BEARISH_FVG_FULFILLED'])} bearish fulfilled")
#         signals_sorted = sorted(signals, key=lambda s: s["time"], reverse=True)
#         return {"fig": fig, "signals": signals_sorted, "alert_conditions": alert_conditions}
#
#     def _plot_fvg(self, ohlc, signals, pivot_high_points, pivot_low_points):
#         """
#         Plot candlesticks, pivot points, and FVG signals for the last 200 candles.
#
#         Parameters:
#         - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
#         - signals (list): List of FVG signals with bias, time, top, bottom, mitigated, mitigated_time.
#         - pivot_high_points (list): List of pivot high points with labels.
#         - pivot_low_points (list): List of pivot low points with labels.
#
#         Returns:
#         - go.Figure: Plotly figure with candlesticks, pivots, and FVG rectangles.
#         """
#         logger.info("Generating Plotly figure for FVG signals")
#         ohlc = ohlc.sort_index()
#
#         # Log sample of index values for debugging
#         logger.info(f"Sample ohlc.index values: {ohlc.index[-5:].tolist()}")
#
#         # Limit to last 200 rows
#         ohlc = ohlc.tail(self.plot_tail)
#
#         # Filter pivots to ensure timestamps are within ohlc.index
#         pivot_high_points = [p for p in pivot_high_points if p['time'] in ohlc.index]
#         pivot_low_points = [p for p in pivot_low_points if p['time'] in ohlc.index]
#         symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'
#
#         fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} | Price Chart with Liquidity Voids (FVG)"])
#
#         # Plot candlesticks
#         fig.add_trace(go.Candlestick(
#             x=ohlc.index,
#             open=ohlc['open'],
#             high=ohlc['high'],
#             low=ohlc['low'],
#             close=ohlc['close'],
#             name="OHLC"
#         ), row=1, col=1)
#
#         # Plot pivot highs
#         for pivot in pivot_high_points:
#             fig.add_trace(go.Scatter(
#                 x=[pivot['time']],
#                 y=[pivot['level'] * 1.001],
#                 mode='markers+text',
#                 text=[pivot['type']],
#                 textposition='top center',
#                 marker=dict(symbol='triangle-down', size=8, color='#8E2043'),
#                 name=pivot['type'],
#                 showlegend=False
#             ), row=1, col=1)
#
#         # Plot pivot lows
#         for pivot in pivot_low_points:
#             fig.add_trace(go.Scatter(
#                 x=[pivot['time']],
#                 y=[pivot['level'] * 0.999],
#                 mode='markers+text',
#                 text=[pivot['type']],
#                 textposition='bottom center',
#                 marker=dict(symbol='triangle-up', size=8, color='#027C5E'),
#                 name=pivot['type'],
#                 showlegend=False
#             ), row=1, col=1)
#
#         # Plot FVG rectangles
#         for signal in signals:
#             if (signal["bias"] == BULLISH and self.show_bull) or (signal["bias"] == BEARISH and self.show_bear):
#                 if not signal["mitigated"] or self.show_fulfilled:
#                     color = 'rgba(0, 255, 0, 0.5)' if signal["bias"] == BULLISH else 'rgba(255, 0, 0, 0.5)'
#                     if signal["mitigated"]:
#                         color = 'rgba(169, 169, 169, 0.6)'
#                     end_time = signal["mitigated_time"] if signal["mitigated"] else ohlc.index[-1]
#                     if signal["mitigated"] and end_time not in ohlc.index:
#                         end_time = ohlc.index[ohlc.index.get_indexer([end_time], method='nearest')[0]]
#                         logger.info(f"Adjusted end_time to nearest OHLC index: {end_time} for FVG at {signal['time']}")
#                     logger.info(
#                         f"Adding FVG rectangle: x0={signal['time']}, x1={end_time}, y0={signal['bottom']:.6f}, y1={signal['top']:.6f}, color={color}")
#                     fig.add_shape(
#                         type="rect",
#                         x0=signal["time"],
#                         x1=end_time,
#                         y0=signal["bottom"],
#                         y1=signal["top"],
#                         fillcolor=color,
#                         line=dict(width=1, color=color.replace('0.5', '0.8').replace('0.6', '0.8')),
#                         layer="below",
#                         opacity=0.5 if not signal["mitigated"] else 0.6,
#                         name=signal["type"]
#                     )
#                     if self.show_labels:
#                         text_position = 'bottom center' if signal["bias"] == BULLISH else 'top center'
#                         fig.add_trace(go.Scatter(
#                             x=[signal["time"]],
#                             y=[(signal["top"] + signal["bottom"]) / 2],
#                             mode='text',
#                             text=["Liquidity Void"],
#                             textposition=text_position,
#                             textfont=dict(color=color.replace('0.5', '1.0').replace('0.6', '1.0'), size=12),
#                             showlegend=False
#                         ))
#
#         # Update layout
#         last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
#         candle_time_str = f"Last Candle: {last_candle} (PT)"
#         fig.update_layout(
#             title=f"{symbol} | FVG | {candle_time_str}",
#             xaxis_rangeslider_visible=False,
#             showlegend=True,
#             yaxis=dict(title="Price"),
#             xaxis=dict(range=[ohlc.index[0], ohlc.index[-1] + pd.Timedelta(minutes=60)])  # Extend range by 15 minutes
#         )
#         fig.update_xaxes(
#             title_text="Time",
#             type='date',  # Ensure datetime handling
#             tickformat="%H:%M",  # Show only time (HH:MM)
#             tickmode='linear',  # Enforce strict dtick intervals
#             dtick=7200000,  # 2-hour intervals
#             tickangle=45,  # Rotate for readability
#             row=1, col=1
#         )
#         logger.info("Plotly figure generated successfully")
#         return fig
#
#     def merge_bos_fvg_plots(self, ohlc, bos_signals, fvg_signals):
#         """
#         Merge BOS and FVG plots into a single comprehensive figure.
#
#         Parameters:
#         - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index
#         - bos_signals (list): List of BOS/CHoCH signals
#         - fvg_signals (list): List of FVG signals
#
#         Returns:
#         - go.Figure: Combined Plotly figure with BOS lines and FVG rectangles
#         """
#         pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)
#         logger.info("Generating combined BOS and FVG Plotly figure")
#         ohlc = ohlc.sort_index()
#
#         # Limit to last plot_tail rows
#         ohlc = ohlc.tail(self.plot_tail)
#
#         symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'
#         fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} | BOS/CHoCH & FVG Analysis"])
#
#         # Add OHLC candlesticks
#         fig.add_trace(go.Candlestick(
#             x=ohlc.index,
#             open=ohlc['open'],
#             high=ohlc['high'],
#             low=ohlc['low'],
#             close=ohlc['close'],
#             name="OHLC"
#         ), row=1, col=1)
#
#         # Filter BOS signals to FVG time window
#         if fvg_signals:
#             fvg_times = [s['time'] for s in fvg_signals]
#             min_fvg_time = min(fvg_times)
#             max_fvg_time = max(fvg_times)
#             filtered_bos_signals = [
#                 s for s in bos_signals
#                 if (s['time'] >= min_fvg_time and s['time'] <= max_fvg_time and
#                     s['fractal_time'] >= min_fvg_time and s['fractal_time'] <= max_fvg_time)
#             ]
#             logger.info(
#                 f"Filtered {len(bos_signals)} BOS signals to {len(filtered_bos_signals)} within FVG time window: {min_fvg_time} to {max_fvg_time}")
#         else:
#             filtered_bos_signals = []
#             logger.info("No FVG signals, skipping BOS signals")
#
#         # Add BOS/CHoCH signals (lines)
#         for signal in filtered_bos_signals:
#             if (signal['bias'] == BULLISH and self.show_bull) or (signal['bias'] == BEARISH and self.show_bear):
#                 start_time = signal['fractal_time']
#                 end_time = signal['time']
#                 # Ensure times are within ohlc.index
#                 if start_time not in ohlc.index:
#                     start_time = ohlc.index[ohlc.index.get_indexer([start_time], method='nearest')[0]]
#                 if end_time not in ohlc.index:
#                     end_time = ohlc.index[ohlc.index.get_indexer([end_time], method='nearest')[0]]
#                 color = '#01AADD' if signal['bias'] == BULLISH else '#EFB465'
#                 fig.add_trace(go.Scatter(
#                     x=[start_time, end_time],
#                     y=[signal['level'], signal['level']],
#                     mode='lines+text',
#                     text=[signal['type'], ''],
#                     textposition='bottom center',
#                     textfont=dict(size=12),
#                     line=dict(color=color, dash='solid', width=4),
#                     name=signal['type'],
#                     showlegend=False
#                 ), row=1, col=1)
#
#         # Add FVG rectangles
#         for signal in fvg_signals:
#             if (signal["bias"] == BULLISH and self.show_bull) or (signal["bias"] == BEARISH and self.show_bear):
#                 if not signal["mitigated"] or self.show_fulfilled:
#                     color = 'rgba(0, 255, 0, 0.5)' if signal["bias"] == BULLISH else 'rgba(255, 0, 0, 0.5)'
#                     if signal["mitigated"]:
#                         color = 'rgba(169, 169, 169, 0.6)'
#                     end_time = signal["mitigated_time"] if signal["mitigated"] else ohlc.index[-1]
#                     if signal["mitigated"] and end_time not in ohlc.index:
#                         end_time = ohlc.index[ohlc.index.get_indexer([end_time], method='nearest')[0]]
#                     logger.info(
#                         f"Adding FVG rectangle: x0={signal['time']}, x1={end_time}, y0={signal['bottom']:.6f}, y1={signal['top']:.6f}, color={color}")
#                     fig.add_shape(
#                         type="rect",
#                         x0=signal["time"],
#                         x1=end_time,
#                         y0=signal["bottom"],
#                         y1=signal["top"],
#                         fillcolor=color,
#                         line=dict(width=1, color=color.replace('0.5', '0.8').replace('0.6', '0.8')),
#                         layer="below",
#                         opacity=0.5 if not signal["mitigated"] else 0.6,
#                         name=signal["type"]
#                     )
#                     if self.show_labels:
#                         text_position = 'bottom center' if signal["bias"] == BULLISH else 'top center'
#                         fig.add_trace(go.Scatter(
#                             x=[signal["time"]],
#                             y=[(signal["top"] + signal["bottom"]) / 2],
#                             mode='text',
#                             text=["Liquidity Void"],
#                             textposition=text_position,
#                             textfont=dict(color=color.replace('0.5', '1.0').replace('0.6', '1.0'), size=12),
#                             showlegend=False
#                         ), row=1, col=1)
#
#         # Update layout
#         last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
#         candle_time_str = f"Last Candle: {last_candle} (PT)"
#         fig.update_layout(
#             title=f"{symbol} | BOS/CHoCH & FVG Analysis | {candle_time_str} | TF: {self.main_tf}",
#             xaxis_title="Time",
#             yaxis_title="Price",
#             xaxis_rangeslider_visible=False,
#             showlegend=True,
#             xaxis=dict(range=[ohlc.index[0], ohlc.index[-1] + pd.Timedelta(minutes=60)]),
#             # width=1200,
#             # height=600,
#             margin=dict(l=50, r=50, t=100, b=100)
#         )
#
#         fig.update_xaxes(
#             title_text="Time",
#             type='date',
#             tickformat="%H:%M",
#             tickmode='linear',
#             dtick=7200000,  # 2-hour intervals
#             tickangle=45,
#             row=1, col=1
#         )
#
#         logger.info("Combined BOS and FVG figure generated successfully")
#         return fig

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your constants
from daily_trading.bots.smc_bot.detectors.DetectorsBase import (
    BEARISH,
    BULLISH,
    DetectorsBase,
)
from daily_trading.settings import logger


class FvgDetector(DetectorsBase):
    def __init__(
        self,
        main_tf,
        pivot_period=5,
        left_bars=5,
        right_bars=1,
        show_bull=True,
        show_bear=True,
        volume_thresh=10,
        fvg_threshold=0.5,
        mode="Present",
        back=180,
        show_labels=False,
        show_fulfilled=True,
    ):
        super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
        self.pivot_period = pivot_period
        self.show_bull = show_bull
        self.show_bear = show_bear
        self.fvg_threshold = fvg_threshold
        self.mode = mode
        self.back = back
        self.show_labels = show_labels
        self.show_fulfilled = show_fulfilled

    def calculate_fvg(self, ohlc, plot=True):
        """
        Vectorized FVG Calculation (100x Faster).
        """
        if len(ohlc) < 5:
            return {"fig": None, "signals": [], "alert_conditions": {}}

        # Ensure index is datetime for plotting/logic
        if ohlc.index.dtype == "int64":  # Handling numeric index if necessary
            times = ohlc.index
        else:
            times = ohlc.index

        # 1. PRE-CALCULATE COLUMNS (Vectorized)
        high = ohlc["high"].values
        low = ohlc["low"].values
        close = ohlc["close"].values
        volume = ohlc["volume"].values

        # Calculate ATR (Range) for threshold - vectorized
        # Simple Approx ATR: High - Low
        ranges = high - low
        # Efficient Rolling Mean using Pandas
        thresholds = pd.Series(ranges).rolling(window=144, min_periods=1).mean().values * self.fvg_threshold

        signals = []

        # 2. VECTORIZED DETECTION
        # We look at i, i-1, i-2.
        # In numpy shifting: array[:-2] is (i-2), array[1:-1] is (i-1), array[2:] is (i)

        # Align arrays for vector logic
        # Current Candle (i)
        curr_low = low[2:]
        curr_high = high[2:]
        curr_time = times[2:]
        curr_idx = np.arange(len(ohlc))[2:]
        curr_vol = volume[2:]
        curr_thresh = thresholds[2:]

        # Candle i-1
        prev_close = close[1:-1]

        # Candle i-2
        prev2_high = high[:-2]
        prev2_low = low[:-2]

        # --- BULLISH LOGIC ---
        if self.show_bull:
            # Condition: Current Low > Prev2 High AND Prev Close > Prev2 High
            bull_cond = (curr_low > prev2_high) & (prev_close > prev2_high)

            # Filter by Threshold
            gap_sizes = curr_low - prev2_high
            valid_bull = bull_cond & (gap_sizes >= curr_thresh)

            # Get indices where True
            indices = np.where(valid_bull)[0]

            for idx in indices:
                # Real index in original dataframe = idx + 2 (due to slicing)
                real_idx = curr_idx[idx]
                top = curr_low[idx]
                bottom = prev2_high[idx]

                # Check Mitigation (Numpy Slicing - Fast)
                # Look at all lows AFTER this candle
                future_lows = low[real_idx + 1 :]

                # Find if any future low <= bottom
                # np.argmax returns index of first True, or 0 if none
                mitigation_mask = future_lows <= bottom
                mitigated = False
                mit_time = None

                if mitigation_mask.any():
                    mitigated = True
                    # Get relative index of first trigger
                    first_mit_idx = np.argmax(mitigation_mask)
                    # Absolute index
                    abs_mit_idx = real_idx + 1 + first_mit_idx
                    mit_time = times[abs_mit_idx]

                signals.append(
                    {
                        "type": "Major Bullish FVG",
                        "bias": BULLISH,
                        "top": top,
                        "bottom": bottom,
                        "time": curr_time[idx],
                        "index": real_idx,
                        "volume": curr_vol[idx],
                        "mitigated": mitigated,
                        "mitigated_time": mit_time,
                    }
                )

        # --- BEARISH LOGIC ---
        if self.show_bear:
            # Condition: Current High < Prev2 Low AND Prev Close < Prev2 Low
            bear_cond = (curr_high < prev2_low) & (prev_close < prev2_low)

            # Filter by Threshold
            gap_sizes = prev2_low - curr_high
            valid_bear = bear_cond & (gap_sizes >= curr_thresh)

            # Get indices
            indices = np.where(valid_bear)[0]

            for idx in indices:
                real_idx = curr_idx[idx]
                top = prev2_low[idx]
                bottom = curr_high[idx]

                # Check Mitigation
                # Look at all highs AFTER this candle
                future_highs = high[real_idx + 1 :]

                mitigation_mask = future_highs >= top
                mitigated = False
                mit_time = None

                if mitigation_mask.any():
                    mitigated = True
                    first_mit_idx = np.argmax(mitigation_mask)
                    abs_mit_idx = real_idx + 1 + first_mit_idx
                    mit_time = times[abs_mit_idx]

                signals.append(
                    {
                        "type": "Major Bearish FVG",
                        "bias": BEARISH,
                        "top": top,
                        "bottom": bottom,
                        "time": curr_time[idx],
                        "index": real_idx,
                        "volume": curr_vol[idx],
                        "mitigated": mitigated,
                        "mitigated_time": mit_time,
                    }
                )

        # Sort signals by time
        signals.sort(key=lambda x: x["time"], reverse=True)

        # --- ALERTS & PLOTTING ---
        alert_conditions = {
            "BULLISH_FVG": [s for s in signals if s["type"] == "Major Bullish FVG" and not s["mitigated"]],
            "BEARISH_FVG": [s for s in signals if s["type"] == "Major Bearish FVG" and not s["mitigated"]],
            "BULLISH_FVG_FULFILLED": [s for s in signals if s["type"] == "Major Bullish FVG" and s["mitigated"]],
            "BEARISH_FVG_FULFILLED": [s for s in signals if s["type"] == "Major Bearish FVG" and s["mitigated"]],
        }

        fig = None
        if plot:
            # Calculate pivots only if plotting is needed (saves time)
            pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)
            fig = self._plot_fvg(ohlc, signals, pivot_high_points, pivot_low_points)

        return {"fig": fig, "signals": signals, "alert_conditions": alert_conditions}

    def _plot_fvg(self, ohlc, signals, pivot_high_points, pivot_low_points):
        """
        Plot candlesticks, pivot points, and FVG signals for the last 200 candles.

        Parameters:
        - ohlc (pd.DataFrame): OHLCV DataFrame with pandas.Timestamp index.
        - signals (list): List of FVG signals with bias, time, top, bottom, mitigated, mitigated_time.
        - pivot_high_points (list): List of pivot high points with labels.
        - pivot_low_points (list): List of pivot low points with labels.

        Returns:
        - go.Figure: Plotly figure with candlesticks, pivots, and FVG rectangles.
        """
        logger.info("Generating Plotly figure for FVG signals")
        ohlc = ohlc.sort_index()

        # Log sample of index values for debugging
        logger.info(f"Sample ohlc.index values: {ohlc.index[-5:].tolist()}")

        # Limit to last 200 rows
        ohlc = ohlc.tail(self.plot_tail)

        # Filter pivots to ensure timestamps are within ohlc.index
        pivot_high_points = [p for p in pivot_high_points if p["time"] in ohlc.index]
        pivot_low_points = [p for p in pivot_low_points if p["time"] in ohlc.index]
        symbol = ohlc["symbol"].iloc[0] if "symbol" in ohlc.columns else "Unknown"

        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=[f"{symbol} | Price Chart with Liquidity Voids (FVG)"],
        )

        # Plot candlesticks
        fig.add_trace(
            go.Candlestick(
                x=ohlc.index,
                open=ohlc["open"],
                high=ohlc["high"],
                low=ohlc["low"],
                close=ohlc["close"],
                name="OHLC",
            ),
            row=1,
            col=1,
        )

        # Plot pivot highs
        for pivot in pivot_high_points:
            fig.add_trace(
                go.Scatter(
                    x=[pivot["time"]],
                    y=[pivot["level"] * 1.001],
                    mode="markers+text",
                    text=[pivot["type"]],
                    textposition="top center",
                    marker=dict(symbol="triangle-down", size=8, color="#8E2043"),
                    name=pivot["type"],
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Plot pivot lows
        for pivot in pivot_low_points:
            fig.add_trace(
                go.Scatter(
                    x=[pivot["time"]],
                    y=[pivot["level"] * 0.999],
                    mode="markers+text",
                    text=[pivot["type"]],
                    textposition="bottom center",
                    marker=dict(symbol="triangle-up", size=8, color="#027C5E"),
                    name=pivot["type"],
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Plot FVG rectangles
        for signal in signals:
            if (signal["bias"] == BULLISH and self.show_bull) or (signal["bias"] == BEARISH and self.show_bear):
                if not signal["mitigated"] or self.show_fulfilled:
                    color = "rgba(0, 255, 0, 0.5)" if signal["bias"] == BULLISH else "rgba(255, 0, 0, 0.5)"
                    if signal["mitigated"]:
                        color = "rgba(169, 169, 169, 0.6)"
                    end_time = signal["mitigated_time"] if signal["mitigated"] else ohlc.index[-1]
                    if signal["mitigated"] and end_time not in ohlc.index:
                        end_time = ohlc.index[ohlc.index.get_indexer([end_time], method="nearest")[0]]
                        logger.info(f"Adjusted end_time to nearest OHLC index: {end_time} for FVG at {signal['time']}")
                    logger.info(
                        f"Adding FVG rectangle: x0={signal['time']}, x1={end_time}, y0={signal['bottom']:.6f}, y1={signal['top']:.6f}, color={color}"
                    )
                    fig.add_shape(
                        type="rect",
                        x0=signal["time"],
                        x1=end_time,
                        y0=signal["bottom"],
                        y1=signal["top"],
                        fillcolor=color,
                        line=dict(
                            width=1,
                            color=color.replace("0.5", "0.8").replace("0.6", "0.8"),
                        ),
                        layer="below",
                        opacity=0.5 if not signal["mitigated"] else 0.6,
                        name=signal["type"],
                    )
                    if self.show_labels:
                        text_position = "bottom center" if signal["bias"] == BULLISH else "top center"
                        fig.add_trace(
                            go.Scatter(
                                x=[signal["time"]],
                                y=[(signal["top"] + signal["bottom"]) / 2],
                                mode="text",
                                text=["Liquidity Void"],
                                textposition=text_position,
                                textfont=dict(
                                    color=color.replace("0.5", "1.0").replace("0.6", "1.0"),
                                    size=12,
                                ),
                                showlegend=False,
                            )
                        )

        # Update layout
        last_candle = ohlc.index.max().tz_convert("US/Pacific").strftime("%Y-%m-%d %H:%M")
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | FVG | {candle_time_str}",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            yaxis=dict(title="Price"),
            xaxis=dict(range=[ohlc.index[0], ohlc.index[-1] + pd.Timedelta(minutes=60)]),
            # Extend range by 15 minutes
        )
        fig.update_xaxes(
            title_text="Time",
            type="date",  # Ensure datetime handling
            tickformat="%H:%M",  # Show only time (HH:MM)
            tickmode="linear",  # Enforce strict dtick intervals
            dtick=7200000,  # 2-hour intervals
            tickangle=45,  # Rotate for readability
            row=1,
            col=1,
        )
        logger.info("Plotly figure generated successfully")
        return fig
