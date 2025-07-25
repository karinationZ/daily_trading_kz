import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.detectors.SMCBase import SMCBase, Trend
from daily_trading.settings import logger


class OrderBlocksDetector(SMCBase):
    def __init__(self, main_tf, left_bars=5, right_bars=5, tap_back_ratio=0.1):
        super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
        self.tap_back_ratio = tap_back_ratio
        logger.debug(
            f"Initialized OrderBlocksDetector with main_tf={main_tf}, tap_back_ratio={tap_back_ratio}.")

    def detect_order_block(self, ohlc: pd.DataFrame) -> list:
        """
        Detects Order Blocks (OBs) on 1H timeframe as the last opposing candle before a break of structure.
        OBs are mitigated if price exits (closes beyond block) and taps back into 10% of the block's range.

        Parameters:
        - ohlc: DataFrame with 'open', 'high', 'low', 'close' columns

        Returns:
        - List of dicts: {"type": str, "top": float, "bottom": float, "time": Timestamp, "index": int, "breaker": bool, "break_loc": Timestamp/None}
        """
        logger.debug(f"Detecting Order Blocks (OB) on OHLC data with {len(ohlc)} rows")
        ohlc = self.validate_data(ohlc)
        pivot_highs, pivot_lows = self.calculate_pivots(ohlc)
        bullish_ob = []
        bearish_ob = []
        min_distance = 6  # Minimum candles between OB and pivot break

        crossed_highs = {p["index"]: False for p in pivot_highs}
        crossed_lows = {p["index"]: False for p in pivot_lows}

        for i in range(1, len(ohlc)):
            current_price = ohlc['close'].iloc[i]

            # Detect bullish OB: last bearish candle before breaking swing high
            for pivot in pivot_highs:
                if pivot["index"] < i and not crossed_highs[pivot["index"]]:
                    if current_price > pivot["level"]:
                        crossed_highs[pivot["index"]] = True
                        # Find last bearish candle (close < open) before break
                        for j in range(i - 1, max(pivot["index"] - 1, i - min_distance - 1), -1):
                            if ohlc['close'].iloc[j] < ohlc['open'].iloc[j]:
                                ob_item = {
                                    "type": "bullish",
                                    "top": ohlc['high'].iloc[j],
                                    "bottom": ohlc['low'].iloc[j],
                                    "time": ohlc.index[j],
                                    "index": j,  # Use OB candle's index
                                    "breaker": False,
                                    "break_loc": None
                                }
                                bullish_ob.append(ob_item)
                                logger.debug(
                                    f"Bullish OB detected at {ohlc.index[j]}: top={ob_item['top']:.6f}, bottom={ob_item['bottom']:.6f}")
                                break

            # Detect bearish OB: last bullish candle before breaking swing low
            for pivot in pivot_lows:
                if pivot["index"] < i and not crossed_lows[pivot["index"]]:
                    if current_price < pivot["level"]:
                        crossed_lows[pivot["index"]] = True
                        # Find last bullish candle (close > open) before break
                        for j in range(i - 1, max(pivot["index"] - 1, i - min_distance - 1), -1):
                            if ohlc['close'].iloc[j] > ohlc['open'].iloc[j]:
                                ob_item = {
                                    "type": "bearish",
                                    "top": ohlc['high'].iloc[j],
                                    "bottom": ohlc['low'].iloc[j],
                                    "time": ohlc.index[j],
                                    "index": j,  # Use OB candle's index
                                    "breaker": False,
                                    "break_loc": None
                                }
                                bearish_ob.append(ob_item)
                                logger.debug(
                                    f"Bearish OB detected at {ohlc.index[j]}: top={ob_item['top']:.6f}, bottom={ob_item['bottom']:.6f}")
                                break

        # Process mitigation
        for ob in bullish_ob:
            for i in range(ob["index"] + 1, len(ohlc)):
                if ohlc.index[i] <= ob["time"]:
                    continue
                # Check exit (close above top)
                if not ob["breaker"] and ohlc['close'].iloc[i] > ob["top"]:
                    ob["exited"] = True
                    logger.debug(
                        f"Bullish OB exited at {ohlc.index[i]}: top={ob['top']:.6f}, close={ohlc['close'].iloc[i]:.6f}")
                # Check 10% tap-back after exit
                elif ob.get("exited", False):
                    block_size = ob["top"] - ob["bottom"]
                    tap_threshold = ob["top"] - self.tap_back_ratio * block_size
                    if ohlc['low'].iloc[i] <= tap_threshold:
                        ob["breaker"] = True
                        ob["break_loc"] = ohlc.index[i]
                        logger.debug(
                            f"Bullish OB mitigated at {ohlc.index[i]}: top={ob['top']:.6f}, bottom={ob['bottom']:.6f}, tap_threshold={tap_threshold:.6f}")
                        break

        for ob in bearish_ob[:]:
            for i in range(ob["index"] + 1, len(ohlc)):
                if ohlc.index[i] <= ob["time"]:
                    continue
                # Check exit (close below bottom)
                if not ob["breaker"] and ohlc['close'].iloc[i] < ob["bottom"]:
                    ob["exited"] = True
                    logger.debug(
                        f"Bearish OB exited at {ohlc.index[i]}: bottom={ob['bottom']:.6f}, close={ohlc['close'].iloc[i]:.6f}")
                # Check 10% tap-back after exit
                elif ob.get("exited", False):
                    block_size = ob["top"] - ob["bottom"]
                    tap_threshold = ob["bottom"] + self.tap_back_ratio * block_size
                    if ohlc['high'].iloc[i] >= tap_threshold:
                        ob["breaker"] = True
                        ob["break_loc"] = ohlc.index[i]
                        logger.debug(
                            f"Bearish OB mitigated at {ohlc.index[i]}: top={ob['top']:.6f}, bottom={ob['bottom']:.6f}, tap_threshold={tap_threshold:.6f}")
                        break
            # Remove if latest close is beyond block
            if ob["breaker"] and ohlc['close'].iloc[-1] < ob["bottom"]:
                bearish_ob.remove(ob)
                logger.debug(
                    f"Bearish OB removed: bottom={ob['bottom']:.6f}, current_price={ohlc['close'].iloc[-1]:.6f}")

        # Remove bullish OBs if latest close is beyond block
        bullish_ob = [ob for ob in bullish_ob if not (ob["breaker"] and ohlc['close'].iloc[-1] > ob["top"])]
        order_blocks = bullish_ob + bearish_ob
        logger.debug(f"Detected {len(order_blocks)} OBs: {len(bullish_ob)} bullish, {len(bearish_ob)} bearish")
        return order_blocks

    def calculate_order_blocks(self, ohlc, plot=True):
        order_blocks = self.detect_order_block(ohlc)
        fig = None
        if plot:
            fig = self._plot_order_blocks(ohlc, order_blocks)

        logger.debug(f"OB: {order_blocks}")
        return {"fig": fig, "order_blocks": order_blocks}

    def _plot_order_blocks(self, ohlc, order_blocks):
        logger.debug("Generating Plotly figure for Order Blocks")

        ohlc = self.validate_data(ohlc)
        # Limit to last X rows
        ohlc = ohlc.tail(self.plot_tail)
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'

        # Filter order blocks to ensure timestamps are within ohlc.index
        order_blocks = [ob for ob in order_blocks if ob['time'] in ohlc.index]

        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} | Price Chart with Order Blocks"])

        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ), row=1, col=1)

        pivot_highs, pivot_lows = self.calculate_pivots(ohlc)
        for pivot in pivot_highs:
            fig.add_trace(go.Scatter(
                x=[pivot['time']],
                y=[pivot['level'] * 1.001],
                mode='markers+text',
                text=[pivot['type']],
                textposition='top center',
                marker=dict(symbol='triangle-down', size=8, color='#8E2043'),
                name=pivot['type'],
                showlegend=False
            ), row=1, col=1)

        for pivot in pivot_lows:
            fig.add_trace(go.Scatter(
                x=[pivot['time']],
                y=[pivot['level'] * 0.999],
                mode='markers+text',
                text=[pivot['type']],
                textposition='bottom center',
                marker=dict(symbol='triangle-up', size=8, color='#027C5E'),
                name=pivot['type'],
                showlegend=False
            ), row=1, col=1)

        for ob in order_blocks:
            if ob["breaker"]:
                fillcolor = "rgba(0, 0, 0, 0)"  # Transparent for fulfilled blocks
                line_color = "rgba(255, 0, 0, 0.8)" if ob["type"] == "bearish" else "rgba(0, 0, 255, 0.8)"
                line_dash = "dash"
            else:
                fillcolor = "rgba(255, 0, 0, 0.5)" if ob["type"] == "bearish" else "rgba(0, 0, 255, 0.5)"
                line_color = "rgba(255, 0, 0, 0.8)" if ob["type"] == "bearish" else "rgba(0, 0, 255, 0.8)"
                line_dash = "solid"

            end_time = ob["break_loc"] if ob["breaker"] else ohlc.index[-1]
            if ob["breaker"] and end_time not in ohlc.index:
                end_time = ohlc.index[ohlc.index.get_loc(end_time, method='nearest')]
                logger.debug(f"Adjusted end_time to nearest OHLC index: {end_time} for OB at {ob['time']}")
            logger.debug(
                f"Adding OB rectangle: x0={ob['time']}, x1={end_time}, y0={ob['bottom']:.6f}, y1={ob['top']:.6f}, "
                f"fillcolor={fillcolor}, line_color={line_color}, line_dash={line_dash}")
            fig.add_shape(
                type="rect",
                x0=ob["time"],
                x1=end_time,
                y0=ob["bottom"],
                y1=ob["top"],
                fillcolor=fillcolor,
                line=dict(width=1, color=line_color, dash=line_dash),
                layer="below",
                opacity=0.5 if not ob["breaker"] else 1.0,
                name=ob["type"]
            )

        last_candle = ohlc.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | Order Blocks | {candle_time_str} | TF: {self.main_tf}",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            yaxis=dict(title="Price"),
            xaxis=dict(range=[ohlc.index[0], ohlc.index[-1] + pd.Timedelta(minutes=60)])  # Extend range by 15 minutes
        )
        fig.update_xaxes(
            title_text="Time",
            type='date',  # Ensure datetime handling
            tickformat="%H:%M",  # Show only time (HH:MM)
            tickmode='linear',  # Enforce strict dtick intervals
            dtick=7200000,  # 2-hour intervals
            tickangle=45,  # Rotate for readability
            row=1, col=1
        )
        logger.debug("Plotly figure generated successfully")
        return fig

    def merge_bos_ob_plots(self, ohlc_15m: pd.DataFrame, bos_signals: list, ob_signals: list, main_tf: str = '15m'):
        """
        Merge BOS (1H) and OB (15m) plots into a single comprehensive figure with BOS/CHOCH lines
        interrupted by centered text.

        Parameters:
        - ohlc_15m (pd.DataFrame): OHLCV DataFrame for 15m with pandas.Timestamp index
        - bos_signals (list): List of BOS/CHoCH signals from 1H
        - ob_signals (list): List of Order Block signals from 15m
        - main_tf (str): Timeframe for the plot (default: '15m')

        Returns:
        - go.Figure: Combined Plotly figure with interrupted BOS lines and OB rectangles
        """
        logger.debug("Generating combined BOS and OB Plotly figure")
        ohlc_15m = ohlc_15m.sort_index()

        # Limit to last plot_tail rows
        plot_tail = getattr(self, 'plot_tail', 100)  # Default to 100 if not set
        ohlc_15m = ohlc_15m.tail(plot_tail)

        symbol = ohlc_15m['symbol'].iloc[0] if 'symbol' in ohlc_15m.columns else 'Unknown'
        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} | BOS (1H) & OB (15m) Analysis"])

        # Add OHLC candlesticks
        fig.add_trace(go.Candlestick(
            x=ohlc_15m.index,
            open=ohlc_15m['open'],
            high=ohlc_15m['high'],
            low=ohlc_15m['low'],
            close=ohlc_15m['close'],
            name="OHLC"
        ), row=1, col=1)

        # Filter BOS signals to a wider time window around OB signals
        if ob_signals:
            ob_times = [s['time'] for s in ob_signals]
            min_ob_time = min(ob_times) - pd.Timedelta(hours=24)  # Extend window 24 hours before
            max_ob_time = max(ob_times) + pd.Timedelta(hours=24)  # Extend window 24 hours after
            filtered_bos_signals = [
                s for s in bos_signals
                if (s['time'] >= min_ob_time and s['time'] <= max_ob_time) or
                   (s['fractal_time'] >= min_ob_time and s['fractal_time'] <= max_ob_time)
            ]
            logger.debug(
                f"Filtered {len(bos_signals)} BOS signals to {len(filtered_bos_signals)} within expanded time window: {min_ob_time} to {max_ob_time}")
        else:
            filtered_bos_signals = bos_signals  # If no OB signals, include all BOS signals
            logger.debug("No OB signals, including all BOS signals")

        # Add BOS/CHoCH signals with interrupted lines and centered text
        for signal in filtered_bos_signals:
            if (signal['bias'] == Trend.BULLISH.value and getattr(self, 'show_bull', True)) or \
                    (signal['bias'] == Trend.BEARISH.value and getattr(self, 'show_bear', True)):
                start_time = signal['fractal_time']
                end_time = ohlc_15m.index[-1] + pd.Timedelta(minutes=15)
                # Ensure start_time is within ohlc_15m.index
                if start_time not in ohlc_15m.index:
                    start_time = ohlc_15m.index[ohlc_15m.index.get_indexer([start_time], method='nearest')[0]]
                # Calculate midpoint for text placement
                midpoint_time = start_time + (end_time - start_time) / 2
                # Define gap for line interruption (e.g., ±30 minutes around midpoint)
                gap_size = pd.Timedelta(minutes=30)
                pre_text_time = midpoint_time - gap_size
                post_text_time = midpoint_time + gap_size
                color = 'green' if signal['bias'] == Trend.BULLISH.value else 'orange'

                # Draw first line segment (before text)
                if pre_text_time > start_time:
                    fig.add_trace(go.Scatter(
                        x=[start_time, pre_text_time],
                        y=[signal['level'], signal['level']],
                        mode='lines',
                        line=dict(color=color, dash='solid', width=2),
                        name=signal['type'],
                        showlegend=False
                    ), row=1, col=1)

                # Draw second line segment (after text)
                if end_time > post_text_time:
                    fig.add_trace(go.Scatter(
                        x=[post_text_time, end_time],
                        y=[signal['level'], signal['level']],
                        mode='lines',
                        line=dict(color=color, dash='solid', width=2),
                        name=signal['type'],
                        showlegend=False
                    ), row=1, col=1)

                # Add text at midpoint
                fig.add_trace(go.Scatter(
                    x=[midpoint_time],
                    y=[signal['level']],
                    mode='text',
                    text=[signal['type']],
                    textposition='middle center',
                    textfont=dict(size=12, color=color),
                    showlegend=False
                ), row=1, col=1)

        # Add OB rectangles
        for ob in ob_signals:
            if (ob["type"].upper() == Trend.BULLISH.name and getattr(self, 'show_bull', True)) or \
                    (ob["type"].upper() == Trend.BEARISH.name and getattr(self, 'show_bear', True)):
                if not ob["breaker"] or getattr(self, 'show_fulfilled', True):
                    fillcolor = "rgba(0, 0, 255, 0.5)" if ob[
                                                              "type"].upper() == Trend.BULLISH.name else "rgba(255, 0, 0, 0.5)"
                    line_color = "rgba(0, 0, 255, 0.8)" if ob[
                                                               "type"].upper() == Trend.BULLISH.name else "rgba(255, 0, 0, 0.8)"
                    line_dash = "solid"
                    if ob["breaker"]:
                        fillcolor = "rgba(169, 169, 169, 0.6)"
                        line_color = "rgba(255, 0, 0, 0.8)" if ob[
                                                                   "type"].upper() == Trend.BEARISH.name else "rgba(0, 0, 255, 0.8)"
                        line_dash = "dash"

                    end_time = ob["break_loc"] if ob["breaker"] else ohlc_15m.index[-1]
                    if ob["breaker"] and end_time not in ohlc_15m.index:
                        end_time = ohlc_15m.index[ohlc_15m.index.get_indexer([end_time], method='nearest')[0]]
                        logger.debug(f"Adjusted end_time to nearest OHLC index: {end_time} for OB at {ob['time']}")

                    logger.debug(
                        f"Adding OB rectangle: x0={ob['time']}, x1={end_time}, y0={ob['bottom']:.6f}, y1={ob['top']:.6f}, "
                        f"fillcolor={fillcolor}, line_color={line_color}, line_dash={line_dash}")
                    fig.add_shape(
                        type="rect",
                        x0=ob["time"],
                        x1=end_time,
                        y0=ob["bottom"],
                        y1=ob["top"],
                        fillcolor=fillcolor,
                        line=dict(width=1, color=line_color, dash=line_dash),
                        layer="below",
                        opacity=0.5 if not ob["breaker"] else 0.6,
                        name=ob["type"]
                    )

        # Update layout
        last_candle = ohlc_15m.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')
        candle_time_str = f"Last Candle: {last_candle} (PT)"
        fig.update_layout(
            title=f"{symbol} | BOS (1H) & OB (15m) Analysis | {candle_time_str} | TF: {main_tf}",
            xaxis_title="Time",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            showlegend=True,
            xaxis=dict(range=[ohlc_15m.index[0], ohlc_15m.index[-1] + pd.Timedelta(minutes=15)]),
            margin=dict(l=50, r=50, t=100, b=100)
        )

        fig.update_xaxes(
            title_text="Time",
            type='date',
            tickformat="%H:%M",
            tickmode='linear',
            dtick=900000,  # 15-minute intervals
            tickangle=45,
            row=1, col=1
        )

        logger.debug("Combined BOS and OB figure generated successfully")
        return fig