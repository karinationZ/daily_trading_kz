import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.clients.SMCBase import SMCBase
from daily_trading.settings import logger


class OrderBlocksDetector(SMCBase):
    def __init__(self, main_tf, left_bars=5, right_bars=5, tap_back_ratio=0.1, plot_tail=200):
        super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
        self.tap_back_ratio = tap_back_ratio
        self.plot_tail = plot_tail
        logger.info(
            f"Initialized OrderBlocksDetector with main_tf={main_tf}, tap_back_ratio={tap_back_ratio}, plot_tail={plot_tail}")

    def detect_order_block(self, ohlc: pd.DataFrame) -> list:
        """
        Detects Order Blocks (OBs) on 1H timeframe as the last opposing candle before a break of structure.
        OBs are mitigated if price exits (closes beyond block) and taps back into 10% of the block's range.

        Parameters:
        - ohlc: DataFrame with 'open', 'high', 'low', 'close' columns

        Returns:
        - List of dicts: {"type": str, "top": float, "bottom": float, "time": Timestamp, "index": int, "breaker": bool, "break_loc": Timestamp/None}
        """
        logger.info(f"Detecting Order Blocks (OB) on OHLC data with {len(ohlc)} rows")
        ohlc = self.validate_data(ohlc)
        pivot_highs, pivot_lows = self.calculate_pivots(ohlc)
        bullish_ob = []
        bearish_ob = []

        crossed_highs = {p["index"]: False for p in pivot_highs}
        crossed_lows = {p["index"]: False for p in pivot_lows}

        for i in range(1, len(ohlc)):
            current_price = ohlc['close'].iloc[i]
            current_time = ohlc.index[i]

            # Detect bullish OB: last bearish candle before breaking swing high
            for pivot in pivot_highs:
                if pivot["index"] < i and not crossed_highs[pivot["index"]]:
                    if current_price > pivot["level"]:
                        crossed_highs[pivot["index"]] = True
                        # Find last bearish candle (close < open) before break
                        for j in range(i - 1, pivot["index"] - 1, -1):
                            if ohlc['close'].iloc[j] < ohlc['open'].iloc[j]:
                                ob_item = {
                                    "type": "bullish",
                                    "top": ohlc['high'].iloc[j],
                                    "bottom": ohlc['low'].iloc[j],
                                    "time": ohlc.index[j],
                                    "index": pivot["index"],
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
                        for j in range(i - 1, pivot["index"] - 1, -1):
                            if ohlc['close'].iloc[j] > ohlc['open'].iloc[j]:
                                ob_item = {
                                    "type": "bearish",
                                    "top": ohlc['high'].iloc[j],
                                    "bottom": ohlc['low'].iloc[j],
                                    "time": ohlc.index[j],
                                    "index": pivot["index"],
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
        logger.info(f"Detected {len(order_blocks)} OBs: {len(bullish_ob)} bullish, {len(bearish_ob)} bearish")
        return order_blocks

    def calculate_order_blocks(self, ohlc, plot=True):
        order_blocks = self.detect_order_block(ohlc)
        fig = None
        if plot:
            fig = self._plot_order_blocks(ohlc, order_blocks)

        logger.info(f"OB: {order_blocks}")
        return {"fig": fig, "order_blocks": order_blocks}

    def _plot_order_blocks(self, ohlc, order_blocks):
        logger.info("Generating Plotly figure for Order Blocks")

        ohlc = self.validate_data(ohlc)
        # Limit to last 200 rows
        ohlc = ohlc.tail(self.plot_tail)

        # Filter order blocks to ensure timestamps are within ohlc.index
        order_blocks = [ob for ob in order_blocks if ob['time'] in ohlc.index]

        fig = make_subplots(rows=1, cols=1, subplot_titles=["Price Chart with Order Blocks"])

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

        fig.update_layout(
            title="Order Blocks [SMC]",
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
