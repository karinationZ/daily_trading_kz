import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from daily_trading.bots.smc_bot.clients.SMCBase import BEARISH, BULLISH, SMCBase
from daily_trading.settings import logger


class FvgDetector(SMCBase):
    def __init__(self, main_tf, pivot_period=None, left_bars=2, right_bars=1, show_bull=True, show_bear=True,
                 volume_thresh=10, fvg_threshold=0.5, mode='Present', back=180,
                 show_labels=False, show_fulfilled=True):
        super().__init__(main_tf=main_tf, left_bars=left_bars, right_bars=right_bars)
        self.pivot_period = pivot_period if pivot_period is not None else self.length
        self.show_bull = show_bull
        self.show_bear = show_bear
        self.fvg_threshold = fvg_threshold
        self.mode = mode
        self.back = back
        self.show_labels = show_labels
        self.show_fulfilled = show_fulfilled
        self.bull_count = 0
        self.bear_count = 0
        self.bull_mitigated = 0
        self.bear_mitigated = 0
        self.fvg_records = []
        if fvg_threshold == 0:
            logger.warning("fvg_threshold=0, all gaps will be detected, which may increase noise")
        logger.info(f"Initialized FvgDetector with lower_tf={main_tf}, pivot_period={self.pivot_period}, "
                    f"left_bars={self.left_bars}, right_bars={self.right_bars}, show_bull={show_bull}, "
                    f"show_bear={show_bear}, volume_thresh={volume_thresh}, "
                    f"fvg_threshold={fvg_threshold}, mode={mode}, back={back}, show_labels={show_labels}, "
                    f"show_fulfilled={show_fulfilled}")

    def _get_threshold(self, ohlc, i):
        atr = (ohlc['high'] - ohlc['low']).rolling(window=144).mean().iloc[i]
        return atr * self.fvg_threshold if not np.isnan(atr) else 0.0

    def detect_fvg(self, ohlc, i, signals):
        if i < 2:
            return signals
        high = ohlc['high'].iloc[i]
        low = ohlc['low'].iloc[i]
        time = ohlc.index[i]
        volume = ohlc['volume'].iloc[i]
        threshold = self._get_threshold(ohlc, i)

        if self.mode == 'Present' and (len(ohlc) - 1 - i) > self.back:
            return signals

        if self.show_bull:
            bullish_gap = low > ohlc['high'].iloc[i - 2] and ohlc['close'].iloc[i - 1] > ohlc['high'].iloc[i - 2]
            if bullish_gap:
                gap_size = low - ohlc['high'].iloc[i - 2]
                if gap_size >= threshold:
                    fvg_item = {
                        "type": "Major Bullish FVG",
                        "bias": BULLISH,
                        "top": low,
                        "bottom": ohlc['high'].iloc[i - 2],
                        "time": time,
                        "index": i,
                        "volume": volume,
                        "mitigated": False,
                        "mitigated_time": None,
                        "mitigated_index": None
                    }
                    signals.append(fvg_item)
                    self.fvg_records.append(fvg_item)
                    self.bull_count += 1
                    logger.info(f"Bullish FVG detected: top={fvg_item['top']:.6f}, bottom={fvg_item['bottom']:.6f}, "
                                f"time={time}, gap_size={gap_size:.6f}, threshold={threshold:.6f}")

        if self.show_bear:
            bearish_gap = high < ohlc['low'].iloc[i - 2] and ohlc['close'].iloc[i - 1] < ohlc['low'].iloc[i - 2]
            if bearish_gap:
                gap_size = ohlc['low'].iloc[i - 2] - high
                if gap_size >= threshold:
                    fvg_item = {
                        "type": "Major Bearish FVG",
                        "bias": BEARISH,
                        "top": ohlc['low'].iloc[i - 2],
                        "bottom": high,
                        "time": time,
                        "index": i,
                        "volume": volume,
                        "mitigated": False,
                        "mitigated_time": None,
                        "mitigated_index": None
                    }
                    signals.append(fvg_item)
                    self.fvg_records.append(fvg_item)
                    self.bear_count += 1
                    logger.info(f"Bearish FVG detected: top={fvg_item['top']:.6f}, bottom={fvg_item['bottom']:.6f}, "
                                f"time={time}, gap_size={gap_size:.6f}, threshold={threshold:.6f}")

        return signals

    def check_mitigation(self, ohlc, fvg_item, start_index):
        for j in range(start_index + 2, len(ohlc)):
            high = ohlc['high'].iloc[j]
            low = ohlc['low'].iloc[j]
            time = ohlc.index[j]
            index = j
            # Ensure candle fully crosses the gap
            if (fvg_item["type"] == "Major Bullish FVG" and high >= fvg_item["top"] and low <= fvg_item["bottom"]) or \
                    (fvg_item["type"] == "Major Bearish FVG" and high >= fvg_item["top"] and low <= fvg_item["bottom"]):
                fvg_item["mitigated"] = True
                fvg_item["mitigated_time"] = time
                fvg_item["mitigated_index"] = index
                if fvg_item["type"] == "Major Bullish FVG":
                    self.bull_mitigated += 1
                else:
                    self.bear_mitigated += 1
                logger.info(
                    f"{fvg_item['type']} mitigated: bottom={fvg_item['bottom']:.6f}, top={fvg_item['top']:.6f}, "
                    f"time={time}, high={high:.6f}, low={low:.6f}")
                return True
        return False

    def calculate_fvg(self, ohlc, plot=True):
        logger.info(f"Starting FVG calculation with pivot_period={self.pivot_period}, "
                    f"left_bars={self.left_bars}, right_bars={self.right_bars}, fvg_threshold={self.fvg_threshold}, "
                    f"mode={self.mode}, back={self.back}")
        if len(ohlc) < 3:
            logger.error(f"OHLC data too short ({len(ohlc)} rows)")
            return {"fig": None, "signals": [], "alert_conditions": {
                "BULLISH_FVG": [], "BEARISH_FVG": [], "BULLISH_FVG_FULFILLED": [], "BEARISH_FVG_FULFILLED": []
            }}
        ohlc = self.validate_data(ohlc)

        self.bull_count = 0
        self.bear_count = 0
        self.bull_mitigated = 0
        self.bear_mitigated = 0
        self.fvg_records = []
        signals = []

        for i in range(self.left_bars, len(ohlc)):
            signals = self.detect_fvg(ohlc, i, signals)
            for fvg_item in self.fvg_records:
                if not fvg_item["mitigated"]:
                    self.check_mitigation(ohlc, fvg_item, fvg_item["index"])

        pivot_high_points, pivot_low_points = self.calculate_pivots(ohlc)

        logger.info(f"Detected {len(signals)} FVGs: {self.bull_count} bullish, {self.bear_count} bearish, "
                    f"{self.bull_mitigated} bullish mitigated, {self.bear_mitigated} bearish mitigated")

        fig = None
        if plot:
            fig = self._plot_fvg(ohlc, signals, pivot_high_points, pivot_low_points)

        alert_conditions = {
            "BULLISH_FVG": [s for s in signals if s["type"] == "Major Bullish FVG" and not s["mitigated"]],
            "BEARISH_FVG": [s for s in signals if s["type"] == "Major Bearish FVG" and not s["mitigated"]],
            "BULLISH_FVG_FULFILLED": [s for s in signals if s["type"] == "Major Bullish FVG" and s["mitigated"]],
            "BEARISH_FVG_FULFILLED": [s for s in signals if s["type"] == "Major Bearish FVG" and s["mitigated"]]
        }

        logger.info(f"Completed FVG calculation: "
                    f"{len(alert_conditions['BULLISH_FVG'])} bullish FVGs, "
                    f"{len(alert_conditions['BEARISH_FVG'])} bearish FVGs, "
                    f"{len(alert_conditions['BULLISH_FVG_FULFILLED'])} bullish fulfilled, "
                    f"{len(alert_conditions['BEARISH_FVG_FULFILLED'])} bearish fulfilled")
        return {"fig": fig, "signals": signals, "alert_conditions": alert_conditions}

    def _plot_fvg(self, ohlc, signals, pivot_high_points, pivot_low_points):
        logger.info("Generating Plotly figure for FVG signals")
        ohlc = ohlc.sort_index()
        ohlc = ohlc.tail(self.plot_tail)
        # Filter pivots to ensure timestamps are within ohlc.index
        pivot_high_points = [p for p in pivot_high_points if p['time'] in ohlc.index]
        pivot_low_points = [p for p in pivot_low_points if p['time'] in ohlc.index]
        symbol = ohlc['symbol'].iloc[0] if 'symbol' in ohlc.columns else 'Unknown'

        fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{symbol} | Price Chart with Liquidity Voids (FVG)"])

        fig.add_trace(go.Candlestick(
            x=ohlc.index,
            open=ohlc['open'],
            high=ohlc['high'],
            low=ohlc['low'],
            close=ohlc['close'],
            name="OHLC"
        ), row=1, col=1)

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
            ), row=1, col=1)

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
            ), row=1, col=1)

        for signal in signals:
            if (signal["bias"] == BULLISH and self.show_bull) or (signal["bias"] == BEARISH and self.show_bear):
                if not signal["mitigated"] or self.show_fulfilled:
                    color = 'rgba(0, 255, 0, 0.5)' if signal["bias"] == BULLISH else 'rgba(255, 0, 0, 0.5)'
                    if signal["mitigated"]:
                        color = 'rgba(169, 169, 169, 0.6)'
                    end_time = signal["mitigated_time"] if signal["mitigated"] else ohlc.index[-1]
                    if signal["mitigated"] and end_time not in ohlc.index:
                        end_time = ohlc.index[ohlc.index.get_loc(end_time, method='nearest')]
                        logger.info(f"Adjusted end_time to nearest OHLC index: {end_time} for FVG at {signal['time']}")
                    logger.info(
                        f"Adding FVG rectangle: x0={signal['time']}, x1={end_time}, y0={signal['bottom']:.6f}, y1={signal['top']:.6f}, color={color}")
                    fig.add_shape(
                        type="rect",
                        x0=signal["time"],
                        x1=end_time,
                        y0=signal["bottom"],
                        y1=signal["top"],
                        fillcolor=color,
                        line=dict(width=1, color=color.replace('0.5', '0.8').replace('0.6', '0.8')),
                        layer="below",
                        opacity=0.5 if not signal["mitigated"] else 0.6,
                        name=signal["type"]
                    )
                    if self.show_labels:
                        text_position = 'bottom center' if signal["bias"] == BULLISH else 'top center'
                        fig.add_trace(go.Scatter(
                            x=[signal["time"]],
                            y=[(signal["top"] + signal["bottom"]) / 2],
                            mode='text',
                            text=["Liquidity Void"],
                            textposition=text_position,
                            textfont=dict(color=color.replace('0.5', '1.0').replace('0.6', '1.0'), size=12),
                            showlegend=False
                        ))

        fig.update_layout(
            title=f"{symbol} | Liquidity Voids (FVG) [SMC]",
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
