from datetime import datetime, time, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.detectors.SMCBase import BIAS
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeAction
from daily_trading.settings import logger


class SupplyDemandBacktester:
    """
    MTF Backtester for SupplyDemandZoneTapStrategy.
    Handles syncing 5m trigger data with 1H context data.
    """

    def __init__(self, strategy_instance, timezone: ZoneInfo, plot_tail: int = 300):
        self.strategy = strategy_instance
        self.timezone = timezone
        self.plot_tail = plot_tail
        self.trades: List[dict] = []
        self.pending_trade: Optional[dict] = None

    @staticmethod
    def load_csv(file_path: str, symbol: str, csv_tz: str = "UTC") -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
        time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
        if df[time_col].dtype in [np.int64, np.float64]:
            df['timestamp'] = pd.to_datetime(df[time_col], unit='s', utc=True)
        else:
            df['timestamp'] = pd.to_datetime(df[time_col]).dt.tz_localize(csv_tz).dt.tz_convert("UTC")
        df = df.set_index('timestamp').sort_index()
        df['symbol'] = symbol
        return df[~df.index.duplicated(keep='first')]

    def run(self, ohlc_5m: pd.DataFrame, ohlc_1h: pd.DataFrame, lookback_5m: int = 200, lookback_1h: int = 200):
        """
        Syncs a wide 1H context window with a narrow 5m trigger window.
        Includes Smart Exit detection and Intra-bar Fill tracking.
        """
        logger.info(f"MTF Backtest Started | Symbol: {self.strategy.symbol}")

        # 1. Cleanup & Localization
        ohlc_5m = ohlc_5m.sort_index()
        ohlc_1h = ohlc_1h.sort_index()
        if ohlc_5m.index.tz is None: ohlc_5m.index = ohlc_5m.index.tz_localize("UTC")
        if ohlc_1h.index.tz is None: ohlc_1h.index = ohlc_1h.index.tz_localize("UTC")

        self.strategy.reset_session_state()
        self.trades, self.pending_trade = [], None

        # 2. Session Bound Logic (DST-Aware)
        s_info = SESSION_MAP[self.timezone.key]
        sess_start_wall = time(s_info["start_hour"], s_info["start_minute"])
        sess_end_wall = time(s_info.get("end_hour", 16), s_info.get("end_minute", 0))

        # Calculate pre-market (1hr lookback for zone preparation)
        dummy_dt = datetime.combine(datetime.min, sess_start_wall) - timedelta(hours=1)
        pre_market_wall = dummy_dt.time()

        for i in range(lookback_5m, len(ohlc_5m)):
            ts_utc = ohlc_5m.index[i]
            ts_local = ts_utc.astimezone(self.timezone)
            ts_local_time = ts_local.time()

            # --- DST-AWARE SESSION CHECK ---
            is_interesting = False
            if pre_market_wall < sess_end_wall:  # Standard intra-day
                if pre_market_wall <= ts_local_time <= sess_end_wall:
                    is_interesting = True
            else:  # Spanning midnight
                if ts_local_time >= pre_market_wall or ts_local_time <= sess_end_wall:
                    is_interesting = True

            if not is_interesting: continue

            # 3. Sliding Window Coordinates
            window_5m = ohlc_5m.iloc[i - lookback_5m: i + 1]
            # Context Window: 1H bars that closed BEFORE current 5m bar time
            window_1h = ohlc_1h[ohlc_1h.index < ts_utc].tail(lookback_1h)

            if len(window_1h) < 8: continue

            # 4. Execute Strategy Logic
            result = self.strategy.analyze(window_5m, window_1h)
            curr_bar = ohlc_5m.iloc[i]

            # --- FILL DETECTION (Check before Exit to handle intra-bar open/close) ---
            if self.pending_trade and self.strategy.trade_status.name == "FILLED":
                if self.pending_trade["status"] == "PENDING":
                    self.pending_trade["status"] = "FILLED"
                    self.pending_trade["fill_time"] = ts_utc

            if result.trade_signal:
                sig = result.trade_signal

                if sig.action == TradeAction.ENTER:
                    self.pending_trade = {
                        "timestamp": sig.timestamp,
                        "utc_signal": ts_utc,
                        "direction": sig.direction.name,
                        "entry": sig.price,
                        "sl": sig.sl,
                        "tp": sig.tp,
                        "size": sig.size,
                        "status": "PENDING",
                        "metadata": sig.metadata,
                        "fill_time": None,
                        "comments": ""
                    }


                elif sig.action == TradeAction.EXIT and self.pending_trade:

                    t_rec = self.pending_trade
                    is_long = t_rec["direction"] == "LONG"
                    exit_comment = ""
                    # 1. Standard SL/TP Hit detection
                    sl_hit = curr_bar["low"] <= (t_rec["sl"]) if is_long else curr_bar["high"] >= (t_rec["sl"])
                    tp_hit = curr_bar["high"] >= (t_rec["tp"]) if is_long else curr_bar["low"] <= (t_rec["tp"])

                    if sl_hit:
                        final_price, final_res, exit_comment = t_rec["sl"], "LOSS", "Hard SL Hit"
                    elif tp_hit:
                        final_price, final_res, exit_comment = t_rec["tp"], "WIN", "Hard TP Hit"

                    else:
                        final_price, final_res, exit_comment = curr_bar["close"], (
                            "WIN" if self._is_win(t_rec, curr_bar["close"]) else "LOSS"), "Session Cutoff (Partial)"

                    # 2. CALC ACTUAL DOLLAR PNL (The "Honest" part)
                    # Get point value from the strategy instance
                    pv = self.strategy.POINT_VALUE[self.strategy.symbol]
                    if is_long:
                        dollar_pnl = (final_price - t_rec["entry"]) * t_rec["size"] * pv

                    else:
                        dollar_pnl = (t_rec["entry"] - final_price) * t_rec["size"] * pv

                    t_rec.update({
                        "exit_time": ts_utc,
                        "exit_price": final_price,
                        "result": final_res,
                        "pnl": round(dollar_pnl, 2),  # NEW: Actual dollar profit/loss
                        "status": "CLOSED",
                        "comments": exit_comment
                    })

                    self.trades.append(t_rec)
                    self.pending_trade = None

        return pd.DataFrame(self.trades)

    @staticmethod
    def _is_win(trade: dict, exit_price: float) -> bool:
        """Smart Win check: Is the exit better than the entry?"""
        if trade["direction"] == "LONG":
            return exit_price > (trade["entry"])
        return exit_price < (trade["entry"])

    def plot_trade(self, trade_idx: int, ohlc_5m: pd.DataFrame, ohlc_1h: pd.DataFrame) -> Optional[go.Figure]:
        """
        Generates Plot on-demand.
        Shows 5m candlesticks and overlays the 1H context zones.
        """
        if trade_idx >= len(self.trades): return None
        trade = self.trades[trade_idx]

        # 1. Windows
        start_5m, signal_5m = trade["window_indices_5m"]
        exit_ts = trade.get("exit_time")
        end_5m = ohlc_5m.index.get_loc(exit_ts) + 20 if exit_ts and exit_ts in ohlc_5m.index else signal_5m + 50
        end_5m = min(len(ohlc_5m) - 1, end_5m)

        view_5m = ohlc_5m.iloc[start_5m: end_5m + 1]
        # Sync 1H zones active at time of signal
        view_1h = ohlc_1h[ohlc_1h.index <= trade["utc_signal"]].tail(100)

        # 2. Base Plot (5m Candles)
        fig = go.Figure(data=[go.Candlestick(
            x=view_5m.index, open=view_5m['open'], high=view_5m['high'],
            low=view_5m['low'], close=view_5m['close'], name="5m Trigger TF"
        )])

        # 3. Draw 1H Context Zones
        swing_1h, internal_1h = self.strategy.sd_detector_1h.calculate(view_1h)
        for z in (swing_1h + internal_1h):
            color = "rgba(0, 200, 0, 0.15)" if z['bias'] == BIAS.LONG else "rgba(200, 0, 0, 0.15)"
            fig.add_shape(type="rect", x0=view_5m.index[0], x1=view_5m.index[-1],
                          y0=z['bottom'], y1=z['top'], fillcolor=color, line_width=0, layer="below")

        # 4. R/R Boxes
        entry_ts = trade["utc_signal"]
        chart_end = view_5m.index[-1]
        is_win = trade["result"] == "WIN"

        fig.add_shape(type="rect", x0=entry_ts, x1=chart_end, y0=trade["entry"], y1=trade["tp"],
                      fillcolor="rgba(0, 255, 100, 0.2)" if is_win else "rgba(0, 255, 100, 0.05)",
                      line=dict(width=1, color="green"), layer="below")
        fig.add_shape(type="rect", x0=entry_ts, x1=chart_end, y0=trade["entry"], y1=trade["sl"],
                      fillcolor="rgba(255, 0, 0, 0.05)" if is_win else "rgba(255, 0, 0, 0.2)",
                      line=dict(width=1, color="red"), layer="below")

        # 5. Annotation
        border_color = "green" if is_win else "red"
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99, xanchor="left", yanchor="top",
            text=f"<b>{trade['result']}</b><br>PnL: {trade['size']} units<br>Entry: {trade['entry']}",
            showarrow=False, bgcolor=border_color, font=dict(color="white", size=12),
            bordercolor="white", borderwidth=2, borderpad=6, opacity=0.9
        )

        fig.update_layout(xaxis_rangeslider_visible=False, height=700, template="plotly_white", margin=dict(r=50))
        return fig
