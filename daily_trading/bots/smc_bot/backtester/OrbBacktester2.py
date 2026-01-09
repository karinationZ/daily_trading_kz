from datetime import datetime, time, timedelta
from typing import List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeAction
from daily_trading.settings import logger


class ORBBacktester2:
    """
    Institutional Backtester for ORB SMC Strategy.
    - Optimized for OrderBlocksDetector.
    - Performance Guard: Ensures plotting is disabled during simulation.
    - Corrected visualization for fractal-based OBs.
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

    def run(self, ohlc: pd.DataFrame, lookback: int = 200, bar_range: Optional[Tuple[int, int]] = None):
        if bar_range is not None:
            start_i, end_i = bar_range
            ohlc = ohlc.iloc[start_i:end_i].copy()
            logger.warning(f"⚠️ Backtest restricted to bars [{start_i}:{end_i}]")

        start_dt = ohlc.index.min().strftime('%Y-%m-%d')
        end_dt = ohlc.index.max().strftime('%Y-%m-%d')
        logger.info(f"Backtest Started | Symbol: {self.strategy.symbol} | Range: {start_dt} to {end_dt}")

        if ohlc.index.tz is None:
            ohlc.index = ohlc.index.tz_localize("UTC")

        self.strategy.reset_session_state()
        self.trades = []
        self.pending_trade = None

        s_info = SESSION_MAP[self.timezone.key]
        sess_start_wall = time(s_info["start_hour"], s_info["start_minute"])
        sess_end_wall = time(s_info.get("end_hour", 16), s_info.get("end_minute", 0))

        dummy_dt = datetime.combine(datetime.min, sess_start_wall) - timedelta(hours=1)
        pre_market_wall = dummy_dt.time()

        # PERFORMANCE CHECK:
        # Ensure the strategy's analyze method is not forcing plot=True internally.
        # This is the #1 cause of slow backtests.

        for i in range(100, len(ohlc)):
            ts_utc = ohlc.index[i]
            ts_local = ts_utc.astimezone(self.timezone)
            ts_local_time = ts_local.time()

            # --- SESSION FILTER (Performance Optimization) ---
            is_interesting = False
            if pre_market_wall < sess_end_wall:
                if pre_market_wall <= ts_local_time <= sess_end_wall:
                    is_interesting = True
            else:
                if ts_local_time >= pre_market_wall or ts_local_time <= sess_end_wall:
                    is_interesting = True

            if not is_interesting:
                continue

            window = ohlc.iloc[max(0, i - lookback): i + 1]
            curr_bar = ohlc.iloc[i]

            # Trigger strategy analysis
            # NOTE: If this is slow, the OrderBlocksDetector.detect_order_block
            # nested loops are the cause.
            result = self.strategy.analyze(window)

            if result.trade_signal:
                sig = result.trade_signal

                if sig.action == TradeAction.ENTER:
                    sl_dist = abs(sig.price - sig.sl)
                    sl_pct = (sl_dist / sig.price) * 100 if sig.price != 0 else 0

                    self.pending_trade = {
                        "timestamp": sig.timestamp,
                        "utc_signal": ts_utc,
                        "window_indices": (max(0, i - lookback), i),
                        "direction": sig.direction.name,
                        "entry": sig.price,
                        "sl": sig.sl,
                        "sl_pct": round(sl_pct, 4),
                        "tp": sig.tp,
                        "size": sig.size,
                        "status": "PENDING",
                        "metadata": sig.metadata
                    }

                elif sig.action == TradeAction.CANCEL and self.pending_trade:
                    t_rec = self.pending_trade
                    t_rec.update({
                        "exit_time": ts_utc,
                        "exit_price": curr_bar["close"],
                        "result": "CANCELLED",
                        "status": "CANCELLED"
                    })
                    self.trades.append(t_rec)
                    self.pending_trade = None

                elif sig.action == TradeAction.EXIT and self.pending_trade:
                    t_rec = self.pending_trade
                    is_long = t_rec["direction"] == "LONG"

                    # Precise Exit Check
                    tp_hit = curr_bar["high"] >= (t_rec["tp"] - 1e-6) if is_long else curr_bar["low"] <= (
                                t_rec["tp"] + 1e-6)
                    sl_hit = curr_bar["low"] <= (t_rec["sl"] + 1e-6) if is_long else curr_bar["high"] >= (
                                t_rec["sl"] - 1e-6)

                    if tp_hit:
                        final_price, final_res = t_rec["tp"], "WIN"
                    elif sl_hit:
                        final_price, final_res = t_rec["sl"], "LOSS"
                    else:
                        final_price = curr_bar["close"]
                        final_res = "WIN" if self._is_win(t_rec, final_price) else "LOSS"

                    t_rec.update({
                        "exit_time": ts_utc,
                        "exit_price": final_price,
                        "result": final_res,
                        "status": "CLOSED"
                    })
                    self.trades.append(t_rec)
                    self.pending_trade = None

            if self.pending_trade and self.strategy.trade_status.name == "FILLED":
                if self.pending_trade["status"] == "PENDING":
                    self.pending_trade["status"] = "FILLED"
                    self.pending_trade["fill_time"] = ts_utc

        return pd.DataFrame(self.trades)

    @staticmethod
    def _is_win(trade: dict, exit_price: float) -> bool:
        if trade["direction"] == "LONG":
            return exit_price >= (trade["tp"] - 1e-6)
        else:
            return exit_price <= (trade["tp"] + 1e-6)

    def plot_trade(self, trade_idx: int, ohlc: pd.DataFrame) -> Optional[go.Figure]:
        """
        Generates Plot on-demand using OrderBlocksDetector plotting logic.
        - TradingView style: Extended R/R boxes and side-aligned labels.
        - No vertical entry line.
        """
        if trade_idx >= len(self.trades):
            logger.error("Trade index out of range.")
            return None

        trade = self.trades[trade_idx]
        if "window_indices" not in trade:
            logger.error("Missing window_indices. Run backtest first.")
            return None

        # 1. Calculate Window with Padding
        start_idx, signal_idx = trade["window_indices"]
        exit_ts = trade.get("exit_time")

        # Determine the end of the plot (Exit + 30 bars for visual space)
        if exit_ts and exit_ts in ohlc.index:
            exit_loc = ohlc.index.get_loc(exit_ts)
            end_idx = min(len(ohlc) - 1, exit_loc + 30)
        else:
            end_idx = min(len(ohlc) - 1, signal_idx + 60)

        window = ohlc.iloc[start_idx: end_idx + 1]

        # 2. Get base strategy plot (Candlesticks + OB Rectangles)
        # We call the detector's internal plotting method
        fig = self.strategy.ob_detector._plot_order_blocks(
            window,
            self.strategy.ob_detector.detect_order_block(window)
        )

        # 3. Execution Visuals (TradingView Style)
        entry_ts = trade["utc_signal"]
        final_exit_ts = trade.get("exit_time", window.index[-1])
        # To make it "Wider", we extend the R/R boxes to the end of the visible window index
        chart_end_ts = window.index[-1]

        # Win/Loss Colors
        is_win = trade["result"] == "WIN"
        target_color = "rgba(0, 200, 80, 0.25)" if is_win else "rgba(0, 200, 80, 0.1)"
        stop_color = "rgba(255, 50, 50, 0.1)" if is_win else "rgba(255, 50, 50, 0.25)"
        border_color = "green" if is_win else "red"

        # Draw TP Box (Extended to right side of chart)
        fig.add_shape(type="rect", x0=entry_ts, x1=chart_end_ts,
                      y0=trade["entry"], y1=trade["tp"],
                      fillcolor=target_color, line=dict(width=1, color="green"), layer="below")

        # Draw SL Box (Extended to right side of chart)
        fig.add_shape(type="rect", x0=entry_ts, x1=chart_end_ts,
                      y0=trade["entry"], y1=trade["sl"],
                      fillcolor=stop_color, line=dict(width=1, color="red"), layer="below")

        # 4. Clean Win/Loss Label (Positioned at the side of the exit)
        # We place it at the exit timestamp, but shifted to the right for "side" placement
        # 4. Clean Win/Loss Label — FIXED (Top-left, screen-anchored)
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,  # Left side
            y=0.99,  # Top
            xanchor="left",
            yanchor="top",
            text=f"<b>{trade['result']}</b><br>PnL: {trade['size']} units",
            showarrow=False,
            bgcolor=border_color,
            font=dict(color="white", size=13),
            bordercolor="white",
            borderwidth=2,
            borderpad=6,
            opacity=0.95,
        )

        # Update Layout for better R/R visibility
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=False,
            margin=dict(r=80)  # Add margin on the right for the side label
        )

        return fig