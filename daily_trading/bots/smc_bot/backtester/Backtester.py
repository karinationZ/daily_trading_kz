from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# ==========================================================
# REFACTORED BACKTESTER (NO ENUMS)
# ==========================================================

class Backtester:
    """
    Data-First Backtester.
    Phase 1: Generate a log of all Strategy intent (Signals).
    Phase 2: Reconstruct the ledger by matching signals to price action.

    Ambiguity Logic: If a candle hits both SL and TP, it is recorded as a LOSS (SL).
    """

    def __init__(self, strategy, timezone: ZoneInfo):
        self.strategy = strategy
        self.timezone = timezone

    # ======================================================
    # DATA LOADER
    # ======================================================
    @staticmethod
    def load_csv(file_path: str, symbol: str, csv_tz: str = "UTC") -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = [c.lower() for c in df.columns]
        time_col = "timestamp" if "timestamp" in df.columns else "time"

        if df[time_col].dtype in (np.int64, np.float64):
            ts = pd.to_datetime(df[time_col], unit="s", utc=True)
        else:
            ts = pd.to_datetime(df[time_col])
            if ts.dt.tz is None: ts = ts.dt.tz_localize(csv_tz)
            ts = ts.dt.tz_convert("UTC")

        df["timestamp"] = ts
        df = df.set_index("timestamp").sort_index()
        df["symbol"] = symbol
        return df[~df.index.duplicated(keep="first")]

    # ======================================================
    # PHASE 1: GENERATE SIGNALS
    # ======================================================
    def generate_signal_log(self, ohlc_5m: pd.DataFrame, ohlc_1h: Optional[pd.DataFrame], lookback_5m: int,
                            lookback_1h: int) -> List[Dict]:
        """
        Runs the strategy logic over the data to produce a raw list of intended actions.
        Does NOT calculate PnL. Only records 'I want to Buy here' or 'I want to Cancel here'.
        """
        self.strategy.reset_session_state()
        signal_log = []

        ts_1h_np = ohlc_1h.index.to_numpy() if ohlc_1h is not None else None

        # Simple start index
        start_idx = max(lookback_5m, 1)

        for i in range(start_idx, len(ohlc_5m)):
            # 1. Data Slicing
            ts = ohlc_5m.index[i]
            window_5m = ohlc_5m.iloc[i - lookback_5m: i]

            # 2. MTF Slicing
            window_1h = None
            if ohlc_1h is not None:
                idx_1h = np.searchsorted(ts_1h_np, ts)
                start_1h = max(0, idx_1h - lookback_1h)
                window_1h = ohlc_1h.iloc[start_1h: idx_1h]
                if len(window_1h) < 10: continue

            # 3. Strategy Analysis
            result = self.strategy.analyze(window_5m, window_1h) if ohlc_1h is not None else self.strategy.analyze(
                window_5m)

            # 4. Record Signal
            if result.trade_signal:
                sig = result.trade_signal
                signal_log.append({
                    "timestamp": ts,
                    "bar_index": i,
                    "action": sig.action.name,  # ENTER, EXIT, CANCEL
                    "direction": sig.direction.name,
                    "price": sig.price,
                    "sl": sig.sl,
                    "tp": sig.tp,
                    "size": sig.size,
                    "metadata": sig.metadata
                })

        return signal_log

    def process_ledger(self, ohlc_5m: pd.DataFrame, signal_log: List[Dict]) -> pd.DataFrame:
        """
        Simulates the Exchange.
        Iterates through the Signal Log and verifies against OHLC if they actually filled.
        """
        trades = []
        active_order = None  # Pending (Dict)
        active_position = None  # Filled (Dict)

        # Convert OHLC to numpy for speed
        opens = ohlc_5m['open'].to_numpy()
        highs = ohlc_5m['high'].to_numpy()
        lows = ohlc_5m['low'].to_numpy()
        closes = ohlc_5m['close'].to_numpy()
        times = ohlc_5m.index

        # Map signals to bar indices for quick lookup
        sig_map = {}
        for s in signal_log:
            idx = s['bar_index']
            if idx not in sig_map: sig_map[idx] = []
            sig_map[idx].append(s)

        for i in range(len(ohlc_5m)):
            curr_time = times[i]
            op, hi, lo, cl = opens[i], highs[i], lows[i], closes[i]

            # --- A. Check Active Position (Exits) ---
            if active_position:
                pos = active_position
                is_long = pos['direction'] == "LONG"

                # 1. Strategy Force Exits
                if i in sig_map:
                    for sig in sig_map[i]:
                        if sig['action'] == "EXIT":
                            self._record_trade(trades, pos, cl, curr_time, "Strategy Exit", status="STRATEGY_EXIT")
                            active_position = None
                            break

                # 2. Price Limits (SL/TP)
                if active_position:
                    exit_type, price = self._check_sl_tp(op, hi, lo, is_long, pos['sl'], pos['tp'])
                    if exit_type:
                        # STATUS FIX: Explicitly pass status
                        self._record_trade(trades, pos, price, curr_time, f"{exit_type} Hit", status=f"{exit_type} Hit")
                        active_position = None

            # --- B. Check Pending Order (Fills or Cancels) ---
            elif active_order:
                ord_ = active_order
                is_long = ord_['direction'] == "LONG"

                # 1. Strategy Cancels
                cancelled = False
                if i in sig_map:
                    for sig in sig_map[i]:
                        if sig['action'] == "CANCEL":
                            self._record_trade(trades, ord_, 0, curr_time, "Strategy Cancel", status="CANCELLED")
                            active_order = None
                            cancelled = True
                            break

                # 2. Check Fill
                if not cancelled and active_order:
                    fill_price = ord_['price']
                    # Limit logic: Buy if Low <= Price, Sell if High >= Price
                    filled = (lo <= fill_price) if is_long else (hi >= fill_price)

                    if filled:
                        active_position = ord_
                        active_position['entry_time'] = curr_time
                        active_position['status'] = "FILLED"
                        active_order = None

                        # --- IMMEDIATE CANDLE STOPOUT CHECK ---
                        # This block handles the case where fill and SL happen in the same candle
                        exit_type, price = self._check_sl_tp(op, hi, lo, is_long, active_position['sl'],
                                                             active_position['tp'])

                        if exit_type:
                            self._record_trade(trades, active_position, price, curr_time, f"{exit_type} Hit",
                                               status=f"{exit_type} Hit")
                            active_position = None

            # --- C. New Signals (Overwrite pending) ---
            if i in sig_map:
                for sig in sig_map[i]:
                    if sig['action'] == "ENTER":
                        active_order = sig

        return pd.DataFrame(trades)

    def _record_trade(self, trades_list, trade_dict, exit_price, exit_time, comment, status="CLOSED"):
        pv = 1.0
        if self.strategy is not None:
            pv = getattr(self.strategy, "point_value", 1.0)

        entry_price = trade_dict['price']
        size = trade_dict.get('size', 1.0)
        direction = trade_dict['direction']

        if status == "CANCELLED":
            pnl = 0.0
        elif direction == "LONG":
            pnl = (exit_price - entry_price) * size * pv
        else:
            pnl = (entry_price - exit_price) * size * pv

        # --- FIX START: Determine Result Label ---
        if pnl > 0:
            result = "WIN"
        elif pnl < 0:
            result = "LOSS"
        else:
            result = "BE"  # Break Even
        # --- FIX END ---

        trade = {
            'entry_time': trade_dict.get('entry_time', trade_dict.get('created_time')),
            'exit_time': exit_time,
            'symbol': getattr(self.strategy, "symbol", "TEST"),
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'result': result,  # <--- Added Field
            'status': status,
            'comment': comment
        }
        trades_list.append(trade)

    def _check_sl_tp(self, open_: float, high: float, low: float, is_long: bool, sl: float, tp: float):
        """
        Checks if SL or TP were hit in the candle.
        Returns (exit_type, price) or (None, None) if nothing hit.
        """
        if sl is None or tp is None:
            return None, None

        if is_long:
            # Check Stop Loss (Low triggers SL)
            if low <= sl:
                return "SL", sl
            # Check Take Profit (High triggers TP)
            if high >= tp:
                return "TP", tp
        else:  # Short
            # Check Stop Loss (High triggers SL)
            if high >= sl:
                return "SL", sl
            # Check Take Profit (Low triggers TP)
            if low <= tp:
                return "TP", tp

        # FIX: Explicit return of None tuple to prevent unpacking errors
        return None, None

    # ======================================================
    # MAIN ENTRY POINT
    # ======================================================
    def run(self, ohlc_5m, ohlc_1h=None, lookback_5m=200, lookback_1h=200):
        # 1. Get Strategy Intent
        signals = self.generate_signal_log(ohlc_5m, ohlc_1h, lookback_5m, lookback_1h)

        # 2. Verify against Market Data
        ledger = self.process_ledger(ohlc_5m, signals)

        return ledger
