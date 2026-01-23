import asyncio
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytz

from daily_trading.bots.smc_bot.bots.TelegramBot import TelegramBot
from daily_trading.bots.smc_bot.detectors.DetectorsBase import (
    BIAS,
    GET_READY_ALERT,
    Trend,
)
from daily_trading.bots.smc_bot.strategy.BosFvgFillStrategy_old import SignalSummary
from daily_trading.bots.smc_bot.strategy.StrategyBase import (
    ExitType,
    SESSION_MAP,
    TradeAction,
    TradeSignal,
    TradeStatus,
)
from daily_trading.brokers.BitUnixClient import BitUnixClient
from daily_trading.brokers.IBKRClient import IBKRClient
from daily_trading.settings import (
    IBKR_PORT,
    logger,
    TELEGRAM_DEBUG_CHANNEL_ID,
    TELEGRAM_MAIN_CHANNEL_CHAT_ID,
    TELEGRAM_MY_DEBUG_BOT_ID,
    USE_AI,
)
from daily_trading.technicals.AITradeAssistants.GeminiTradeAssistant import (
    GEMINI_PRO,
    GeminiTradeAssistant,
    Ohlc,
)
from daily_trading.technicals.DivergentAssetsFinder import BitUnixDivergentAssetsFinder


# ==========================
# TimeZone Container
# ==========================
@dataclass(frozen=True)
class TimeZone:
    UTC = ZoneInfo("UTC")
    NY = ZoneInfo("America/New_York")
    LONDON = ZoneInfo("Europe/London")
    ASIA = ZoneInfo("Asia/Tokyo")


class SMCBotBase:
    """SMC-based trading bot for high-probability continuation trades across multiple symbols.
    Uses SMCAnalyser for analysis and Telegram messaging, targeting scalping opportunities
    in volatile crypto markets with 1H/4H trend alignment, 15m liquidity pools between
    1H HH/HL or LH/LL, and FVG-based entries.
    """

    # [FIX] Static Lock Manager to prevent Multiple Strategies (ORB vs BOS 1H)
    # fighting over the same symbol simultaneously.
    # Format: { 'MNQ': 'ORB_STRATEGY_TAG', 'SIL': '1H_BOS_TAG' }
    ACTIVE_SYMBOL_LOCKS = {}

    def __init__(
        self,
        symbols,
        main_tf: str = "1h",
        symbol_volume_threshold: float = 30000000,
        lookback_days: int = 1,
        telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID,
        market_type: str = "crypto",
        ib_client=None,
    ):
        """
        Initialize SMCTradingBot.

        Args:
            symbols (list): Trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT']).
            main_tf (str): Main timeframe for trend/BOS (e.g., '1h').
            symbol_volume_threshold (float): Minimum volume for symbol selection.
            lookback_days (int): Lookback period in days for OHLC mgc.
        """
        # If a client is passed (Shared Mode), use it.
        # If not (Standalone Mode), create a new one.
        if ib_client:
            self.ib_client = ib_client
            self.is_shared_client = True
        else:
            self.ib_client = IBKRClient(port=IBKR_PORT, client_id=1)
            self.is_shared_client = False
        self.static_symbols = symbols
        self.symbols = symbols
        self.telegram_bot = TelegramBot(chat_id=telegram_chat_id)
        self.telegram_bot_debug = TelegramBot(chat_id=TELEGRAM_DEBUG_CHANNEL_ID)
        self.telegram_my_debug_bot = TelegramBot(chat_id=TELEGRAM_MY_DEBUG_BOT_ID)
        self.main_tf = main_tf
        self.volume_threshold = symbol_volume_threshold
        self.lookback_days = lookback_days
        self.default_symbols = []
        self.market_type = market_type
        self.bitunix_client = BitUnixClient()
        self.ibkr_client = IBKRClient(port=4002, client_id=88)
        self.asset_finder = BitUnixDivergentAssetsFinder()
        self.timezone = pytz.UTC
        self.symbols_update_timestamp = None
        self.candle_duration = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
        }
        self.lookback_candles = self.lookback_days_converter(lookback_days)
        logger.info(f"SMCTradingBot initialized with {len(symbols)} symbols, {main_tf} TF.")
        self.strategy_name = None

    def lookback_days_converter(self, num_days: int) -> dict:
        """Convert lookback days to candles for each timeframe.

        Args:
            num_days (int): Lookback period in days.

        Returns:
            dict: Lookback candles for lower_tf, higher_tf, and 15m.
        """
        timeframe_minutes = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 1440,
        }
        lower_minutes = timeframe_minutes.get(self.main_tf, 60)
        lookback = {
            self.main_tf: int(num_days * 24 * 60 / lower_minutes),
            "15m": int(num_days * 24 * 60 / 15),  # Added for 15m timeframe
        }
        logger.info(f"Lookback: {self.main_tf}={lookback[self.main_tf]} candles, " f"15m={lookback['15m']} candles")
        return lookback

    @staticmethod
    def sync_state(strategy, current_positions: list, open_orders: list):
        """
        Restores self.active_trade and self.trade_status from IBKR data.
        """
        # If we already have state in memory, do nothing
        if strategy.trade_status != TradeStatus.NONE:
            return

        # 1. Filter orders strictly belonging to THIS strategy
        my_orders = [o for o in open_orders if o.order.orderRef == strategy.ext_tag]

        # 2. Check for ACTIVE POSITIONS
        my_pos = next(
            (p for p in current_positions if p.contract.symbol == strategy.symbol and p.position != 0),
            None,
        )

        if my_pos:
            # We found a position on the symbol. Is it ours?
            # We assume it is ours ONLY if we have matching SL/TP orders with our tag.
            if not my_orders:
                # We have a position, but NO orders with our tag.
                # This belongs to another strategy (e.g., ORB vs BOS 1H) or is a manual trade.
                # Do NOT adopt it.
                return
            # We have a live trade. Reconstruct it.
            direction = BIAS.LONG if my_pos.position > 0 else BIAS.SHORT
            size = abs(int(my_pos.position))
            avg_price = my_pos.avgCost

            # Try to find associated Bracket Orders (SL/TP)
            sl_price = 0.0
            tp_price = 0.0

            # Filter orders that belong to THIS strategy tag
            my_active_orders = [o for o in open_orders if o.order.orderRef == strategy.ext_tag]

            for trade in my_active_orders:
                # IBKR 'lmtPrice' is 0 for Stop orders, use 'auxPrice'
                price = trade.order.lmtPrice if trade.order.lmtPrice > 0 else trade.order.auxPrice

                # Deduce if it's SL or TP based on price location vs Entry
                if direction == BIAS.LONG:
                    if price < avg_price:
                        sl_price = price
                    elif price > avg_price:
                        tp_price = price
                else:
                    if price > avg_price:
                        sl_price = price
                    elif price < avg_price:
                        tp_price = price

            # Rehydrate the TradeSignal object
            strategy.active_trade = TradeSignal(
                action=TradeAction.ENTER,
                symbol=strategy.symbol,
                direction=direction,
                price=avg_price,
                sl=sl_price,
                tp=tp_price,
                size=size,
                timestamp=pd.Timestamp.now(tz=strategy.timezone),
                metadata={"zone_id": "RECOVERED"},  # We lose the original Zone ID, but that's acceptable
            )
            strategy.trade_status = TradeStatus.FILLED

            # Re-acquire Lock on recovery
            SMCBotBase.ACTIVE_SYMBOL_LOCKS[strategy.symbol] = strategy.ext_tag

            strategy.log.info(f"♻️ RESTORED STATE: FILLED {direction.name} @ {avg_price} (SL:{sl_price} TP:{tp_price})")
            return

        # 3. Check for PENDING ORDERS (Limit Entry Waiting)
        my_orders = [o for o in open_orders if o.order.orderRef == strategy.ext_tag]

        # Find the Parent Entry Order (Limit order with no parentId usually)
        # Note: In a bracket, the parent has parentId=0 (or empty)
        parent = next(
            (t for t in my_orders if t.order.parentId == 0 and t.order.orderType in ["LMT", "STP LMT"]),
            None,
        )

        if parent:
            direction = BIAS.LONG if parent.order.action == "BUY" else BIAS.SHORT
            entry_price = parent.order.lmtPrice
            size = int(parent.order.totalQuantity)

            # Find Children
            sl_price = 0.0
            tp_price = 0.0
            for t in my_orders:
                if t.order.parentId == parent.order.orderId:
                    price = t.order.lmtPrice if t.order.lmtPrice > 0 else t.order.auxPrice
                    if direction == BIAS.LONG:
                        if price < entry_price:
                            sl_price = price
                        else:
                            tp_price = price
                    else:
                        if price > entry_price:
                            sl_price = price
                        else:
                            tp_price = price

            meta = {"zone_id": "RECOVERED"}

            if hasattr(strategy, "restore_metadata_defaults"):
                meta.update(strategy.restore_metadata_defaults)

            strategy.active_trade = TradeSignal(
                action=TradeAction.ENTER,
                symbol=strategy.symbol,
                direction=direction,
                price=entry_price,
                sl=sl_price,
                tp=tp_price,
                size=size,
                timestamp=pd.Timestamp.now(tz=strategy.timezone),
                metadata=meta,
            )
            strategy.trade_status = TradeStatus.PENDING
            # Re-acquire Lock on recovery
            SMCBotBase.ACTIVE_SYMBOL_LOCKS[strategy.symbol] = strategy.ext_tag
            strategy.log.info(f"♻️ RESTORED STATE: PENDING {direction.name} @ {entry_price}")

    async def run_strategy_cycle(self, strategy, *args, **kwargs):
        try:
            symbol = strategy.symbol
            # Check Locking to prevent collisions
            current_lock = self.ACTIVE_SYMBOL_LOCKS.get(symbol)
            if current_lock and current_lock != strategy.ext_tag:
                # Symbol is busy with another strategy (e.g., ORB has it, BOS 1H can't touch it)
                # Fail silently or log debug to avoid spam
                logger.info(f"Skipping {symbol}, busy with {current_lock}")
                return
            # -----------------------------------------------------------------
            # 1. ANALYSIS & SIGNAL GENERATION
            # -----------------------------------------------------------------
            # The strategy analyzes price/indicators and returns a result.
            analysis = strategy.analyze(*args, **kwargs)
            sig = analysis.trade_signal

            # If no signal, just update the dashboard/log (optional) and return
            if not sig:
                return

            # -----------------------------------------------------------------
            # 2. EXECUTION (The "Doer")
            # -----------------------------------------------------------------
            execution_success = False
            try:
                # This will raise Exception if order placement fails (e.g. Insufficient Funds)
                execution_success = await self.handle_execution(strategy, analysis)
                if not execution_success:
                    return

                # Manage Locks based on action
                if sig.action == TradeAction.ENTER:
                    self.ACTIVE_SYMBOL_LOCKS[symbol] = strategy.ext_tag
                elif sig.action in [TradeAction.EXIT, TradeAction.CANCEL]:
                    self.ACTIVE_SYMBOL_LOCKS.pop(symbol, None)

            except Exception as exe_err:
                strategy.log.error(f"Execution Aborted: {exe_err}")
                # Do NOT proceed to reporting if execution failed
                return

            # -----------------------------------------------------------------
            # 3. POST-EXECUTION WAIT & MONITOR (The "Safety Net")
            # -----------------------------------------------------------------

            # --- A. WAIT FOR BROKER ---
            if sig.action == TradeAction.EXIT:
                exit_type = sig.metadata.get("exit_type", ExitType.SL_TP_HIT)

                if exit_type == ExitType.SL_TP_HIT:
                    # Passive Exit: Wait long for IBKR API to update Position
                    await asyncio.sleep(15.0)
                else:
                    # Force Exit: Wait short for Market Order fill
                    await asyncio.sleep(5.0)

                # --- B. CHECK FLATNESS (Pass 1) ---
                is_flat, pos_size = await self.ib_client.check_flatness(sig.symbol)

                # --- C. RETRY LOGIC (Pass 2) ---
                # If not flat and it was a Passive exit, force refresh and try again
                for _ in range(3):
                    if not is_flat and exit_type == ExitType.SL_TP_HIT:
                        logger.warning(f"⚠️ Exit (SL_TP_HIT) - Flatness Check Failed for {sig.symbol}. Retrying...")
                        await asyncio.sleep(3.0)
                        await self.ib_client.get_active_positions()  # Request update from Server
                        is_flat, pos_size = await self.ib_client.check_flatness(sig.symbol)
                        if is_flat:
                            break

                # --- D. FINAL DECISION ---
                if not is_flat:
                    if abs(pos_size) > 0:
                        warn_msg = f"🚨 <b>CRITICAL FAILURE:</b> Position {pos_size} " f"stuck after EXIT signal. Force Closing."
                        logger.error(warn_msg)
                        await self.telegram_bot.send_message(warn_msg)
                        await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)
                        await self.ib_client.close_positions_smartly(sig.symbol)
                    else:
                        # False alarm
                        await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)

                else:
                    # ✅ SUCCESS: We are flat (either on Pass 1 or Pass 2).
                    # ALWAYS run cleanup here to ensure no orphan orders remain.
                    await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)

            elif sig.action == TradeAction.ENTER:
                # Wait 2s for the Bracket Order to register in the system
                await asyncio.sleep(2.0)

            # -----------------------------------------------------------------
            # 4. REPORTING (Telemetry)
            # -----------------------------------------------------------------

            # ADD HERE TELEGRAM LOG that TRADE has been filled: if strategy.trade_status = TradeStatus.FILLED
            if sig.action not in [
                TradeAction.ENTER,
                TradeAction.EXIT,
                TradeAction.CANCEL,
            ]:
                return

            real_fill_price = 0.0
            debug_note = ""

            # --- A. EXIT REPORTING ---
            if sig.action == TradeAction.EXIT:
                # Try to get the actual price we got filled at
                real_fill_price = await self.ib_client.get_last_fill_price(sig.symbol)

                # Check for Slippage (Difference between Signal Price and Actual Fill)
                # Note: sig.price might be the TP target, but real_fill_price includes slippage.
                sig.metadata.get("entry_price", 0)
                target_ref = sig.price

                # Only calc slippage if we have valid numbers and it wasn't a huge jump (data error)
                if real_fill_price > 0 and target_ref > 0:
                    diff = real_fill_price - target_ref
                    # Filter out noise (0.25 ticks)
                    if abs(diff) > 0.5:
                        debug_note = f"\n💸 <b>Slippage/Diff:</b> {diff:+.2f} pts"

            # --- B. ENTRY REPORTING (Restored) ---
            elif sig.action == TradeAction.ENTER:
                # Verify that the Broker actually created the Bracket Orders correctly
                bracket = await self.ib_client.get_active_bracket_levels(sig.symbol, strategy.ext_tag, sig.direction.name)

                if not bracket["found"]:
                    # Retry once after short wait
                    await asyncio.sleep(3.0)
                    bracket = await self.ib_client.get_active_bracket_levels(sig.symbol, strategy.ext_tag, sig.direction.name)
                    if not bracket["found"]:
                        # This means the order might have failed or hasn't appeared yet
                        debug_note = "\n⚠️ <b>Warning:</b> Bracket orders not seen in API yet."
                else:
                    # Verify SL matches strategy logic
                    if abs(bracket["sl"] - sig.sl) > 0.5:  # Allow small float diff
                        debug_note = f"\n⚠️ <b>SL Divergence:</b> Strat {sig.sl} vs Broker {bracket['sl']}"

            # --- C. BUILD TELEGRAM MESSAGE ---

            # Title Generation
            title = sig.action.name
            if sig.action == TradeAction.EXIT:
                res = getattr(sig, "trade_result", "N/A")
                icon = "🟢" if res == "WIN" else "🔴"
                title = f"{icon} EXIT ({res})"
            elif sig.action == TradeAction.ENTER:
                title = "🚀 ENTER"
            elif sig.action == TradeAction.CANCEL:
                title = "🗑️ CANCEL"

            # Price Display (Use Real Fill for Exits if available)
            display_price = real_fill_price if (sig.action == TradeAction.EXIT and real_fill_price > 0) else sig.price

            lines = [
                f"<b>{title}</b> | {strategy.log_tag}",
                f"Action: {sig.direction.name} {sig.action.name}",
                f"Price: {display_price:,.2f}",
            ]

            # PnL Calculation (Exits Only)
            if sig.action == TradeAction.EXIT and display_price > 0:
                # Retrieve the Entry Price stored in metadata
                entry_price = sig.metadata.get("entry_price", 0)
                # Fallback: If not in metadata, try to find it from the strategy logic (less reliable)
                if entry_price == 0 and strategy.active_trade:
                    entry_price = strategy.active_trade.price

                if entry_price > 0:
                    if sig.direction == BIAS.LONG:
                        points = display_price - entry_price
                    else:
                        points = entry_price - display_price

                    multiplier = getattr(strategy, "point_value", 1.0)
                    cash_pnl = points * sig.size * multiplier

                    pnl_emoji = "🤑" if cash_pnl > 0 else "🔻"
                    lines.append(f"{pnl_emoji} <b>PnL: ${cash_pnl:,.2f}</b> ({points:+.2f} pts)")

            # Entry Details
            if sig.action == TradeAction.ENTER:
                lines.append(f"Bracket: SL {sig.sl:,.2f} | TP {sig.tp:,.2f}")
                lines.append(f"Size: {sig.size}")

            # Append Debug Notes & Timestamp
            if debug_note:
                lines.append(debug_note)

            time_str = sig.timestamp.strftime("%H:%M:%S")
            lines.append(f"<i>{time_str} UTC</i>")

            # Send Message
            # Note: getattr(sig, 'fig', None) checks if a chart image was generated
            await self.telegram_bot.send_message("\n".join(lines), getattr(sig, "fig", None))

            # -----------------------------------------------------------------
            # 5. DAILY SUMMARY (Optional)
            # -----------------------------------------------------------------
            if sig.action == TradeAction.EXIT:
                try:
                    # Wait a moment for PnL to register in the account ledger
                    await asyncio.sleep(4)
                    report = await self.ib_client.get_daily_summary()
                    if report:
                        await self.telegram_bot.send_message(report)
                except Exception:
                    pass

        except Exception:
            logger.error(f"Strategy Cycle Error: {traceback.format_exc()}")

    async def handle_execution(self, strategy, analysis: SignalSummary) -> bool:

        if not analysis or not analysis.trade_signal:
            return False

        sig = analysis.trade_signal
        symbol = strategy.symbol

        # Get the exit type (default to PASSIVE to be safe)
        exit_type = sig.metadata.get("exit_type", ExitType.SL_TP_HIT)

        try:
            # --- ENTER ---
            if sig.action == TradeAction.ENTER:
                strategy.log.info(f"🚀 ENTER {sig.direction.name} @ {sig.price}")
                direction = "LONG" if sig.direction == BIAS.LONG else "SHORT"
                # Place Parent + Bracket (OCA)
                await self.ib_client.place_bracket_order(
                    symbol=symbol,
                    quantity=sig.size,
                    entry=sig.price,
                    sl=sig.sl,
                    tp=sig.tp,
                    direction=direction,
                    order_ref=strategy.ext_tag,
                )
                # Verification Step:
                # Wait briefly to confirm the order wasn't rejected immediately (e.g. Insufficient Funds)
                await asyncio.sleep(2.0)

                # We check active orders for this specific tag
                orders = await self.ib_client.get_open_orders()
                my_active_orders = [o for o in orders if o.order.orderRef == strategy.ext_tag]

                if not my_active_orders:
                    # CRITICAL: Order placed but not found 2 seconds later.
                    # It was likely rejected/cancelled immediately.
                    # IBKRClient usually logs errors, but we must stop the flow here.
                    err_msg = f" Order placed for {symbol} but not found in Active list. Likely Rejected."
                    strategy.log.error(err_msg)
                    await self.telegram_bot.send_message(f"⚠️ <b>ORDER FAILED</b> | {strategy.log_tag}" f"\nReason: {err_msg}")
                    return False

            # --- EXIT ---
            elif sig.action == TradeAction.EXIT:

                if exit_type == ExitType.MARKET_FORCE:
                    # CASE A: TIME CUTOFF
                    # 1. Cancel everything (Limit/SL/TP) so they don't trigger while we market close
                    strategy.log.warning("⏰ TIME CUTOFF - FORCING EXIT")
                    await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)
                    # 2. Dump position at Market
                    await self.ib_client.close_positions_smartly(symbol)

                else:
                    # CASE B: SL OR TP HIT (PASSIVE)
                    # DO NOTHING. Trust the IBKR Bracket to execute.
                    # If we interfere, we might create orphan orders.
                    strategy.log.info(f"👀 PASSIVE EXIT DETECTED ({sig.trade_result}). Monitor Only.")

            # --- CANCEL (Pending Entry) ---
            elif sig.action == TradeAction.CANCEL:
                strategy.log.info("🚫 CANCEL PENDING ENTRY")
                await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)

            return True

        except Exception as e:
            strategy.log.error(f"EXECUTION FAILURE: {e}")
            await self.telegram_bot.send_message(f"🚨 <b>EXECUTION ERROR</b> | {strategy.log_tag}\n{str(e)}")
            return False

    @staticmethod
    async def sleep_until_next_interval(interval_minutes: int = 5):
        """Synced Sleep."""
        now = datetime.now()
        minutes_to_sleep = interval_minutes - (now.minute % interval_minutes)
        seconds_to_sleep = (minutes_to_sleep * 60) - now.second
        if seconds_to_sleep <= 0:
            seconds_to_sleep = interval_minutes * 60
        logger.debug(f"SYNC: Waiting {seconds_to_sleep}s for next candle.")
        await asyncio.sleep(seconds_to_sleep)

    @staticmethod
    def is_strategy_in_window(strategy) -> bool:
        """
        Checks if the strategy is within its operating window (Session +/- Buffers).
        Prevents running NY logic during Asian session, etc.
        """
        # 1. Get Setup Info
        tz_key = str(strategy.timezone)
        s_info = SESSION_MAP.get(tz_key)
        if not s_info:
            raise ValueError(f"Unknown timezone/session: {tz_key}")

        now = datetime.now(strategy.timezone)

        # 2. Define Buffers (in minutes)
        pre_session_buffer = 60  # Start 1 hour before session open
        post_session_buffer = 30  # End 30 mins after session close

        # 3. Construct Session Start/End times for Today
        start_time = now.replace(
            hour=s_info.get("start_hour", 9),
            minute=s_info.get("start_minute", 0),
            second=0,
            microsecond=0,
        )
        end_time = now.replace(
            hour=s_info.get("end_hour", 16),
            minute=s_info.get("end_minute", 0),
            second=0,
            microsecond=0,
        )

        # 4. Apply Buffers
        active_start = start_time - timedelta(minutes=pre_session_buffer)
        active_end = end_time + timedelta(minutes=post_session_buffer)

        # 5. Check Window (simple comparison, no midnight wrapping needed)
        return active_start <= now <= active_end

    @staticmethod
    async def sleep_until_next_hour():
        # Get current time in UTC
        now = datetime.now(pytz.UTC)

        # Next full hour + 3 minute
        next_candle = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, minutes=3)

        # Time to sleep (in seconds)
        sleep_duration = (next_candle - now).total_seconds()

        logger.info(f"Sleeping for {int(sleep_duration)} seconds ({sleep_duration / 60:.2f} minutes) until {next_candle}")
        await asyncio.sleep(sleep_duration)

    async def send_final_message(self, analysis: SignalSummary, current_candle_time, symbol, title):
        events = [e for e in analysis.recent_smc_events if e is not None]
        events_str = "\n• ".join([event[0] if isinstance(event, list) else event for event in events])

        action = f"Go {analysis.direction}." if analysis.direction != Trend.NEUTRAL.name else "Сидим в засаде"
        msg = f"{title} - {symbol} {current_candle_time}\n" f"Action: {action}\n" f"Trend: {analysis.trend}\n" f"SMC Events:\n• {events_str}"
        logger.info(msg)
        is_success = await self.telegram_bot.send_message(msg, analysis.fig)

        # # Send to my debug bot as well if symbol is in default_symbols like BTCUSDT, ETHUSDT, XAUTUSDT
        # if symbol in self.default_symbols and action not in [Trend.NEUTRAL.name, "Сидим в засаде"]:
        #     await self.telegram_my_debug_bot.send_message(msg, analysis.fig)

        return is_success

    async def fetch_bitunix_ohlc_data(self, symbol: str, timeframe: str, num_candles: int = None) -> pd.DataFrame:
        """Fetch OHLC mgc with caching to reduce API calls.

        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            timeframe (str): Timeframe (e.g., '1h', '15m').
            num_candles (int, optional): Number of candles to fetch.

        Returns:
            pd.DataFrame: OHLC mgc with time index and float columns.
        """
        logger.info(f"Fetching OHLC mgc for {symbol.upper()} ({timeframe})")
        try:
            required_candles = self.lookback_candles.get(timeframe, 24) + 2
            num_candles = required_candles if num_candles is None else num_candles
            fetch_candles = int(num_candles)
            minutes_per_candle = self.candle_duration.get(timeframe, 60)
            current_time = int(datetime.now().timestamp() * 1000)
            start_time = current_time - (fetch_candles * minutes_per_candle * 60 * 1000)
            end_time = current_time

            max_limit = 200
            all_klines = []
            seen_timestamps = set()
            current_end = end_time
            api_calls = 0

            while len(set(timestamps := [int(k["time"]) for k in all_klines])) < fetch_candles and api_calls < 10:
                fetch_limit = min(max_limit, fetch_candles - len(set(timestamps)))
                kline_data = self.bitunix_client.get_klines(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=start_time,
                    end_time=current_end,
                    limit=fetch_limit,
                    kline_type="LAST_PRICE",
                )

                api_calls += 1
                if not kline_data:
                    current_end = start_time
                    start_time -= fetch_limit * minutes_per_candle * 60 * 1000
                    continue

                unique_klines = [k for k in kline_data if int(k["time"]) not in seen_timestamps]
                for k in unique_klines:
                    seen_timestamps.add(int(k["time"]))
                all_klines.extend(unique_klines)

                if unique_klines:
                    unique_klines = sorted(unique_klines, key=lambda x: int(x["time"]))
                    current_end = int(unique_klines[0]["time"]) - 1000
                    start_time = current_end - (fetch_limit * minutes_per_candle * 60 * 1000)
                else:
                    current_end -= fetch_limit * minutes_per_candle * 60 * 1000
                    start_time = current_end - (fetch_limit * minutes_per_candle * 60 * 1000)

            if not all_klines:
                logger.error(f"No kline mgc for {symbol} ({timeframe})")
                return pd.DataFrame()

            df = pd.DataFrame(all_klines, columns=["time", "open", "high", "low", "close", "quoteVol"])
            df = df.rename(columns={"quoteVol": "volume"})
            df["time"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="ms", utc=True)
            df["symbol"] = symbol
            df.set_index("time", inplace=True)
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].astype(float)
            df = df[["symbol", "open", "high", "low", "close", "volume"]]
            logger.info(f"Cached OHLC mgc for {symbol} ({timeframe}): {len(df)} rows")
            return df.sort_values(by="time")

        except Exception as e:
            logger.error(f"Error fetching OHLC for {symbol} ({timeframe}): {e}")
            raise e

    async def fetch_ibkr_ohlc_data(self, symbol: str, timeframe: str, num_candles: int = None) -> pd.DataFrame:
        await self.ibkr_client.connect()
        kline_data = await self.ibkr_client.get_klines(
            symbol=symbol,
            interval=timeframe,
        )
        kline_data = kline_data[num_candles:]
        if not kline_data.empty:
            # 1. Drop Duplicates based on Index
            kline_data = kline_data[~kline_data.index.duplicated(keep="last")]

            # 2. Sort by Index (Chronological)
            kline_data.sort_index(inplace=True)

            # 3. Add 'time' column from Index
            kline_data["time"] = kline_data.index
        else:
            logger.error("IBKR returned empty OHLC data!")
        return kline_data

    async def needs_symbol_refresh(self) -> bool:
        """Check if symbols list needs to be refreshed (empty, never set, or >2h old)."""
        if not self.symbols or self.symbols_update_timestamp is None:
            return True

        dt_now = datetime.now(self.timezone)
        time_diff = (dt_now - self.symbols_update_timestamp).total_seconds()
        return time_diff >= 14400  # 4 hours

    async def update_symbols(self):
        """Update high-volume symbols every 2 hours.

        Uses BitUnixDivergentAssetsFinder to select non-correlated assets with sufficient volume.
        """
        logger.info("Updating high-volume symbols")
        try:
            results = await self.asset_finder.find_divergent_assets(volume_threshold=7000000, lookback_hours=3)
            self.symbols = [asset["symbol"] for asset in results["non_correlated"]]
            self.symbols.extend(self.default_symbols)

            # if not self.symbols:
            #     logger.info(f"There are no divergent assets, using default symbols - {self.default_symbols}")
            #     self.symbols = self.default_symbols

            logger.info(f"Updated symbols: {self.symbols}")
            self.symbols_update_timestamp = datetime.now(self.timezone)

        except Exception as e:
            logger.error(f"Failed to update symbols: {e}")
            raise RuntimeError(f"Failed to update symbols: {e}")

    async def generate_signal(
        self,
        symbol: str,
        analysis: SignalSummary,
        ohlc_1d: pd.DataFrame,
        ohlc_4h: pd.DataFrame,
        ohlc_1h: pd.DataFrame,
        ohlc_15m: pd.DataFrame,
        send_non_signal=True,
    ):
        """Generate SMC signals."""
        try:
            last_candle = ohlc_15m.index.max().tz_convert("US/Pacific").strftime("%Y-%m-%d %H:%M")

            if analysis.direction in [BIAS.LONG.value, BIAS.SHORT.value]:
                # AI Request
                ohlc = Ohlc(ohlc_1d, ohlc_4h, ohlc_1h, ohlc_15m)
                ai_agent = GeminiTradeAssistant(symbol=symbol, model=GEMINI_PRO, ohlc=ohlc, debug=False)
                within_ai_hours = ai_agent.within_ai_request_hours(start_hour=8, end_hour=21)
                formatted_ai_message = None
                if within_ai_hours and USE_AI.lower() == "true":
                    formatted_ai_message, raw_message = await ai_agent.analyze_asset_trade_data()
                    if raw_message["entry"]:
                        fig = ai_agent.update_plot_with_entry_data(analysis.fig, raw_message)
                        analysis.fig = fig
                is_success = await self.send_final_message(analysis, last_candle, symbol, title=analysis.title)

                if formatted_ai_message and is_success:
                    await self.telegram_bot.send_message(formatted_ai_message)

            elif analysis.direction == GET_READY_ALERT:
                await self.send_final_message(analysis, last_candle, symbol, title=analysis.title)

            else:
                events = [e for e in analysis.recent_smc_events if e is not None]
                events_str = "\n • ".join([event[0] if isinstance(event, list) else event for event in events])
                msg = f"{symbol}\n{self.strategy_name}\n" f"Action: Сидим в засаде\n" f"Trend: {analysis.trend}\n" f"SMC Events:\n • {events_str}\n"
                logger.info(msg)
                if send_non_signal:
                    await self.telegram_bot_debug.send_message(msg, analysis.fig)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
