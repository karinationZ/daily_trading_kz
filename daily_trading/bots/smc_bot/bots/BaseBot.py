import asyncio
import json
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
from daily_trading.bots.smc_bot.strategy.old.BosFvgFillStrategy_old import SignalSummary
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


# Add this near your other dataclasses
@dataclass
class LockInfo:
    strategy_tag: str
    strategy_instance: object
    status: TradeStatus  # PENDING or FILLED
    priority: int  # 1 = Low (BOS), 2 = High (ORB)
    timestamp: datetime


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
        self.telegram_main_channel = TelegramBot(chat_id=telegram_chat_id)
        self.telegram_debug_channel = TelegramBot(chat_id=TELEGRAM_DEBUG_CHANNEL_ID)
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
        self.semaphore = asyncio.Semaphore(3)

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

        # In SMCBotBase class

    async def acquire_lock(self, strategy) -> bool:
        """
        Attempts to acquire a lock on the symbol.
        1. If Free -> Locks it immediately (Reservation).
        2. If PENDING + Lower Priority -> Overrides, Cancels, and Relocks immediately.
        """
        symbol = strategy.symbol
        current_lock: LockInfo = self.ACTIVE_SYMBOL_LOCKS.get(symbol)
        priority = getattr(strategy, 'priority', 1)

        # 1. Symbol is free
        if not current_lock:
            # CLAIM IT IMMEDIATELY to prevent race conditions
            self.ACTIVE_SYMBOL_LOCKS[symbol] = LockInfo(
                strategy_tag=strategy.ext_tag,
                strategy_instance=strategy,
                status=TradeStatus.NONE,  # Temporary status, will be updated end of cycle
                priority=priority,
                timestamp=datetime.now(),
            )
            return True

        # 2. We already own the lock (re-entrant check)
        if current_lock.strategy_tag == strategy.ext_tag:
            return True

        # 3. Symbol is occupied. Can we steal it?
        if current_lock.status == TradeStatus.PENDING and priority > current_lock.priority:
            msg = f"⚡ OVERRIDE: {strategy.ext_tag} (P{priority}) kicking out {current_lock.strategy_tag}" f" (P{current_lock.priority}) on {symbol}"
            strategy.log.info(msg)
            strategy.telegram_session_bot.send_message(msg)

            if current_lock.strategy_instance:
                # Reset the old strategy's trade state
                current_lock.strategy_instance.trade_status = TradeStatus.NONE

            # --- CRITICAL: CANCEL THE EXISTING ORDERS ---
            try:
                # Cancel orders of the OLD strategy
                strategy.log.info(f" cancelling orders for {current_lock.strategy_tag} on {symbol}")
                await self.ib_client.cancel_all_orders(order_ref=current_lock.strategy_tag)

                # CLAIM IT IMMEDIATELY after flattening
                self.ACTIVE_SYMBOL_LOCKS[symbol] = LockInfo(
                    strategy_tag=strategy.ext_tag,
                    strategy_instance=strategy,
                    status=TradeStatus.NONE,
                    priority=strategy.priority,
                    timestamp=datetime.now()
                )
                return True

            except Exception as e:
                logger.error(f"Error during lock override: {e}")
                return False

        # 4. Hard Block
        # logger.info(f"🔒 {symbol} locked by {current_lock.strategy_tag}. Skipping.")
        return False

    async def sync_state(self, strategy, current_positions: list, open_orders: list):
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
            # CALCULATE DIRECTION
            size = abs(int(my_pos.position))
            direction = BIAS.LONG if my_pos.position > 0 else BIAS.SHORT
            direction_str = "LONG" if direction == BIAS.LONG else "SHORT"
            avg_price = my_pos.avgCost

            # FETCH BRACKETS (The new robust method)
            sl, tp = self.ib_client.get_active_sl_tp(symbol=strategy.symbol, order_ref=strategy.ext_tag,
                                                     direction=direction_str)

            # LOG WARNINGS for missing SL/TP
            # If IBKR returns 0.0 for TP (e.g., user cancelled it manually),
            # set it to Infinity. This prevents the bot from thinking "Price > 0, Exit Now!"
            if tp <= 0:
                msg = f"⚠️ TP not found for {strategy.ext_tag}. Panic Close will be triggered."
                strategy.log.warning(msg)
                await strategy.telegram_session_bot.send_message(msg)

            # If SL is missing, just log it. Bot can still run without SL (risky, but valid software state).
            if sl <= 0:
                msg = f"⚠️ SL not found for {strategy.ext_tag}."
                strategy.log.warning(msg)
                await strategy.telegram_session_bot.send_message(msg)

            # Rehydrate the TradeSignal object
            strategy.active_trade = TradeSignal(
                action=TradeAction.ENTER,
                symbol=strategy.symbol,
                direction=direction,
                price=avg_price,
                sl=sl,
                tp=tp,
                size=size,
                timestamp=pd.Timestamp.now(tz=strategy.timezone),
                metadata={"zone_id": "RECOVERED"},  # We lose the original Zone ID, but that's acceptable
            )
            strategy.trade_status = TradeStatus.FILLED

            # Re-acquire Lock on recovery
            self.ACTIVE_SYMBOL_LOCKS[strategy.symbol] = LockInfo(
                strategy_tag=strategy.ext_tag,
                status=strategy.trade_status,
                strategy_instance=strategy,
                priority=getattr(strategy, "priority", 1),
                timestamp=datetime.now()
            )

            strategy.log.info(f"♻️ RESTORED STATE: FILLED {direction.name} @ {avg_price} (SL:{sl} TP:{tp})")
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
            self.ACTIVE_SYMBOL_LOCKS[strategy.symbol] = LockInfo(
                strategy_tag=strategy.ext_tag,
                status=strategy.trade_status,
                strategy_instance=strategy,
                priority=getattr(strategy, 'priority', 1),
                timestamp=datetime.now()
            )
            strategy.log.info(f"♻️ RESTORED STATE: PENDING {direction.name} @ {entry_price}")

    async def cleanup_orphaned_orders(self, strategy):
        """
        Checks for orders that exist when the strategy believes it is FLAT.
        """
        # 1. Only run this check if we think we are not in a trade
        if strategy.trade_status != TradeStatus.NONE:
            return

        # 2. Get all open orders from IBKR
        open_orders = self.ib_client.get_open_orders()

        # 3. Filter for orders belonging to THIS strategy
        # We assume 'orderRef' matches strategy.ext_tag
        my_orders = [o for o in open_orders if o.order.orderRef == strategy.ext_tag]

        if my_orders:
            logger.warning(
                f"🧹 ORPHAN CHECK: Found {len(my_orders)} hanging orders for {strategy.symbol} while Status=NONE. Purging.")

            # 4. Log what we are killing for debugging
            for o in my_orders:
                logger.info(
                    f"   -> Killing Orphan: {o.order.action} {o.order.totalQuantity} @ {o.order.lmtPrice or o.order.auxPrice}")

            # 5. Nuke them
            await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)
            await strategy.telegram_session_bot.send_message(
                f"🧹 <b>ORPHAN CLEANUP</b> | Removed {len(my_orders)} hanging orders for {strategy.symbol}."
            )

    async def _send_trade_telemetry(self, strategy, sig):
        """Constructs and sends the Telegram report."""
        if sig.action not in [TradeAction.ENTER, TradeAction.EXIT, TradeAction.CANCEL]:
            return

        real_fill_price = 0.0
        debug_note = ""

        # --- GATHER DATA ---
        if sig.action == TradeAction.EXIT:
            real_fill_price = await self.ib_client.get_last_fill_price(sig.symbol)
            # Calc Slippage
            target_ref = sig.price
            if real_fill_price > 0 and target_ref > 0:
                diff = real_fill_price - target_ref
                if abs(diff) > 0.5:
                    debug_note = f"\n💸 <b>Slippage/Diff:</b> {diff:+.2f} pts"

        elif sig.action == TradeAction.ENTER:
            # Verify Brackets
            bracket = await self.ib_client.get_active_bracket_levels(sig.symbol, strategy.ext_tag, sig.direction.name)
            if not bracket["found"]:
                await asyncio.sleep(3.0)  # One retry
                bracket = await self.ib_client.get_active_bracket_levels(sig.symbol, strategy.ext_tag,
                                                                         sig.direction.name)
                if not bracket["found"]:
                    debug_note = "\n⚠️ <b>Warning:</b> Bracket orders not seen in API yet."
            else:
                if abs(bracket["sl"] - sig.sl) > 1:
                    debug_note = f"\n⚠️ <b>SL Mismatch:</b> Bot {sig.sl} vs Broker {bracket['sl']}"

        # --- BUILD MESSAGE ---
        title = sig.action.name
        if sig.action == TradeAction.EXIT:
            res = getattr(sig, "trade_result", "N/A")
            icon = "🟢" if res == "WIN" else "🔴"
            title = f"{icon} EXIT ({res})"
        elif sig.action == TradeAction.ENTER:
            title = "🚀 ENTER"
        elif sig.action == TradeAction.CANCEL:
            title = "🗑️ CANCEL"

        display_price = real_fill_price if (sig.action == TradeAction.EXIT and real_fill_price > 0) else sig.price

        lines = [
            f"<b>{title}</b> | {strategy.log_tag}",
            f"Action: {sig.direction.name} {sig.action.name}",
            f"Price: {display_price:,.2f}",
        ]
        if sig.reason:
            lines.append(f"Reason: {sig.reason}")
        if sig.metadata:
            lines.append(f"<pre>{json.dumps(sig.metadata, indent=2, default=str)}</pre>")

        # Calc PnL
        if sig.action == TradeAction.EXIT and display_price > 0:
            entry_price = sig.metadata.get("entry_price", 0) or (
                strategy.active_trade.price if strategy.active_trade else 0)
            if entry_price > 0:
                points = (display_price - entry_price) if sig.direction == BIAS.LONG else (entry_price - display_price)
                multiplier = getattr(strategy, "point_value", 1.0)
                cash_pnl = points * sig.size * multiplier
                pnl_emoji = "🤑" if cash_pnl > 0 else "🔻"
                lines.append(f"{pnl_emoji} <b>PnL: ${cash_pnl:,.2f}</b> ({points:+.2f} pts)")

        if sig.action == TradeAction.ENTER:
            lines.append(f"Bracket: SL {sig.sl:,.2f} | TP {sig.tp:,.2f}")
            lines.append(f"Size: {sig.size}")

        if debug_note:
            lines.append(debug_note)
        lines.append(f"<i>{sig.timestamp.strftime('%H:%M:%S')} UTC</i>")

        # --- SEND ---
        await strategy.telegram_session_bot.send_message("\n".join(lines), getattr(sig, "fig", None))

        # --- DAILY SUMMARY (Optional Side Effect) ---
        if sig.action == TradeAction.EXIT:
            try:
                await asyncio.sleep(4)
                report = await self.ib_client.get_daily_summary()
                if report:
                    await strategy.telegram_session_bot.send_message(report)
            except Exception as e:
                strategy.log.error(f"Failed to send daily summary: {e}")

    async def _perform_post_execution_safety(self, strategy, sig):
        """
        Validates that Exits actually exited and Entries actually entered.
        """
        # --- A. EXIT VALIDATION ---
        if sig.action == TradeAction.EXIT:
            sig.metadata.get("exit_type", ExitType.SL_TP_HIT)
            await asyncio.sleep(5.0)

            is_flat = False
            pos_size = 0.0

            for attempt in range(5):
                is_flat, pos_size = self.ib_client.check_flatness(sig.symbol)
                if is_flat:
                    break

                if attempt < 4:  # Don't sleep on the last attempt
                    strategy.log.warning(f"⚠️ {sig.symbol} not flat (Size: {pos_size}). Retry {attempt + 1}/3...")
                    await asyncio.sleep(2.0)

            # 4. FINAL REMEDIATION
            if not is_flat:
                if abs(pos_size) > 0:
                    warn_msg = f"🚨 SL/TP HIT, but Position is not flat. Check {strategy.symbol} to have 0 orders!"
                    strategy.log.error(warn_msg)
                    await strategy.telegram_session_bot.send_message(warn_msg)

                    # --- CRITICAL: CANCEL ORDERS FIRST ---
                    # Stop the old SL/TP from triggering while we market close
                    await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)

                    # --- THEN MARKET CLOSE ---
                    await self.ib_client.close_positions_smartly(sig.symbol)
            else:
                # SUCCESS: We are flat.
                # Small extra sleep to ensure 'openTrades' cache is updated by IBKR
                await asyncio.sleep(2.0)

                # Check for Orphan Orders (Hanging SL/TP)
                open_orders = self.ib_client.get_open_orders()  # Using the cache-based getter
                hanging_orders = [o for o in open_orders if o.order.orderRef == strategy.ext_tag]

                if hanging_orders:
                    logger.error(f"⚠️ ORPHAN ORDERS: {len(hanging_orders)} orders remain. Purging.")
                    await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)

        # --- B. ENTRY VALIDATION ---
        elif sig.action == TradeAction.ENTER:
            # We just sleep here because handle_execution already verified order placement.
            # This sleep allows the background task to stay alive until the broker registers the bracket.
            await asyncio.sleep(2.0)

    async def _finalize_locks_and_orphans(self, strategy):
        """Syncs locks and ensures no orphan orders remain. Runs in 'finally' block."""
        # A. Sync Locks
        if strategy.trade_status == TradeStatus.NONE:
            # Release lock if we hold it
            current = self.ACTIVE_SYMBOL_LOCKS.get(strategy.symbol)
            if current and current.strategy_tag == strategy.ext_tag:
                self.ACTIVE_SYMBOL_LOCKS.pop(strategy.symbol, None)
        else:
            # Update/Confirm lock
            self.ACTIVE_SYMBOL_LOCKS[strategy.symbol] = LockInfo(
                strategy_tag=strategy.ext_tag,
                status=strategy.trade_status,
                strategy_instance=strategy,
                priority=getattr(strategy, 'priority', 1),
                timestamp=datetime.now(),
            )

    async def _is_symbol_available(self, strategy, symbol):
        """Checks if the symbol is available for trading."""
        can_trade = await self.acquire_lock(strategy)
        if not can_trade:
            msg = f"🔒{strategy.ext_tag} | {symbol} is locked and used by another strategy. Skipping ENTER signal."
            logger.info(msg)
            await strategy.telegram_session_bot.send_message(msg)
            return False
        return True

    async def _check_and_update_pending_status(self, strategy):
        """Checks if a PENDING trade has actually been filled at the broker."""
        if strategy.trade_status == TradeStatus.PENDING:
            is_flat, pos_size = self.ib_client.check_flatness(strategy.symbol)

            if not is_flat:
                # ✅ WE ARE FILLED!
                strategy.trade_status = TradeStatus.FILLED
                fill_price = await self.ib_client.get_last_fill_price(strategy.symbol)
                msg = f"⚡ ORDER FILLED | {strategy.log_tag} @ {fill_price:.2f}"
                strategy.log.info(msg)
                await strategy.telegram_session_bot.send_message(msg)

    async def run_strategy_cycle(self, strategy, *args, **kwargs):
        """
        Main orchestration loop. Logic is split into helper methods for readability.
        """
        strategy.log.info(f"🔵 run_strategy_cycle({strategy.symbol}) STARTED")

        try:
            fresh_orders = self.ib_client.ib.openTrades()
            fresh_positions = self.ib_client.ib.positions()
            await self.sync_state(strategy, fresh_positions, fresh_orders)
            await self._check_and_update_pending_status(strategy)

            analysis = await strategy.analyze(*args, **kwargs)
            sig = analysis.trade_signal
            if not sig:
                return
            if sig.action == TradeAction.ENTER:
                if not await self._is_symbol_available(strategy, sig.symbol):
                    return

            # EXECUTION & LOCK MANAGEMENT
            success = await self.handle_execution(strategy, analysis)
            if not success:
                strategy.log.error("Execution failed. Aborting cycle.")
                if sig.action == TradeAction.ENTER:
                    self.ACTIVE_SYMBOL_LOCKS.pop(strategy.symbol, None)
                return

            # Update Locks on Success
            if sig.action == TradeAction.ENTER:
                self.ACTIVE_SYMBOL_LOCKS[strategy.symbol] = LockInfo(
                    strategy_tag=strategy.ext_tag,
                    status=strategy.trade_status,
                    strategy_instance=strategy,
                    priority=strategy.priority,
                    timestamp=datetime.now()
                )
            elif sig.action in [TradeAction.EXIT, TradeAction.CANCEL]:
                self.ACTIVE_SYMBOL_LOCKS.pop(strategy.symbol, None)

            # FIRE AND FORGET (Safety and Telemetry)
            # We don't 'await' these, so the cycle can complete and release the gather.
            asyncio.create_task(self._background_post_trade_logic(strategy, sig))

            strategy.log.info(f"🟢 run_strategy_cycle({strategy.symbol}) DISPATCHED")
            locked_symbols = list(self.ACTIVE_SYMBOL_LOCKS.keys())
            strategy.log.info(f"🔒 LOCKED SYMBOLS: {locked_symbols}")

        except Exception as e:
            strategy.log.error(f"Strategy Cycle Error: {traceback.format_exc()}\n{e}")

    async def _background_post_trade_logic(self, strategy, sig):
        """Handles all the slow IBKR/Telegram IO without blocking the main loop."""
        try:
            # Sync locks first
            await self._finalize_locks_and_orphans(strategy)

            # Wait and verify (Safety)
            await self._perform_post_execution_safety(strategy, sig)

            # Send Telegram (Telemetry)
            await self._send_trade_telemetry(strategy, sig)
        except Exception as e:
            strategy.log.error(f"Background logic failed for {strategy.symbol}: {e}")

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
                await asyncio.sleep(1.0)

                # We check active orders for this specific tag
                orders = self.ib_client.get_open_orders()
                my_active_orders = [o for o in orders if o.order.orderRef == strategy.ext_tag]

                if not my_active_orders:
                    # CRITICAL: Order placed but not found 2 seconds later.
                    # It was likely rejected/cancelled immediately.
                    # IBKRClient usually logs errors, but we must stop the flow here.
                    err_msg = f" Order placed for {symbol} but not found in Active list. Likely Rejected."
                    strategy.log.error(err_msg)
                    await strategy.telegram_session_bot.send_message(
                        f"⚠️ <b>ORDER FAILED</b> | {strategy.log_tag}" f"\nReason: {err_msg}")
                    return False

            # --- EXIT ---
            elif sig.action == TradeAction.EXIT:
                # Check if IBKR already closed it for us (The Bracket worked)
                is_flat, pos_size = self.ib_client.check_flatness(symbol)

                if is_flat:
                    strategy.log.info("✅ Position already closed (likely by Bracket). Syncing state.")
                    # Even if flat, run cancel_all to ensure the OTHER side of the bracket is gone
                    # (e.g. SL hit, ensure TP is cancelled)
                    await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)
                    return True

                if exit_type == ExitType.MARKET_FORCE:
                    # CASE A: TIME CUTOFF
                    # 1. Cancel everything (Limit/SL/TP) so they don't trigger while we market close
                    strategy.log.warning("⏰ TIME CUTOFF - FORCING EXIT")
                    await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)
                    # 2. Dump position at Market
                    await self.ib_client.close_positions_smartly(symbol)

                else:
                    strategy.log.info(f"👀 PASSIVE EXIT DETECTED ({sig.trade_result}). Monitor Only.")
                    return True

            elif sig.action == TradeAction.CANCEL:
                strategy.log.info("🚫 CANCEL PENDING ENTRY")
                await self.ib_client.cancel_all_orders(order_ref=strategy.ext_tag)

            return True

        except Exception as e:
            strategy.log.error(f"EXECUTION FAILURE: {e}")
            await strategy.telegram_session_bot.send_message(f"🚨 <b>EXECUTION ERROR</b> | {strategy.log_tag}\n{str(e)}")
            return False

    async def execute_cycle(self, market_data: dict, strategy_list: list):
        """
        Called by MasterBot.
        Needs both 5m and 1h data.
        """
        logger.info(f"🔵 {self.__class__.__name__}.execute_cycle()")
        tasks = []
        await asyncio.gather(self.ib_client.ib.reqPositionsAsync(), self.ib_client.ib.reqOpenOrdersAsync())
        for strategy in strategy_list:
            sym_data = market_data.get(strategy.symbol, {})
            df_5m = sym_data.get("5m")
            df_1h = sym_data.get("1h", None)

            # 3. Validate Data
            if df_5m is None or df_5m.empty:
                continue

            curr = df_5m.iloc[-1]
            ts_now = curr.name.tz_convert(self.timezone)
            is_session_started = strategy.is_pre_session_started(ts_now, strategy.timezone)

            if self.is_strategy_in_window(strategy) and is_session_started:
                dfs = [df_5m]
                if self.__class__.__name__ != "ORBBot" and df_1h is not None:
                    dfs.append(df_1h)
                coro = self.run_strategy_cycle(strategy, *dfs)
                tasks.append(asyncio.wait_for(coro, timeout=45))
            else:
                strategy.log.info(f"💤 {strategy.tz_short} Session has not started yet!")

        if tasks:
            logger.info(f"🔧 Running {len(tasks)} strategy tasks...")
            try:
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=180)

                # Check each result
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # 1. Identify WHICH strategy failed
                        strategy_name = self.strategies[i].symbol if i < len(self.strategies) else "Unknown"

                        # 2. Extract the full stack trace as a string
                        tb_lines = traceback.format_exception(type(result), result, result.__traceback__)
                        tb_text = "".join(tb_lines)

                        # 3. Log the full text clearly
                        logger.error(f"❌ STRATEGY FAILURE: {strategy_name} (Task {i})\n{tb_text}")
                    else:
                        logger.info(f"✅ Task {i} completed successfully")

            except asyncio.TimeoutError:
                logger.error(f"⏰ TIMEOUT: {self.__class__.__name__} tasks took longer than 3 min!")
                raise
            except Exception as e:
                logger.error(f"❌ Unexpected error in gather: {e}", exc_info=True)
                raise
        else:
            logger.info(f"⚪ No active strategies in {self.__class__.__name__}")

        logger.info(f"🟢 {self.__class__.__name__}.execute_cycle() COMPLETED")

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

        logger.info(
            f"Sleeping for {int(sleep_duration)} seconds ({sleep_duration / 60:.2f} minutes) until {next_candle}")
        await asyncio.sleep(sleep_duration)

    async def send_final_message(self, analysis: SignalSummary, current_candle_time, symbol, title):
        events = [e for e in analysis.recent_smc_events if e is not None]
        events_str = "\n• ".join([event[0] if isinstance(event, list) else event for event in events])

        action = f"Go {analysis.direction}." if analysis.direction != Trend.NEUTRAL.name else "Сидим в засаде"
        msg = f"{title} - {symbol} {current_candle_time}\n" f"Action: {action}\n" f"Trend: {analysis.trend}\n" f"SMC Events:\n• {events_str}"
        logger.info(msg)
        is_success = await self.telegram_main_channel.send_message(msg, analysis.fig)

        return is_success

    @staticmethod
    async def sleep_until_next_interval(interval_minutes: int = 5):
        """Synced Sleep."""
        now = datetime.now()
        minutes_to_sleep = interval_minutes - (now.minute % interval_minutes)
        seconds_to_sleep = (minutes_to_sleep * 60) - now.second
        if seconds_to_sleep <= 0:
            seconds_to_sleep = interval_minutes * 60
        logger.info(f"SYNC: Waiting {seconds_to_sleep}s for next candle.")
        await asyncio.sleep(seconds_to_sleep)

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
                    await self.telegram_main_channel.send_message(formatted_ai_message)

            elif analysis.direction == GET_READY_ALERT:
                await self.send_final_message(analysis, last_candle, symbol, title=analysis.title)

            else:
                events = [e for e in analysis.recent_smc_events if e is not None]
                events_str = "\n • ".join([event[0] if isinstance(event, list) else event for event in events])
                msg = f"{symbol}\n{self.strategy_name}\n" f"Action: Сидим в засаде\n" f"Trend: {analysis.trend}\n" f"SMC Events:\n • {events_str}\n"
                logger.info(msg)
                if send_non_signal:
                    await self.telegram_debug_channel.send_message(msg, analysis.fig)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
