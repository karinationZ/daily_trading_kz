import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytz

from daily_trading.bots.smc_bot.bots.TelegramBot import TelegramBot
from daily_trading.bots.smc_bot.detectors.ORBDetector import SESSION_MAP
from daily_trading.bots.smc_bot.detectors.SMCBase import BIAS, GET_READY_ALERT, Trend
from daily_trading.bots.smc_bot.strategy.BosFvgFillStrategy import SignalSummary
from daily_trading.bots.smc_bot.strategy.StrategyBase import TradeAction, TradeSignal, TradeStatus
from daily_trading.brokers.BitUnixClient import BitUnixClient
from daily_trading.brokers.IBKRClient import IBKRClient
# from daily_trading.brokers.YFinance import YFinanceClient
from daily_trading.settings import IBKR_PORT, logger, TELEGRAM_DEBUG_CHANNEL_ID, TELEGRAM_MAIN_CHANNEL_CHAT_ID, \
    TELEGRAM_MY_DEBUG_BOT_ID, USE_AI
from daily_trading.technicals.AITradeAssistants.GeminiTradeAssistant import GEMINI_PRO, GeminiTradeAssistant, Ohlc
from daily_trading.technicals.DivergentAssetsFinder import BitUnixDivergentAssetsFinder


# ==========================
# TimeZone Container
# ==========================
@dataclass(frozen=True)
class TimeZone:
    UTC = ZoneInfo("UTC")
    NY = ZoneInfo("America/New_York")
    LONDON = ZoneInfo("Europe/London")


class SMCBotBase:
    """SMC-based trading bot for high-probability continuation trades across multiple symbols.
    Uses SMCAnalyser for analysis and Telegram messaging, targeting scalping opportunities
    in volatile crypto markets with 1H/4H trend alignment, 15m liquidity pools between
    1H HH/HL or LH/LL, and FVG-based entries.
    """

    def __init__(self, symbols, main_tf: str = '1h',
                 symbol_volume_threshold: float = 30000000,
                 lookback_days: int = 1,
                 telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID,
                 market_type: str = "crypto",
                 ib_client=None):
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
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "2h": 120, "4h": 240, "6h": 360, "12h": 720, "1d": 1440
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
            "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
            "4h": 240, "6h": 360, "12h": 720, "1d": 1440
        }
        lower_minutes = timeframe_minutes.get(self.main_tf, 60)
        lookback = {
            self.main_tf: int(num_days * 24 * 60 / lower_minutes),
            "15m": int(num_days * 24 * 60 / 15)  # Added for 15m timeframe
        }
        logger.info(f"Lookback: {self.main_tf}={lookback[self.main_tf]} candles, "
                    f"15m={lookback['15m']} candles")
        return lookback

    @staticmethod
    def sync_state(strategy, current_positions: list, open_orders: list):
        """
        Restores self.active_trade and self.trade_status from IBKR data.
        """
        # If we already have state in memory, do nothing
        if strategy.trade_status != TradeStatus.NONE:
            return

        # 2. Check for ACTIVE POSITIONS
        my_pos = next((p for p in current_positions if p.contract.symbol == strategy.symbol and p.position != 0), None)

        if my_pos:
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
                metadata={"zone_id": "RECOVERED"}  # We lose the original Zone ID, but that's acceptable
            )
            strategy.trade_status = TradeStatus.FILLED
            strategy.log.info(f"♻️ RESTORED STATE: FILLED {direction.name} @ {avg_price} (SL:{sl_price} TP:{tp_price})")
            return

        # 3. Check for PENDING ORDERS (Limit Entry Waiting)
        my_orders = [o for o in open_orders if o.order.orderRef == strategy.ext_tag]

        # Find the Parent Entry Order (Limit order with no parentId usually)
        # Note: In a bracket, the parent has parentId=0 (or empty)
        parent = next((t for t in my_orders if t.order.parentId == 0 and t.order.orderType in ['LMT', 'STP LMT']), None)

        if parent:
            direction = BIAS.LONG if parent.order.action == 'BUY' else BIAS.SHORT
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

            strategy.active_trade = TradeSignal(
                action=TradeAction.ENTER,
                symbol=strategy.symbol,
                direction=direction,
                price=entry_price,
                sl=sl_price,
                tp=tp_price,
                size=size,
                timestamp=pd.Timestamp.now(tz=strategy.timezone),
                metadata={"zone_id": "RECOVERED"}
            )
            strategy.trade_status = TradeStatus.PENDING
            strategy.log.info(f"♻️ RESTORED STATE: PENDING {direction.name} @ {entry_price}")

    async def handle_execution(self, strategy, analysis: SignalSummary):
        """
        Translates TradeActions into IBKR Orders.
        Uses a unique 'order_ref' tag per strategy instance to prevent collisions.
        """
        if not analysis or not analysis.trade_signal:
            return

        sig = analysis.trade_signal
        symbol = strategy.symbol
        try:
            # --- ENTER ---
            if sig.action == TradeAction.ENTER:
                strategy.log.info(f"SIGNAL: ENTER {sig.direction.name} @ {sig.price}. Placing Bracket.")
                direction = "LONG" if sig.direction == BIAS.LONG else "SHORT"
                await self.ib_client.place_bracket_order(
                    symbol=symbol,
                    quantity=sig.size,
                    entry=sig.price,
                    sl=sig.sl, tp=sig.tp,
                    direction=direction,
                    order_ref=strategy.ext_tag
                )

            # --- CANCEL ---
            elif sig.action == TradeAction.CANCEL:
                strategy.log.info(f"SIGNAL: CANCEL. Cleaning book.")
                await self.ib_client.cancel_all_orders(order_ref=ext_tag)

            # --- EXIT ---
            elif sig.action == TradeAction.EXIT:
                # Check for EOD hard exit
                ts_now = datetime.now(strategy.timezone)
                s_info = SESSION_MAP[strategy.timezone.key]
                end_hour = s_info.get("end_hour", 16)
                end_minute = s_info.get("end_minute", 0)

                # Check if we are in the last 6 minutes of the session
                session_end = ts_now.replace(hour=end_hour, minute=end_minute, second=0)
                is_eod = ts_now >= (session_end - timedelta(minutes=6))

                if is_eod:
                    strategy.log.info(f"EOD LIQUIDATION: Market Exit.")
                    await self.ib_client.cancel_all_orders(order_ref=ext_tag)
                    side = "SELL" if sig.direction == BIAS.LONG else "BUY"
                    await self.ib_client.place_market_order(symbol, quantity=sig.size, side=side, order_ref=ext_tag)
                else:
                    # Normal TP/SL Hit logic
                    strategy.log.info(f"STRATEGY EXIT: Cleaning up.")
                    await self.ib_client.cancel_all_orders(order_ref=ext_tag)

        except Exception as e:
            strategy.log.error(f"EXECUTION FAILURE: {e}", level="error")

    @staticmethod
    async def sleep_until_next_interval(interval_minutes: int = 5):
        """Synced Sleep."""
        now = datetime.now()
        minutes_to_sleep = interval_minutes - (now.minute % interval_minutes)
        seconds_to_sleep = (minutes_to_sleep * 60) - now.second
        if seconds_to_sleep <= 0: seconds_to_sleep = interval_minutes * 60
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
        PRE_SESSION_BUFFER = 60  # Start 1 hour before session open
        POST_SESSION_BUFFER = 30  # End 30 mins after session close

        # 3. Construct Session Start/End times for Today
        start_time = now.replace(
            hour=s_info.get("start_hour", 9),
            minute=s_info.get("start_minute", 0),
            second=0, microsecond=0
        )
        end_time = now.replace(
            hour=s_info.get("end_hour", 16),
            minute=s_info.get("end_minute", 0),
            second=0, microsecond=0
        )

        # 4. Apply Buffers
        active_start = start_time - timedelta(minutes=PRE_SESSION_BUFFER)
        active_end = end_time + timedelta(minutes=POST_SESSION_BUFFER)

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
        events_str = '\n• '.join([event[0] if isinstance(event, list) else event for event in events])

        action = f"Go {analysis.direction}." if analysis.direction != Trend.NEUTRAL.name else "Сидим в засаде"
        msg = (f"{title} - {symbol} {current_candle_time}\n"
               f"Action: {action}\n"
               f"Trend: {analysis.trend}\n"
               f"SMC Events:\n• {events_str}")
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
                    kline_type="LAST_PRICE"
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
            kline_data = kline_data[~kline_data.index.duplicated(keep='last')]

            # 2. Sort by Index (Chronological)
            kline_data.sort_index(inplace=True)

            # 3. Add 'time' column from Index
            kline_data['time'] = kline_data.index
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
            results = await self.asset_finder.find_divergent_assets(
                volume_threshold=7000000,
                lookback_hours=3
            )
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

    async def generate_signal(self, symbol: str, analysis: SignalSummary,
                              ohlc_1d: pd.DataFrame,
                              ohlc_4h: pd.DataFrame,
                              ohlc_1h: pd.DataFrame,
                              ohlc_15m: pd.DataFrame, send_non_signal=True):
        """Generate SMC signals."""
        try:
            last_candle = ohlc_15m.index.max().tz_convert('US/Pacific').strftime('%Y-%m-%d %H:%M')

            if analysis.direction in [BIAS.LONG.value, BIAS.SHORT.value]:
                # AI Request
                ohlc = Ohlc(ohlc_1d, ohlc_4h, ohlc_1h, ohlc_15m)
                ai_agent = GeminiTradeAssistant(symbol=symbol,
                                                model=GEMINI_PRO,
                                                ohlc=ohlc,
                                                debug=False)
                within_ai_hours = ai_agent.within_ai_request_hours(start_hour=8, end_hour=21)
                formatted_ai_message = None
                if within_ai_hours and USE_AI.lower() == "true":
                    formatted_ai_message, raw_message = await ai_agent.analyze_asset_trade_data()
                    if raw_message['entry']:
                        fig = ai_agent.update_plot_with_entry_data(analysis.fig, raw_message)
                        analysis.fig = fig
                is_success = await self.send_final_message(analysis, last_candle, symbol, title=analysis.title)

                if formatted_ai_message and is_success:
                    await self.telegram_bot.send_message(formatted_ai_message)

            elif analysis.direction == GET_READY_ALERT:
                await self.send_final_message(analysis, last_candle, symbol, title=analysis.title)

            else:
                events = [e for e in analysis.recent_smc_events if e is not None]
                events_str = '\n • '.join([event[0] if isinstance(event, list) else event for event in events])
                msg = f"{symbol}\n{self.strategy_name}\n" \
                      f"Action: Сидим в засаде\n" \
                      f"Trend: {analysis.trend}\n" \
                      f"SMC Events:\n • {events_str}\n"
                logger.info(msg)
                if send_non_signal:
                    await self.telegram_bot_debug.send_message(msg, analysis.fig)
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
