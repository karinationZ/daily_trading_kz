import asyncio
from datetime import datetime, timedelta

import pandas as pd
import pytz

from daily_trading.bots.smc_bot.bots.TelegramBot import TelegramBot
from daily_trading.bots.smc_bot.detectors.SMCBase import GET_READY_ALERT, Signal, Trend
from daily_trading.bots.smc_bot.strategy.BosFvgFillStrategy import SignalSummary
from daily_trading.brokers.BitUnixClient import BitUnixClient
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID, TELEGRAM_DEBUG_CHANNEL_ID, USE_AI
from daily_trading.technicals.AITradeAssistants.GeminiTradeAssistant import GEMINI_PRO, GeminiTradeAssistant, Ohlc
from daily_trading.technicals.DivergentAssetsFinder import BitUnixDivergentAssetsFinder


class SMCBotBase:
    """SMC-based trading bot for high-probability continuation trades across multiple symbols.
    Uses SMCAnalyser for analysis and Telegram messaging, targeting scalping opportunities
    in volatile crypto markets with 1H/4H trend alignment, 15m liquidity pools between
    1H HH/HL or LH/LL, and FVG-based entries.
    """

    def __init__(self, symbols, main_tf: str = '1h',
                 symbol_volume_threshold: float = 30000000,
                 lookback_days: int = 1,
                 telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID
                 ):
        """
        Initialize SMCTradingBot.

        Args:
            symbols (list): Trading pair symbols (e.g., ['BTC/USDT', 'ETH/USDT']).
            main_tf (str): Main timeframe for trend/BOS (e.g., '1h').
            symbol_volume_threshold (float): Minimum volume for symbol selection.
            lookback_days (int): Lookback period in days for OHLC data.
        """
        self.static_symbols = symbols
        self.symbols = symbols
        self.telegram_bot = TelegramBot(chat_id=telegram_chat_id)
        self.telegram_bot_debug = TelegramBot(chat_id=TELEGRAM_DEBUG_CHANNEL_ID)
        self.main_tf = main_tf
        self.volume_threshold = symbol_volume_threshold
        self.lookback_days = lookback_days
        self.default_symbols = ["BTCUSDT"]
        self.client = BitUnixClient()
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
    async def sleep_until_next_interval(interval_minutes: int = 15):
        """
        Sleep until the next specified time interval (e.g., 15, 20 minutes).
        """
        # Get current time in UTC
        now = datetime.now(pytz.UTC)

        # Round to next interval mark
        minute = ((now.minute // interval_minutes) + 1) * interval_minutes
        next_candle = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minute)

        # Handle overflow (e.g., minute == 60)
        if next_candle.minute >= 60:
            next_candle = next_candle.replace(minute=0) + timedelta(hours=1)

        # Time to sleep (in seconds)
        sleep_duration = (next_candle - now).total_seconds()

        logger.info(f"Sleeping for {int(sleep_duration) / 60:.2f} minutes until {next_candle}")
        await asyncio.sleep(sleep_duration)

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
        return is_success

    async def fetch_ohlc_data(self, symbol: str, timeframe: str, num_candles: int = None) -> pd.DataFrame:
        """Fetch OHLC data with caching to reduce API calls.

        Args:
            symbol (str): Trading pair (e.g., 'BTC/USDT').
            timeframe (str): Timeframe (e.g., '1h', '15m').
            num_candles (int, optional): Number of candles to fetch.

        Returns:
            pd.DataFrame: OHLC data with time index and float columns.
        """
        logger.info(f"Fetching OHLC data for {symbol} ({timeframe})")
        timeframe_mapping = {
            "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1d"
        }
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
                kline_data = self.client.get_klines(
                    symbol=symbol,
                    interval=timeframe_mapping.get(timeframe, "1h"),
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
                logger.error(f"No kline data for {symbol} ({timeframe})")
                return pd.DataFrame()

            df = pd.DataFrame(all_klines, columns=["time", "open", "high", "low", "close", "quoteVol"])
            df = df.rename(columns={"quoteVol": "volume"})
            df["time"] = pd.to_datetime(pd.to_numeric(df["time"]), unit="ms", utc=True)
            df["symbol"] = symbol
            df.set_index("time", inplace=True)
            numeric_columns = ["open", "high", "low", "close", "volume"]
            df[numeric_columns] = df[numeric_columns].astype(float)
            df = df[["symbol", "open", "high", "low", "close", "volume"]]
            logger.info(f"Cached OHLC data for {symbol} ({timeframe}): {len(df)} rows")
            return df.sort_values(by="time")
        except Exception as e:
            logger.error(f"Error fetching OHLC for {symbol} ({timeframe}): {e}")
            raise e

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

            if not self.symbols:
                logger.info(f"There are no divergent assets, using default symbols - {self.default_symbols}")
                self.symbols = self.default_symbols

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

            if analysis.direction in [Signal.LONG.value, Signal.SHORT.value]:
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
