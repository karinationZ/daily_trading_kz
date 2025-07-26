import asyncio
import traceback

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.BosFvgFillStrategy import BosFvgFillStrategy
from daily_trading.settings import logger, TELEGRAM_CHANNEL_CHAT_ID


class SMCBosFvgFillBot(SMCBotBase):
    """
    Analyzes OHLC data using Smart Money Concept (SMC) for continuation trade signals.
    Designed for scalping in volatile crypto markets with a 6-step process:
    1. Identify Trend 1H (last 24 candles)
    2. Detect Break of Structure (BOS) on 1H; reset on CHOCH.
    4. Find unmitigated FVG below (uptrend) or above (downtrend) on 15m.
    5. Confirm price touches the nearest FVG on 15m.
    6. Generate LONG/SHORT signal
    """

    def __init__(self, symbols, main_tf: str = '1h', symbol_volume_threshold: float = 30000000, lookback_days: int = 1,
                 debug=False, telegram_chat_id=TELEGRAM_CHANNEL_CHAT_ID):

        super().__init__(symbols, main_tf,
                         symbol_volume_threshold,
                         lookback_days,
                         telegram_chat_id=telegram_chat_id)

        self.chat_id = telegram_chat_id
        self.bos_strategy = BosFvgFillStrategy(main_tf, debug, telegram_chat_id=self.chat_id)
        self.strategy_name = "🎯BOS + FVG Fill"

    async def run_symbol(self, symbol: str):
        """Analyze a single symbol and generate SMC signals."""
        logger.info(f"Starting analysis for {symbol}")
        try:
            ohlc_1d = await self.fetch_ohlc_data(symbol, "1d", num_candles=14)
            ohlc_4h = await self.fetch_ohlc_data(symbol, "4h", num_candles=24)
            ohlc_1h = await self.fetch_ohlc_data(symbol, "1h", num_candles=120)
            ohlc_15m = await self.fetch_ohlc_data(symbol, "15m", num_candles=100)

            if ohlc_1h.empty or ohlc_15m.empty:
                logger.error(f"No data for {symbol}, skipping...")
                await self.bos_strategy.telegram_bot.send_message(f"No data for {symbol}")
                return

            analysis = await self.bos_strategy.analyze_smc(symbol, ohlc_1h=ohlc_1h, ohlc_15m=ohlc_15m)
            analysis.title = self.strategy_name
            await self.generate_signal(symbol, analysis,
                                       ohlc_1d=ohlc_1d,
                                       ohlc_4h=ohlc_4h,
                                       ohlc_1h=ohlc_1h,
                                       ohlc_15m=ohlc_15m)

        except Exception as e:
            # Enhanced error logging with full traceback
            error_msg = f"Error processing {symbol}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Full traceback for {symbol}:")
            logger.error(traceback.format_exc())

    async def run(self):
        """Main loop for monitoring all symbols.
        Update symbols every 2 hours and runs analysis for each symbol concurrently.
        """
        logger.info("Starting SMCTradingBot main loop")
        await self.sleep_until_next_interval(5)
        while True:
            try:
                if not self.static_symbols:
                    needs_refresh = await self.needs_symbol_refresh()
                    if needs_refresh:
                        await self.update_symbols()
                logger.info(f"Monitoring {len(self.symbols)} symbols")
                await self.telegram_bot_debug.send_message(f"****** Monitoring {len(self.symbols)} symbols ******")
                tasks = [self.run_symbol(symbol) for symbol in self.symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for symbol, result in zip(self.symbols, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {symbol}: {result}")
                await self.sleep_until_next_interval()

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await self.sleep_until_next_interval()
                continue
