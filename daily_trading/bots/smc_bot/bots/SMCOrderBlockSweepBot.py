import asyncio

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.OrderBlockSweep import OrderBlockSweepStrategy
from daily_trading.settings import logger, TELEGRAM_MAIN_CHANNEL_CHAT_ID


class SMCOrderBlockSweepBot(SMCBotBase):
    """
    Analyzes OHLC mgc for continuation trade signals based on block order sweeps.
    Designed for scalping in volatile crypto markets with a 4-step process:
    1. Identify Uptrend or Downtrend on 1H (last 24 candles).
    2. Detect Block Orders (OB) on 1H.
    3. For bullish trend, wait for price to pull back and tap into bullish OB below current price;
       for bearish trend, wait for price to tap into bearish OB above current price.
    4. Confirm OB mitigation with 15m candlestick rejection patterns, generate Buy (bullish) or Sell (bearish) signal
       with entry, SL, and TP (1:2 RR).
    Sends Telegram messages with plots (OHLC.tail(100)) for each step, including FVGs in final plot for visibility.
    """

    def __init__(
        self,
        symbols,
        main_tf: str = "1h",
        symbol_volume_threshold: float = 30000000,
        lookback_days: int = 1,
        debug=False,
        telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID,
    ):

        super().__init__(
            symbols,
            main_tf,
            symbol_volume_threshold,
            lookback_days,
            telegram_chat_id=telegram_chat_id,
        )
        self.ob_sweep_strategy = OrderBlockSweepStrategy(main_tf, debug, telegram_chat_id=telegram_chat_id)
        logger.info(f"SMCOrderBlockSweepBot initialized with {len(symbols)} symbols, {main_tf} TF.")
        self.strategy_name = "🧱🧹1H Order Block Sweep"

    async def run_symbol(self, symbol: str):
        """Analyze a single symbol and generate SMC signals."""
        logger.info(f"Starting analysis for {symbol}")
        try:
            ohlc_1d = await self.fetch_bitunix_ohlc_data(symbol, "1d", num_candles=14)
            ohlc_4h = await self.fetch_bitunix_ohlc_data(symbol, "4h", num_candles=24)
            ohlc_1h = await self.fetch_bitunix_ohlc_data(symbol, "1h", num_candles=120)
            ohlc_15m = await self.fetch_bitunix_ohlc_data(symbol, "15m", num_candles=100)

            if any(ohlc.empty for ohlc in [ohlc_1d, ohlc_4h, ohlc_1h, ohlc_15m]):
                logger.error(f"No mgc for {symbol}, skipping...")
                return

            analysis = await self.ob_sweep_strategy.analyze_smc(symbol, ohlc_1h)
            analysis.title = self.strategy_name
            await self.generate_signal(
                symbol,
                analysis,
                ohlc_1d=ohlc_1d,
                ohlc_4h=ohlc_4h,
                ohlc_1h=ohlc_1h,
                ohlc_15m=ohlc_15m,
            )

        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")

    async def run(self):
        """Main loop for monitoring all symbols.
        Update symbols every 2 hours and runs analysis for each symbol concurrently.
        """
        logger.info("Starting SMCTradingBot main loop")
        while True:
            try:
                if not self.static_symbols:
                    needs_refresh = await self.needs_symbol_refresh()
                    if needs_refresh:
                        await self.update_symbols()
                logger.info(f"Monitoring {len(self.symbols)} symbols")
                await self.telegram_bot_debug.send_message(f"****** SMCOrderBlockSweepBot - Monitoring {len(self.symbols)} symbols ******")
                tasks = [self.run_symbol(symbol) for symbol in self.symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for symbol, result in zip(self.symbols, results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing {symbol}: {result}")
                await self.sleep_until_next_hour()

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await self.sleep_until_next_hour()
                continue
