import asyncio

from daily_trading.bots.smc_bot.clients.SMCTardingBot import SMCTradingBot


async def main():
    bot = SMCTradingBot(symbols=['PUMPFUNUSDT'],
                        lookback_days=10,
                        main_tf="1h",
                        symbol_volume_threshold=10000000,
                        debug=False)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
