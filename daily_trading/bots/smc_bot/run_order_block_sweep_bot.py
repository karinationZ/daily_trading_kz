import asyncio

from daily_trading.bots.smc_bot.bots.SMCOrderBlockSweepBot import SMCOrderBlockSweepBot


async def main():
    order_block_sweep_bot = SMCOrderBlockSweepBot(symbols=[],
                                                  lookback_days=10,
                                                  main_tf="1h",
                                                  symbol_volume_threshold=10000000,
                                                  debug=False)
    await order_block_sweep_bot.run()


if __name__ == "__main__":
    asyncio.run(main())
