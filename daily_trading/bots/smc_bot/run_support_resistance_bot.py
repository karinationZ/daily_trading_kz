import asyncio

from daily_trading.bots.smc_bot.bots.SMCSupportResistanceBot import SMCSupportResistanceBot
from daily_trading.settings import TELEGRAM_MY_DEBUG_BOT_ID


async def main():
    order_block_sweep_bot = SMCSupportResistanceBot(symbols=[],
                                                    lookback_days=10,
                                                    main_tf="15m",
                                                    symbol_volume_threshold=7000000,
                                                    debug=False, telegram_chat_id=TELEGRAM_MY_DEBUG_BOT_ID)
    await order_block_sweep_bot.run()


if __name__ == "__main__":
    asyncio.run(main())
