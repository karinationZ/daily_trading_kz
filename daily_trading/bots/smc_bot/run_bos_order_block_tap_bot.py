import asyncio

from daily_trading.bots.smc_bot.bots.SMCBosOrderBlockTapBot import SMCBosOrderBlockTapBot
from daily_trading.settings import TELEGRAM_CHANNEL_CHAT_ID


async def main():
    order_block_sweep_bot = SMCBosOrderBlockTapBot(symbols=[],
                                                   lookback_days=10,
                                                   main_tf="1h",
                                                   symbol_volume_threshold=7000000,
                                                   debug=False, telegram_chat_id=TELEGRAM_CHANNEL_CHAT_ID)
    await order_block_sweep_bot.run()


if __name__ == "__main__":
    asyncio.run(main())
