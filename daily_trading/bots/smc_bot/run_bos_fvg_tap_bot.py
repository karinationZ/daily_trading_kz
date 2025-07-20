import asyncio

from daily_trading.bots.smc_bot.bots.SMCBosFvgTapBot import SMCBosFvgTapBot
from daily_trading.settings import TELEGRAM_CHANNEL_CHAT_ID


async def main():
    bot = SMCBosFvgTapBot(symbols=[],
                          lookback_days=10,
                          main_tf="1h",
                          symbol_volume_threshold=10000000,
                          debug=False, telegram_chat_id=TELEGRAM_CHANNEL_CHAT_ID)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
