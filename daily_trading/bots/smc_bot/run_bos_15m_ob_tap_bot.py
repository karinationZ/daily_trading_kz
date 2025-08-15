import asyncio

from daily_trading.bots.smc_bot.bots.SMCBosOrderBlockTapBot import SMCBosOrderBlockTapBot
from daily_trading.settings import TELEGRAM_MAIN_CHANNEL_CHAT_ID


async def main():
    bos_15m_ob_tap_bot = SMCBosOrderBlockTapBot(symbols=[],
                                                lookback_days=10,
                                                main_tf="1h",
                                                symbol_volume_threshold=7000000,
                                                debug=False, telegram_chat_id=TELEGRAM_MAIN_CHANNEL_CHAT_ID)
    await bos_15m_ob_tap_bot.run()


if __name__ == "__main__":
    asyncio.run(main())
