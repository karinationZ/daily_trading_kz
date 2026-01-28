import asyncio

from daily_trading.bots.smc_bot.bots.ibkr.ibkr_master_bot import MasterBot
from daily_trading.settings import logger

if __name__ == "__main__":
    bot = MasterBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
