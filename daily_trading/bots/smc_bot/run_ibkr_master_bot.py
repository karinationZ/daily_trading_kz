import asyncio

from daily_trading.bots.smc_bot.bots.ibkr.ibkr_master_bot import MasterBot

if __name__ == "__main__":
    bot = MasterBot()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("Bot stopped by user.")
