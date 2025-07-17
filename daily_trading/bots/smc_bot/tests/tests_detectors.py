import pandas as pd
import pytest

from daily_trading.bots.smc_bot.clients.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.clients.FakeBreakoutDetector import FakeBreakoutDetector
from daily_trading.bots.smc_bot.clients.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.clients.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.clients.LevelsBreakDetector import LevelsBreakDetector
from daily_trading.bots.smc_bot.clients.LiquidityPoolDetector import LiquidityPoolDetector
from daily_trading.bots.smc_bot.clients.LiquiditySweepDetector import LiquiditySweepDetector
from daily_trading.bots.smc_bot.clients.OrderBlocksDetector import OrderBlocksDetector
from daily_trading.bots.smc_bot.clients.SMCTardingBot import SMCTradingBot
from daily_trading.settings import logger

symbol = "PUMPFUNUSDT"
main_tf = "1h"
higher_tf = "4h"
lookback_days = 10


@pytest.mark.asyncio
async def test_bos_detector():
    bot = SMCTradingBot(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())

    detector = BosChochDetector(main_tf, show_bull=True, show_bear=True)
    results = detector.calculate_bos_choch(ohlc.tail(100), plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_fake_breakout_detector():
    bot = SMCTradingBot(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(60).to_string())

    detector = FakeBreakoutDetector(main_tf=main_tf)
    results = detector.calculate_false_breakouts(ohlc.tail(250), plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_fvg_detector():
    bot = SMCTradingBot(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(100).to_string())

    detector = FvgDetector(main_tf=main_tf, pivot_period=5, left_bars=5)
    results = detector.calculate_fvg(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_high_low():
    bot = SMCTradingBot(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(200).to_string())
    detector = HighLowDetector(main_tf)
    results = detector.calculate_high_low_swings(ohlc, plot=True)

    trend = detector.detect_trend(results)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_levels_break():
    bot = SMCTradingBot(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    detector = LevelsBreakDetector(main_tf)
    results = detector.calculate_support_resistance_breaks(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_liquidity_sweep():
    bot = SMCTradingBot(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())

    detector = LiquiditySweepDetector(main_tf)
    results = detector.calculate_liquidity_sweeps(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_order_blocks():
    bot = SMCTradingBot(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())

    detector = OrderBlocksDetector(main_tf)
    results = detector.calculate_order_blocks(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_liq_pool():
    bot = SMCTradingBot(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())

    detector = LiquidityPoolDetector(main_tf="1h")
    results = detector.calculate_liquidity_pools(ohlc.tail(100), plot=True)
    fig = results["fig"]
    fig.show()

    # bot = TelegramBot()
    # await bot.send_message(
    #     text="Important alert: Something happened",
    #     image=fig
    # )
