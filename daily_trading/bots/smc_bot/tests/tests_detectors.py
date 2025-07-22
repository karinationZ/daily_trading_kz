import time

import pandas as pd
import pytest

from daily_trading.bots.smc_bot.bots.TelegramBot import TelegramBot
from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.FakeBreakoutDetector import FakeBreakoutDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.LevelsBreakDetector import LevelsBreakDetector
from daily_trading.bots.smc_bot.detectors.LiquidityPoolDetector import LiquidityPoolDetector
from daily_trading.bots.smc_bot.detectors.LiquiditySweepDetector import LiquiditySweepDetector
from daily_trading.bots.smc_bot.detectors.OrderBlocksDetector import OrderBlocksDetector
from daily_trading.settings import logger, TELEGRAM_BOT_CHAT_ID

symbol = "CROSSUSDT"
main_tf = "15m"
lookback_days = 10


@pytest.mark.asyncio
async def test_trend():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)
    detector = HighLowDetector(main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())

    trend = detector.detect_trend(ohlc)
    logger.info(trend)


@pytest.mark.asyncio
async def test_bos_detector():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info("\n{}", ohlc.tail(50).to_string())

    detector = BosChochDetector(main_tf, show_bull=True, show_bear=True)
    results = detector.calculate_bos_choch(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_fake_breakout_detector():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(60).to_string())

    detector = FakeBreakoutDetector(main_tf=main_tf)
    results = detector.calculate_false_breakouts(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_fvg_detector():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(100).to_string())

    detector = FvgDetector(main_tf=main_tf, pivot_period=5, left_bars=5)
    results = detector.calculate_fvg(ohlc, plot=True)
    fig_1 = results["fig"]

    detectors = {
        'fvg_15m': FvgDetector(main_tf="15m", pivot_period=5, left_bars=5)  # Detects FVGs on 15m
    }
    results = detectors["fvg_15m"].calculate_fvg(ohlc, plot=True)
    fig_2 = results["fig"]
    fig_2.show()
    assert fig_2 == fig_1


@pytest.mark.asyncio
async def test_high_low():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())
    detector = HighLowDetector(main_tf)
    results = detector.calculate_high_low_swings(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_levels_break():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    detector = LevelsBreakDetector(main_tf)
    results = detector.calculate_support_resistance_breaks(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_liquidity_sweep():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())

    detector = LiquiditySweepDetector(main_tf)
    results = detector.calculate_liquidity_sweeps(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_order_blocks():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info("\n{}", ohlc.tail(50).to_string())

    detector = OrderBlocksDetector(main_tf)
    results = detector.calculate_order_blocks(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_liq_pool():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info("\n{}", ohlc.tail(50).to_string())

    detector = LiquidityPoolDetector(main_tf="1h")
    results = detector.calculate_liquidity_pools(ohlc, plot=True)
    fig = results["fig"]
    fig.show()
    bot = TelegramBot(chat_id=TELEGRAM_BOT_CHAT_ID)

    for i in range(10):
        await bot.send_message(
            text="TEST TEST TEST",
            image=fig
        )
        time.sleep(1)


@pytest.mark.asyncio
async def test_fvg_bos_merge():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    # ohlc_1h = await bot.fetch_ohlc_data(symbol, "1h", num_candles=180)
    ohlc_15m = await bot.fetch_ohlc_data(symbol, "15m", num_candles=100)
    ohlc_15m = ohlc_15m[:-2]

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc_15m.tail(50).to_string())


    detector_fvg = FvgDetector(main_tf="15min", pivot_period=5, left_bars=5)
    # detector_bos = BosChochDetector(main_tf, show_bull=True, show_bear=True)
    # bos_results = detector_bos.calculate_bos_choch(ohlc_1h, plot=False)
    fvg_results = detector_fvg.calculate_fvg(ohlc_15m, plot=True)
    logger.info(fvg_results["signals"])
    fig = fvg_results["fig"]
    # fig = detector_fvg.merge_bos_fvg_plots(ohlc_15m, bos_signals=bos_results["signals"], fvg_signals=fvg_results["signals"])
    fig.show()