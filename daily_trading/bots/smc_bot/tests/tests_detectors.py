import asyncio
import os

import pandas as pd
import pytest

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase, TimeZone
from daily_trading.bots.smc_bot.bots.TelegramBot import TelegramBot
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.FakeBreakoutDetector import FakeBreakoutDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.LevelsDetector import LevelsDetector
from daily_trading.bots.smc_bot.detectors.LiquidityGrabDetector import LiquidityGrabDetector
from daily_trading.bots.smc_bot.detectors.LiquidityPoolDetector import LiquidityPoolDetector
from daily_trading.bots.smc_bot.detectors.LiquiditySweepDetector import LiquiditySweepDetector
from daily_trading.bots.smc_bot.detectors.ORBDetector import ORBDetector
from daily_trading.bots.smc_bot.detectors.OrderBlocksDetectorOld import OrderBlocksDetectorOld
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.detectors.VolumeProfiler import VolumeProfiler
from daily_trading.brokers.IBKRClient import IBKRClient  # Adjust import path as needed
from daily_trading.settings import logger, PROJECT_DIR, TELEGRAM_ASIA_MONEY_MAKER_CHANNEL_ID, TELEGRAM_BOT_CHAT_ID, \
    TELEGRAM_LONDON_MONEY_MAKER_CHANNEL_ID
from daily_trading.utils.common import load_data, merge_csv_no_dups

symbol = "BTCUSDT"
main_tf = "1h"
lookback_days = 5

# CONFIGURATION
# Ensure this matches your running TWS/Gateway (Paper Trading recommended)
HOST = '127.0.0.1'
PORT = 4002
CLIENT_ID = 999  # Use a unique ID for testing to avoid conflict with running bots

# The symbols we must verify
TARGET_SYMBOLS = ["MNQ", "MES", "MGC", "SIL"]


@pytest.mark.asyncio
async def test_trend():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, main_tf)
    detector = HighLowDetector(main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())

    trend = detector.detect_trend(ohlc)
    logger.info(trend)


@pytest.mark.asyncio
async def test_fake_breakout_detector():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(60).to_string())

    detector = FakeBreakoutDetector(main_tf=main_tf)
    results = detector.calculate_false_breakouts(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_fvg_detector():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, main_tf)

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
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, main_tf)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())
    detector = HighLowDetector(main_tf)
    results = detector.calculate_high_low_swings(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_liquidity_sweep():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, main_tf, num_candles=120)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info("\n{}", ohlc.tail(50).to_string())

    detector = LiquiditySweepDetector(main_tf)
    results = detector.calculate_liquidity_sweeps(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_liq_pool():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, "1h")

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info("\n{}", ohlc.tail(50).to_string())

    detector = LiquidityPoolDetector(main_tf="1h")
    results = detector.calculate_liquidity_pools(ohlc, plot=True)
    fig = results["fig"]
    fig.show()
    # bot = TelegramBot(chat_id=TELEGRAM_BOT_CHAT_ID)
    #
    # await bot.send_message(
    #     text="<b>Bold text</b>")


@pytest.mark.asyncio
async def test_vol_prof():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, "1h")

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.to_string())

    detector = VolumeProfiler(main_tf="1h")
    results = detector.calculate_volume_profile(ohlc, plot=True)
    fig = results["fig"]
    # fig.show()
    bot = TelegramBot(chat_id=TELEGRAM_BOT_CHAT_ID)

    await bot.send_message(
        text="""🎯BOS + FVG Fill - CFXUSDT 2025-07-25 08:30
Action: Go LONG.
Trend: BULLISH
SMC Events:
• BULLISH trend confirmed.
• BOS at 2025-07-25 10:00
• Found nearest BULLISH FVG.
• Current price is in BULLISH FVG.
• Price mitigated BULLISH FVG by 74.55% (>= 60%)
• Generated LONG signal for CFXUSDT based on FVG mitigation.")
""", image=fig)


@pytest.mark.asyncio
async def test_merge_ob_and_bos():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc_1h = await bot.fetch_bitunix_ohlc_data(symbol, "1h", num_candles=120)
    ohlc_15m = await bot.fetch_bitunix_ohlc_data(symbol, "15m", num_candles=100)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info("\n{}", ohlc.tail(50).to_string())

    ob_detector = OrderBlocksDetectorOld(main_tf)
    results = ob_detector.calculate_order_blocks(ohlc_15m, plot=True)
    ob_signals = results["order_blocks"]

    detector = BosChochDetector(main_tf, show_bull=True, show_bear=True)
    results = detector.calculate_bos_choch(ohlc_1h, plot=True)
    bos_signals = results["signals"]

    fig = ob_detector.merge_bos_ob_plots(ohlc_15m, bos_signals, ob_signals=ob_signals)
    fig.show()

    bot = TelegramBot(chat_id=TELEGRAM_BOT_CHAT_ID)
    await bot.send_message(text="TEST TEST Message", image=fig)


@pytest.mark.asyncio
async def test_liq_grab():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, "5m", num_candles=120)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info("\n{}", ohlc.tail(50).to_string())

    ob_detector = LiquidityGrabDetector(main_tf)
    results = ob_detector.calculate_liquidity_grabs(ohlc, plot=True)
    logger.info(results["signals"])
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_support_resistance_levels():
    symbol = "MGC"
    pathlib = f"/project/daily_trading/bots/smc_bot/tests/ohlc_data/{symbol.lower()}_1h.csv"
    ohlc = load_data(
        pathlib,
        start_ts="2025-07-01 00:00:00+00:00",
        end_ts="2025-07-30 00:00:00+00:00"
    )
    # print(ohlc.tail(500).to_string())

    detector = LevelsDetector(main_tf="1h")
    results = detector.calculate_major_levels(ohlc)
    print(results)
    fig = detector.plot_levels(ohlc, results)
    fig.show()


@pytest.mark.asyncio
async def test_telegram():
    ny = TelegramBot(chat_id="-1000000000000")
    london = TelegramBot(chat_id=TELEGRAM_LONDON_MONEY_MAKER_CHANNEL_ID)
    asia = TelegramBot(chat_id=TELEGRAM_ASIA_MONEY_MAKER_CHANNEL_ID)
    await ny.send_message("🟢 Test Message from NY Money Maker Channel")
    await london.send_message("🟢 Test Message from London Money Maker Channel")
    await asia.send_message("🟢 Test Message from Asia Money Maker Channel")


@pytest.mark.asyncio
async def test_orb_calculation():
    symbol = "MGC"
    bot = SMCBotBase(symbols=[symbol], main_tf="1h", market_type="futures")
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, "5m", num_candles=500)

    orb = ORBDetector(timezone=TimeZone.NY)

    result = orb.calculate(ohlc)
    print(result)
    fig = orb.plot(ohlc, result)
    fig.show()


@pytest.mark.asyncio
async def test_supply_demand_zones():
    pass
    # bot = SMCBotBase(symbols=[symbol], main_tf="1h", market_type="futures")
    # ohlc = await bot.fetch_ibkr_ohlc_data(symbol, "1h", num_candles=200)

    pathlib = "/project/daily_trading/bots/smc_bot/tests/ohlc_data/mnq_1h.csv"
    ohlc = load_data(
        pathlib,
        start_ts="2025-02-01 00:30:00+00:00",
        end_ts="2025-02-12 15:00:00+00:00"
    )
    # Tested_V12_MNQ_NY_sw12_int4_off1_atr2_2_thr0_3_rr1_5_results_copy
    detector = SupplyDemandZonesDetector(fvg_based_only=True,
                                         fvg_offset=10,
                                         swing_length=50,
                                         internal_length=5,
                                         zone_precision_factor=1,
                                         fvg_threshold=0.1,
                                         internal_confluence_filter=True)

    #
    # ohlc_before = ohlc.loc[ohlc.index < cutoff]
    # logger.info(ohlc_before.tail(100).to_string())

    active_swing, active_internal = detector.calculate(ohlc)
    logger.info(f"Active Swing Zones: {active_swing}, Active Internal Zones: {active_internal}")
    fig = detector.plot(ohlc)
    fig.show()


@pytest.mark.asyncio
async def test_bos_detector():
    pass
    # bot = SMCBotBase(symbols=[symbol], main_tf="1h", market_type="futures")
    # ohlc = await bot.fetch_ibkr_ohlc_data(symbol, "1h", num_candles=200)
    #
    # pathlib = "/project/daily_trading/bots/smc_bot/tests/ohlc_data/mbt_1h.csv"
    # ohlc = load_data(
    #     pathlib,
    #     start_ts="2025-10-07 00:00:00+00:00",
    #     end_ts="2025-10-09 15:30:00+00:00"
    # )
    client = IBKRClient(host=HOST, port=PORT, client_id=CLIENT_ID)

    # 1. CONNECT
    await client.connect()
    await asyncio.sleep(2)
    ohlc = await client.get_klines("MGC", "5m")

    fvg_detector = FvgDetector(main_tf=main_tf, pivot_period=5, fvg_threshold=0.3)
    fvg_detector.plot_tail = 500
    fvg_results = fvg_detector.calculate_fvg(ohlc, plot=True)
    fig = fvg_results["fig"]
    fig.show()

    # detector = BosChochDetector(main_tf, show_bull=True, show_bear=True, left_bars=8, right_bars=3)
    # detector.plot_tail = 300
    # detector.get_current_trend(ohlc)
    # results = detector.calculate_bos_choch(ohlc, plot=True)
    # logger.info(results["signals"])
    # fig = results["fig"]
    # fig.show()


def test_csv_merge():
    ohlc_dir = f"{PROJECT_DIR}/bots/smc_bot/tests/ohlc_data"
    symbol = "mbt"
    files = [f"{ohlc_dir}/{symbol}.csv"]
    for i in range(1, 10):
        if os.path.exists(f"{ohlc_dir}/{symbol}_{i}.csv"):
            files.append(f"{ohlc_dir}/{symbol}_{i}.csv")
    merge_csv_no_dups(files, out_file=f"{ohlc_dir}/{symbol}.csv")


@pytest.mark.asyncio
async def test_summary():
    client = IBKRClient(host=HOST, port=PORT, client_id=CLIENT_ID)

    # 1. CONNECT
    await client.connect()
    await asyncio.sleep(2)
    assert client.ib.isConnected(), "❌ Failed to connect"
    logger.info("✅ Connected to IBKR")

    summary = await client.get_daily_summary()
    print(summary)
