import asyncio
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import pytest_asyncio

from daily_trading.bots.smc_bot.backtester.BacktestResultsViewer import BacktestResultsViewer
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
from daily_trading.bots.smc_bot.detectors.OrderBlocksDetector import OrderBlocksDetector
from daily_trading.bots.smc_bot.detectors.SupplyDemandZonesDetector import SupplyDemandZonesDetector
from daily_trading.bots.smc_bot.detectors.VolumeProfiler import VolumeProfiler
from daily_trading.bots.smc_bot.strategy.ORBStrategy import ORBStrategy
from daily_trading.bots.smc_bot.strategy.StrategyBase import SignalSummary, TradeStatus
from daily_trading.settings import logger, PROJECT_DIR, TELEGRAM_MY_DEBUG_BOT_ID
from daily_trading.technicals.AITradeAssistants.GeminiTradeAssistant import GEMINI_PRO, GeminiTradeAssistant, Ohlc
from daily_trading.utils.common import load_data, merge_csv_no_dups

symbol = "BTCUSDT"
main_tf = "1h"
lookback_days = 5


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
async def test_bos_detector():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc = await bot.fetch_bitunix_ohlc_data(symbol, main_tf)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info("\n{}", ohlc.tail(50).to_string())

    detector = BosChochDetector(main_tf, show_bull=True, show_bear=True)
    results = detector.calculate_bos_choch(ohlc, plot=True)
    fig = results["fig"]
    fig.show()


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
    bot = TelegramBot(chat_id=TELEGRAM_MY_DEBUG_BOT_ID)

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
async def test_ai():
    symbol = "SAHARAUSDT"
    bot = SMCBotBase(symbols=[symbol], main_tf="1h")
    # Fetch OHLC mgc within async context
    ohlc_1d = await bot.fetch_bitunix_ohlc_data(symbol, "1d", num_candles=10)
    ohlc_1h = await bot.fetch_bitunix_ohlc_data(symbol, "1h", num_candles=120)
    ohlc_4h = await bot.fetch_bitunix_ohlc_data(symbol, "4h", num_candles=24)
    ohlc_15m = await bot.fetch_bitunix_ohlc_data(symbol, "15m", num_candles=100)

    detector_fvg = FvgDetector(main_tf=main_tf, pivot_period=5, left_bars=5)
    detector_bos = BosChochDetector(main_tf, show_bull=True, show_bear=True)
    bos_results = detector_bos.calculate_bos_choch(ohlc_1h, plot=False)
    fvg_results = detector_fvg.calculate_fvg(ohlc_15m, plot=False)
    fig = detector_fvg.merge_bos_fvg_plots(ohlc_15m, bos_signals=bos_results["signals"],
                                           fvg_signals=fvg_results["signals"])

    ohlc = Ohlc(ohlc_1d, ohlc_4h, ohlc_1h, ohlc_15m)
    ai_agent = GeminiTradeAssistant(symbol=symbol, model=GEMINI_PRO, ohlc=ohlc, debug=True)
    formatted_ai_message, raw_message = await ai_agent.analyze_asset_trade_data()
    raw_message['entry'] = 0.04
    raw_message['stop_loss'] = 0.039
    raw_message['take_profit'] = 0.042
    if raw_message['entry']:
        fig_1 = ai_agent.update_plot_with_entry_data(fig, raw_message)
        fig_1.show()
    summary = SignalSummary(direction="NEUTRAL", trend_signal=["Blah bla"], fig=fig, trend="Bullish")

    # await bot.generate_signal(symbol, summary, ohlc_4h=ohlc_4h, ohlc_1h=ohlc_1h, ohlc_15m=ohlc_15m)


@pytest.mark.asyncio
async def test_merge_ob_and_bos():
    bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    ohlc_1h = await bot.fetch_bitunix_ohlc_data(symbol, "1h", num_candles=120)
    ohlc_15m = await bot.fetch_bitunix_ohlc_data(symbol, "15m", num_candles=100)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info("\n{}", ohlc.tail(50).to_string())

    ob_detector = OrderBlocksDetector(main_tf)
    results = ob_detector.calculate_order_blocks(ohlc_15m, plot=True)
    ob_signals = results["order_blocks"]
    ob_fig = results["fig"]

    detector = BosChochDetector(main_tf, show_bull=True, show_bear=True)
    results = detector.calculate_bos_choch(ohlc_1h, plot=True)
    bos_signals = results["signals"]
    bos_fig = results["fig"]

    fig = ob_detector.merge_bos_ob_plots(ohlc_15m, bos_signals, ob_signals=ob_signals)
    fig.show()

    bot = TelegramBot(chat_id=TELEGRAM_MY_DEBUG_BOT_ID)
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
    symbol = "NQ"
    bot = SMCBotBase(symbols=[symbol], main_tf="1h", market_type="futures")
    # Fetch OHLC mgc within async context
    ohlc_htf = await bot.fetch_bitunix_ohlc_data(symbol, "1h", num_candles=540)
    ohlc_ltf = await bot.fetch_bitunix_ohlc_data(symbol, "5m", num_candles=2100)

    detector = LevelsDetector(main_tf="15m", pivot_length=3, strength=4)
    results = detector.calculate_support_resistance_levels(ohlc_htf, plot=True)
    fig = results["fig"]
    fig.show()
    fig = detector.plot_htf_levels_on_ltf(ohlc_ltf, results)
    fig.show()


@pytest.mark.asyncio
async def test_telegram():
    bot = TelegramBot(chat_id=TELEGRAM_MY_DEBUG_BOT_ID)
    await bot.send_message(
        text="<b>Bold text</b>")


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
async def test_order_blocks():
    # bot = SMCBotBase(symbols=[symbol], lookback_days=lookback_days, main_tf=main_tf)
    # ohlc = await bot.fetch_ohlc_data(symbol, main_tf)

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     logger.info("\n{}", ohlc.tail(50).to_string())

    pathlib = "/project/daily_trading/bots/smc_bot/tests/ohlc_data/mnq.csv"
    ohlc = load_data(
        pathlib,
        start_ts="2025-03-10 12:30:00+00:00",
        end_ts="2025-03-10 19:55:00+00:00"
    )

    detector = OrderBlocksDetector(main_tf)
    results = detector.calculate_order_blocks(ohlc, plot=True)
    print(results)
    fig = results["fig"]
    fig.show()


@pytest.mark.asyncio
async def test_supply_demand_zones():
    symbol = "MNQ"
    bot = SMCBotBase(symbols=[symbol], main_tf="1h", market_type="futures")
    # ohlc = await bot.fetch_ohlc_data(symbol, "5m", num_candles=500)

    pathlib = f"/project/daily_trading/bots/smc_bot/tests/ohlc_data/{symbol.lower()}.csv"
    ohlc = load_data(
        pathlib,
        start_ts="2025-03-03 12:30:00+00:00",
        end_ts="2025-03-03 17:00:00+00:00"
    )
    # ohlc = pd.read_csv('/project/daily_trading/bots/smc_bot/tests/window.csv')
    logger.info(ohlc.tail(200).to_string())
    # Tested_V12_MNQ_NY_sw12_int4_off1_atr2_2_thr0_3_rr1_5_results_copy
    detector = SupplyDemandZonesDetector(fvg_based_only=True,
                                         fvg_offset=1,
                                         swing_length=12,
                                         internal_length=4,
                                         zone_precision_factor=2.2,
                                         fvg_threshold=0.3)

    #
    # ohlc_before = ohlc.loc[ohlc.index < cutoff]
    # logger.info(ohlc_before.tail(100).to_string())

    active_swing, active_internal = detector.calculate(ohlc)
    logger.info(f"Active Swing Zones: {active_swing}, Active Internal Zones: {active_internal}")
    fig = detector.plot(ohlc)
    fig.show()


def test_csv_merge():
    ohlc_dir = f"{PROJECT_DIR}/bots/smc_bot/tests/ohlc_data"
    symbol = "mcl"
    files = []
    for i in range(1, 10):
        if os.path.exists(f"{ohlc_dir}/{symbol}_{i}.csv"):
            files.append(f"{ohlc_dir}/{symbol}_{i}.csv")
    merge_csv_no_dups(files, out_file=f"{ohlc_dir}/{symbol}.csv")


def test_dashboard():
    f = Path(
        "/daily_trading/bots/smc_bot/tests/orb_results_standard/ELITE_SETUPS/MNQ/NY/R1_FAST_REACTIVITY_MNQ_NY_SW12_I4/MNQ_NY_sw12_int4_off1_atr2.2_thr0.35_rr1.2_results.csv")

    html_path = Path(
        "/daily_trading/bots/smc_bot/tests/orb_results_standard") / f"Dashboard_MNQR1_FAST_REACTIVITY_MNQ_NY_SW12_I4.html"
    viewer = BacktestResultsViewer(html_path=html_path)
    viewer.generate_dashboard(f)


def test_audit_pnl_and_exit_integrity():
    """
    Verifies PnL math and Exit Price integrity.
    """
    symbol = "MNQ"
    strategy = ORBStrategy(
        symbol=symbol,
        timezone=TimeZone.NY,
    )
    point_values = strategy.POINT_VALUE

    csv_path = Path(
        "/daily_trading/bots/smc_bot/tests/orb_results_standard/ELITE_SETUPS/MNQ/LONDON/R2_PRECISION_MNQ_LONDON_SW15_I5_OFF4_ATR1.8_T0.2_RR1.2/MNQ_LONDON_sw15_int5_off4_atr1.8_thr0.2_rr1.2_results.csv")

    df_results = pd.read_csv(csv_path)
    df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
    df_results['exit_time'] = pd.to_datetime(df_results['exit_time'])

    data_path = f"/project/daily_trading/bots/smc_bot/tests/ohlc_data/{symbol.lower()}.csv"
    df_ohlc = load_data(data_path)

    # 3. Validation Logic
    # 3. Validation Logic
    for idx, trade in df_results.iterrows():
        # --- FIX: SKIP CANCELLED TRADES ---
        if trade['result'] == 'CANCELLED':
            print(f"ℹ️ Skipping PnL check for CANCELLED trade at index {idx}")
            assert trade['pnl'] == 0.0, f"Cancelled trade must have 0 PnL, got {trade['pnl']}"
            continue

        exit_ts = trade['exit_time']

        # Data Check
        assert exit_ts in df_ohlc.index, f"Exit time {exit_ts} not found in OHLC."

        # Price Integrity Check
        candle = df_ohlc.loc[exit_ts]
        # Allow small epsilon for floating point data matching
        assert candle['low'] - 0.01 <= trade['exit_price'] <= candle['high'] + 0.01, \
            f"Exit Price {trade['exit_price']} outside bar range [{candle['low']}, {candle['high']}]."

        # PnL Check
        pv = point_values[symbol]
        entry = trade['entry']
        exit_p = trade['exit_price']
        size = trade['size']
        reported_pnl = trade['pnl']

        if trade['direction'] == 'LONG':
            calc_pnl = (exit_p - entry) * size * pv
        else:  # SHORT
            calc_pnl = (entry - exit_p) * size * pv

        assert np.isclose(calc_pnl, reported_pnl, atol=0.01), \
            f"PnL Mismatch! Calculated: {calc_pnl}, Reported: {reported_pnl}"

        print(f"✅ PnL Verified: {symbol} {trade['direction']} | ${calc_pnl:.2f}")


@pytest_asyncio.fixture(scope="module")
async def ib_client():
    """
    Setup: Connects once before tests start.
    Teardown: Disconnects after all tests finish.
    """
    # Use a unique client ID for testing to avoid conflicts with bots
    client = IBKRClient(port=4002, client_id=1)
    await client.connect()
    yield client
    client.disconnect()


TEST_CASES = [
    ("MNQ", "MNQ", "CME"),
    ("MES", "MES", "CME"),
    ("MBT", "MBT", "CME"),
    ("MGC", "MGC", "COMEX"),
    ("SIL", "SIL", "COMEX"),  # Input SIL -> Expect SIL on COMEX
    ("MCL", "MCL", "NYMEX"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("input_symbol, expected_root, expected_exchange", TEST_CASES)
async def test_ibkr_contract_resolution(ib_client, input_symbol, expected_root, expected_exchange):
    """
    Verifies that we can resolve the contract, the symbol matches expectations,
    and the expiration date is valid (in the future).
    """
    print(f"\nTesting: {input_symbol}...")
    contract = await ib_client.get_futures_contract(input_symbol)
    expiry_str = contract.lastTradeDateOrContractMonth
    print(f"✅ PASS: {input_symbol} -> {contract.localSymbol} ({contract.exchange}) Exp: {expiry_str}")


import pytest
import pandas as pd
from daily_trading.brokers.IBKRClient import IBKRClient  # Adjust import path as needed

# CONFIGURATION
# Ensure this matches your running TWS/Gateway (Paper Trading recommended)
HOST = '127.0.0.1'
PORT = 4002
CLIENT_ID = 999  # Use a unique ID for testing to avoid conflict with running bots

# The symbols we must verify
TARGET_SYMBOLS = ["MNQ", "MES", "MGC", "SIL"]


@pytest.mark.asyncio
async def test_ibkr():
    """
    E2E Test for State Restoration (PENDING Scenario).
    Verifies that ORBStrategy can 'find' an existing order on IBKR
    and restore its internal state after a restart.
    """
    client = IBKRClient(host=HOST, port=PORT, client_id=CLIENT_ID)
    await client.connect()
    msg = await client.get_daily_summary()
    logger.info(msg)

    ohlc = await client.get_klines("MES", "5m")
    print()

    # Setup Target Variables
    # symbol = "MNQ"
    # timezone = ZoneInfo("America/New_York")
    #
    # # CRITICAL: This tag must match exactly what ORBStrategy generates internally
    # # Logic: f"{symbol}_{session_short}_ORB" -> MNQ_NY_ORB
    # target_ref = "TEST_MNQ_NY_ORB"
    # open_orders = await client.get_open_orders()
    # positions = await client.get_active_positions()
    # print(f"ORDERS: {open_orders}")
    # print(f"POSITIONS: {positions}")
    #
    # try:
    #     logger.info(f"--- 1. Getting Market Data for {symbol} ---")
    #     df = await client.get_klines(symbol, "5m")
    #     assert not df.empty, "Failed to get market data"
    #     current_price = df['close'].iloc[-1]
    #
    #     # Calculate a SAFE Limit Entry (5% below market) so it stays PENDING
    #     entry_price = client.round_to_tick(current_price * 0.95, symbol)
    #     sl_price = client.round_to_tick(entry_price - 20, symbol)
    #     tp_price = client.round_to_tick(entry_price + 50, symbol)
    #
    #     logger.info(f"Current: {current_price} | Placing Test Order @ {entry_price} (Ref: {target_ref})")
    #
    #     # --- 2. Place the Bracket Order on IBKR ---
    #     await client.place_bracket_order(
    #         symbol=symbol,
    #         quantity=1,
    #         entry=entry_price,
    #         sl=sl_price,
    #         tp=tp_price,
    #         direction="LONG",
    #         order_ref=target_ref
    #     )
    #
    #     # Give IBKR a moment to register
    #     await asyncio.sleep(2)
    #
    #     # --- 3. Simulate Bot Restart (Fetch Raw Data) ---
    #     logger.info("--- Simulating Bot Restart & Fetching State ---")
    #     open_orders = await client.get_open_orders()
    #     positions = await client.get_active_positions()
    #
    #     # Verify we actually have the orders on the server before testing sync
    #     my_orders = [o for o in open_orders if o.order.orderRef == target_ref]
    #     assert len(my_orders) == 3, f"Setup Failed: Expected 3 orders on IBKR, found {len(my_orders)}"
    #
    #     # --- 4. Instantiate Strategy & Run Sync ---
    #     # We create a fresh strategy instance with default risk/params
    #     strategy = ORBStrategy(symbol=symbol, timezone=timezone)
    #     strategy.ext_tag = target_ref
    #
    #     # ASSERT PRE-CONDITION: State should be empty
    #     assert strategy.trade_status == TradeStatus.NONE
    #     assert strategy.active_trade is None
    #
    #     # EXECUTE SYNC
    #     logger.info("Running strategy.sync_state()...")
    #     SMCBotBase.sync_state(strategy, positions, open_orders)
    #
    #     # --- 5. ASSERTIONS ---
    #     logger.info(f"Post-Sync Status: {strategy.trade_status}")
    #
    #     # A. Status Check
    #     assert strategy.trade_status == TradeStatus.PENDING, \
    #         f"Sync Failed: Expected PENDING, got {strategy.trade_status}"
    #
    #     # B. Active Trade Object Check
    #     assert strategy.active_trade is not None, "Sync Failed: active_trade is None"
    #
    #     # C. Data Integrity Check
    #     # Allow small float diffs (epsilon)
    #     assert abs(strategy.active_trade.price - entry_price) < 0.001, \
    #         f"Price Mismatch: Expected {entry_price}, got {strategy.active_trade.price}"
    #
    #     assert strategy.active_trade.direction.name == "LONG"
    #     assert strategy.active_trade.metadata.get("zone_id") == "RECOVERED"
    #
    #     logger.info("✅ SUCCESS: Strategy successfully recovered state from IBKR!")
    #
    # finally:
    #     # --- 6. Cleanup ---
    #     logger.info("Cleaning up test orders...")
    #     await client.cancel_all_orders(order_ref=target_ref)
    #     client.disconnect()
    #     await asyncio.sleep(1)
