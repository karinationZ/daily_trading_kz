import asyncio
from typing import Dict, List, Tuple

import pandas as pd

from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
from daily_trading.bots.smc_bot.strategy.BosOrderBlockTapStrategy import MajorHHLLRejectionStrategy
from daily_trading.settings import *


class BackTester:
    """
    Backtests MajorHHLLRejectionStrategy by simulating live market conditions.
    Processes X days of OHLC data (1H and 15m) candle by candle, generates signals,
    saves them to a CSV file, and creates plots for verification.
    """

    def __init__(self, symbol: str, main_tf: str = "1h", lookback_days: int = 30):
        self.symbol = symbol
        self.main_tf = main_tf
        self.lookback_days = lookback_days
        self.strategy = MajorHHLLRejectionStrategy(
            main_tf, debug=True, telegram_chat_id=TELEGRAM_BOT_CHAT_ID
        )
        self.output_dir = "./backtest_results"
        os.makedirs(self.output_dir, exist_ok=True)
        self.csv_path = f"{self.output_dir}/signals_{symbol.replace('/', '_')}.csv"

    async def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch 30 days of OHLC data for 1H and 15m timeframes.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: OHLC data for 1H, 15m.
        """
        bot = SMCBotBase(
            symbols=[self.symbol],
            lookback_days=self.lookback_days,
            main_tf=self.main_tf
        )
        ohlc_1h = await bot.fetch_ohlc_data(self.symbol, "1h")
        ohlc_15m = await bot.fetch_ohlc_data(self.symbol, "15m")

        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for df, tf in [(ohlc_1h, '1h'), (ohlc_15m, '15m')]:
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Invalid OHLC data for {tf}: missing columns")
                raise ValueError(f"OHLC data for {tf} must contain {required_columns}")
            if df.empty:
                logger.error(f"Empty OHLC data for {tf}")
                raise ValueError(f"No data retrieved for {tf}")

        logger.info(f"Fetched {len(ohlc_1h)} 1H candles, {len(ohlc_15m)} 15m candles.")
        return ohlc_1h, ohlc_15m

    async def simulate_live_market(self, ohlc_1h: pd.DataFrame, ohlc_15m: pd.DataFrame) -> List[Dict]:
        """
        Simulate live market by processing 1H candles sequentially, generating signals.

        Args:
            ohlc_1h (pd.DataFrame): 1H OHLC data for trend and HH/LL detection.
            ohlc_15m (pd.DataFrame): 15m OHLC data for candlestick patterns.

        Returns:
            List[Dict]: List of signal dictionaries with timestamp, symbol, signal, trend, recent_smc_events.
        """
        signals = []
        start_time = ohlc_1h.index[0]
        end_time = ohlc_1h.index[-1]

        logger.info(f"Simulating live market from {start_time} to {end_time}")

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info("\n1H Data:\n{}", ohlc_1h.to_string())
            logger.info("\n15m Data:\n{}", ohlc_15m.to_string())

        for i, current_time in enumerate(ohlc_1h.index[1:], 1):  # Skip first candle
            # Get data up to current time
            ohlc_1h_past = ohlc_1h[ohlc_1h.index <= current_time]
            ohlc_15m_past = ohlc_15m[ohlc_15m.index <= current_time]

            if len(ohlc_1h_past) < 24:  # Ensure enough data for trend detection
                logger.debug(f"Skipping {current_time}: insufficient 1H data ({len(ohlc_1h_past)} candles)")
                continue

            logger.debug(f"Processing candle at {current_time}")
            trade_signal = await self.strategy.analyze_smc(
                symbol=self.symbol,
                ohlc_htf=ohlc_1h_past,
                ohlc_ltf=ohlc_15m_past
            )

            if trade_signal.direction != "NEUTRAL":
                signal_data = {
                    "timestamp": current_time,
                    "symbol": self.symbol,
                    "signal": trade_signal.direction,
                    "trend": trade_signal.trend,
                    "recent_smc_events": trade_signal.recent_smc_events,
                    "fig": trade_signal.fig
                }
                signals.append(signal_data)

                # Save plot if generated
                if trade_signal.fig:
                    plot_path = f"{self.output_dir}/signal_{self.symbol.replace('/', '_')}_{current_time.strftime('%Y%m%d_%H%M%S')}.html"
                    trade_signal.fig.write_html(plot_path)
                    logger.info(f"Plot saved to {plot_path}")

        return signals

    def save_signals_to_csv(self, signals: List[Dict]):
        """
        Save signals to a CSV file.

        Args:
            signals (List[Dict]): List of signal dictionaries.
        """
        if not signals:
            logger.warning("No signals to save to CSV")
            return

        flattened_data = []
        for signal in signals:
            flat_signal = {
                'timestamp': signal['timestamp'],
                'symbol': signal['symbol'],
                'signal': signal['signal'],
                'trend': signal['trend'],
                'recent_smc_events': '; '.join(signal['recent_smc_events'])
            }
            flattened_data.append(flat_signal)

        # Create DataFrame and save to CSV
        df = pd.DataFrame(flattened_data)
        df.to_csv(self.csv_path, index=False)
        logger.info(f"Signals saved to {self.csv_path}")

    async def run(self):
        """
        Run the backtest: fetch data, simulate market, save signals and plots.
        """
        logger.info(f"Starting backtest for {self.symbol}")
        ohlc_1h, ohlc_15m = await self.fetch_data()
        signals = await self.simulate_live_market(ohlc_1h, ohlc_15m)
        self.save_signals_to_csv(signals)
        logger.info(f"Backtest completed. Generated {len(signals)} signals.")

    @staticmethod
    def validate_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLC DataFrame.

        Args:
            df (pd.DataFrame): OHLC DataFrame to validate.

        Returns:
            pd.DataFrame: Validated DataFrame.

        Raises:
            ValueError: If DataFrame is invalid.
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        if df.empty:
            raise ValueError("DataFrame is empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        return df


async def main():
    """
    Example usage of BackTester.
    """
    symbol = "THETAUSDT"
    backtester = BackTester(
        symbol=symbol,
        main_tf="1h",
        lookback_days=15,
    )
    await backtester.run()


if __name__ == "__main__":
    asyncio.run(main())
