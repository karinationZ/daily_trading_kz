import asyncio
import csv
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from daily_trading.brokers.BitUnixClient import BitUnixClient
from daily_trading.settings import logger, PROJECT_DIR


class BitUnixDivergentAssetsFinder:
    def __init__(self):
        self.client = BitUnixClient()  # Assume async-capable client

    def get_high_volume_symbols(self, threshold=30000000):
        """Find trading pairs with volume above threshold (30M)"""
        all_pairs = self.client.get_trading_pairs()  # Assume async
        all_symbols = [pair["symbol"] for pair in all_pairs]

        # Batch fetch tickers
        tickers = self.client.get_tickers(all_symbols)  # Assume async
        high_volume_symbols = [
            (ticker["symbol"], float(ticker["quoteVol"]))
            for ticker in tickers if float(ticker["quoteVol"]) >= threshold
        ]
        high_volume_symbols.sort(key=lambda x: x[1])
        logger.debug(f"High volume symbols: {len(high_volume_symbols)}")
        return high_volume_symbols

    @staticmethod
    def format_klines_to_dataframe(klines):
        """Convert kline mgc to pandas DataFrame"""
        df = pd.DataFrame(klines, columns=["open", "high", "low", "close", "quoteVol", "baseVol", "time"])
        df[["open", "high", "low", "close", "quoteVol", "time"]] = df[
            ["open", "high", "low", "close", "quoteVol", "time"]].apply(pd.to_numeric)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        return df.set_index("time")[["close"]]  # Only keep 'close' for correlation

    async def fetch_kline_data(self, symbol, interval="5m", lookback_hours=24):
        """Fetch historical kline mgc for a symbol"""
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)

        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=200
        )
        return self.format_klines_to_dataframe(klines)

    @staticmethod
    def calculate_correlation(btc_df, symbol_df):
        """Calculate correlation coefficient between BTC and another symbol"""
        common_index = btc_df.index.intersection(symbol_df.index)
        if len(common_index) < 10:
            return None

        btc_returns = btc_df.loc[common_index, "close"].pct_change().dropna()
        symbol_returns = symbol_df.loc[common_index, "close"].pct_change().dropna()

        if len(btc_returns) < 10 or btc_returns.abs().max() > 1e10 or symbol_returns.abs().max() > 1e10:
            return None

        correlation, _ = pearsonr(btc_returns, symbol_returns)
        return correlation if not np.isnan(correlation) else None

    async def find_non_correlated_assets(self, correlation_threshold=0.2, volume_threshold=30000000, lookback_hours=24):
        """Find assets with weak correlation to BTC"""
        high_volume_symbols = self.get_high_volume_symbols(threshold=volume_threshold)
        logger.info(f"Found {len(high_volume_symbols)} high volume symbols")

        btc_df = await self.fetch_kline_data("BTCUSDT", lookback_hours=lookback_hours)

        # Fetch kline mgc concurrently
        tasks = [
            self.fetch_kline_data(symbol, lookback_hours=lookback_hours)
            for symbol, _ in high_volume_symbols if symbol != "BTCUSDT"
        ]
        symbol_dfs = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for (symbol, volume), symbol_df in zip(
                [(s, v) for s, v in high_volume_symbols if s != "BTCUSDT"], symbol_dfs
        ):
            if isinstance(symbol_df, Exception):
                logger.error(f"Error fetching {symbol}: {symbol_df}")
                continue
            correlation = self.calculate_correlation(btc_df, symbol_df)
            if correlation is not None:
                results.append({
                    "symbol": symbol,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation),
                    "volume": volume
                })

        non_correlated_assets = sorted(
            [r for r in results if abs(r["correlation"]) < correlation_threshold],
            key=lambda x: x["abs_correlation"]
        )
        return non_correlated_assets

    async def find_divergent_assets(self, volume_threshold=30000000, lookback_hours=24, save_results=False):
        """Find assets with weak or inverse correlation to BTC"""
        high_volume_symbols = self.get_high_volume_symbols(threshold=volume_threshold)
        logger.info(f"Found {len(high_volume_symbols)} high volume symbols")

        btc_df = await self.fetch_kline_data("BTCUSDT", lookback_hours=lookback_hours)

        # Fetch kline mgc concurrently
        tasks = [
            self.fetch_kline_data(symbol, lookback_hours=lookback_hours)
            for symbol, _ in high_volume_symbols if symbol != "BTCUSDT"
        ]
        symbol_dfs = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for (symbol, volume), symbol_df in zip(
                [(s, v) for s, v in high_volume_symbols if s != "BTCUSDT"], symbol_dfs
        ):
            if isinstance(symbol_df, Exception):
                logger.error(f"Error fetching {symbol}: {symbol_df}")
                continue
            correlation = self.calculate_correlation(btc_df, symbol_df)
            if correlation is not None:
                results.append({
                    "symbol": symbol,
                    "correlation": correlation,
                    "volume": volume
                })

        results.sort(key=lambda x: x["correlation"])
        non_correlated = [r for r in results if abs(r["correlation"]) < 0.2]
        inverse_correlated = [r for r in results if r["correlation"] < -0.7]

        output = {
            "non_correlated": non_correlated,
            "inverse_correlated": inverse_correlated,
            "all_results": results
        }

        if save_results:
            self.save_results_to_csv(output)
        return output

    @staticmethod
    def save_results_to_csv(results, filename="artifacts/filtered_futures/divergent_assets.csv"):
        """Save divergent assets results to CSV file"""
        full_path = os.path.join(PROJECT_DIR, filename)
        with open(full_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "correlation", "volume"])
            for asset in results["non_correlated"]:
                writer.writerow([asset["symbol"], f"{asset['correlation']:.4f}", asset["volume"]])
            for asset in results["inverse_correlated"]:
                writer.writerow([asset["symbol"], f"{asset['correlation']:.4f}", "inverse_correlated"])
        logger.debug(f"Results saved to {full_path}")
        return full_path


if __name__ == "__main__":
    logger.debug("Searching for assets that move most independently of Bitcoin.")
    finder = BitUnixDivergentAssetsFinder()

    # Run async method
    results = asyncio.run(finder.find_divergent_assets(
        volume_threshold=10000000,
        lookback_hours=3
    ))

    # Log results
    logger.info("\nNon-correlated assets (correlation between -0.2 and 0.2):")
    logger.info(f"Found {len(results['non_correlated'])} asset(s)")
    for asset in results["non_correlated"]:
        logger.info(f"{asset['symbol']}: Correlation = {asset['correlation']:.3f} | Volume: {asset['volume']:.2f}")

    logger.debug("\nInverse-correlated assets (correlation below -0.7):")
    logger.debug(f"Found {len(results['inverse_correlated'])} inverse_correlated asset(s)")
    for asset in results["inverse_correlated"]:
        logger.debug(f"{asset['symbol']}: Correlation = {asset['correlation']:.3f} | Volume: {asset['volume']:.2f}")
