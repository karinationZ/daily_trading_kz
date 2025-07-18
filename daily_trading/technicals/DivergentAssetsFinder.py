import csv
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from daily_trading.settings import logger, PROJECT_DIR
from daily_trading.brokers.BitUnixClient import BitUnixClient


class BitUnixDivergentAssetsFinder:
    def __init__(self):
        self.client = BitUnixClient()

    def get_high_volume_symbols(self, threshold=30000000):
        """Find trading pairs with volume above the threshold (30M)"""
        high_volume_symbols = []

        # Get all available trading pairs
        all_pairs = self.client.get_trading_pairs()
        all_symbols = [pair["symbol"] for pair in all_pairs]

        # Get tickers for all symbols
        tickers = self.client.get_tickers(all_symbols)
        for ticker in tickers:
            if float(ticker["quoteVol"]) >= threshold:
                high_volume_symbols.append((ticker["symbol"], ticker["quoteVol"]))

        high_volume_symbols.sort(key=lambda x: x[1])
        print(json.dumps(high_volume_symbols, indent=1))
        return high_volume_symbols

    @staticmethod
    def format_klines_to_dataframe(klines):
        """Convert kline data to pandas DataFrame"""
        df = pd.DataFrame(klines, columns=["open", "high", "low", "close", "quoteVol", "baseVol", "time"])

        # Convert string values to numeric (including time column)
        numeric_cols = ["open", "high", "low", "close", "quoteVol", "time"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])

        # Now safely convert timestamp to datetime
        df["time"] = pd.to_datetime(df["time"], unit="ms")

        return df.set_index("time")

    def fetch_kline_data(self, symbol, interval="5m", lookback_hours=24):
        """Fetch historical kline data for a symbol"""
        # Calculate start and end times
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=lookback_hours)).timestamp() * 1000)

        # Get kline data
        klines = self.client.get_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            limit=200  # Maximum allowed by API
        )

        df = self.format_klines_to_dataframe(klines)
        return df

    def calculate_correlation(self, btc_df, symbol_df):
        """Calculate correlation coefficient between BTC and another symbol with error handling"""
        # Make sure timestamps align by reindexing
        common_index = btc_df.index.intersection(symbol_df.index)

        if len(common_index) < 10:  # Need enough data points for reliable correlation
            return None

        btc_returns = btc_df.loc[common_index, "close"].pct_change().dropna()
        symbol_returns = symbol_df.loc[common_index, "close"].pct_change().dropna()

        if len(btc_returns) < 10:  # Additional check after pct_change drops an entry
            return None

        # Check for extreme values that might cause overflow
        if btc_returns.abs().max() > 1e10 or symbol_returns.abs().max() > 1e10:
            print(f"Extreme values detected, skipping correlation calculation")
            return None

        # Calculate Pearson correlation coefficient with error handling

        correlation, _ = pearsonr(btc_returns, symbol_returns)
        return correlation if not np.isnan(correlation) else None

    def find_non_correlated_assets(self, correlation_threshold=0.3, volume_threshold=30000000, lookback_hours=24):
        """
        Find assets that don"t follow BTC price movements
        Returns assets with correlation below the threshold (suggesting weak/no/inverse relation to BTC)
        """
        # 1. Get high volume symbols
        high_volume_symbols = self.get_high_volume_symbols(threshold=volume_threshold)
        print(f"Found {len(high_volume_symbols)} high volume symbols")

        # 2. Get BTC price data
        btc_df = self.fetch_kline_data("BTCUSDT", lookback_hours=lookback_hours)

        # 3. Calculate correlation for each high volume symbol
        results = []

        for symbol, volume in high_volume_symbols:
            if symbol == "BTCUSDT":
                continue  # Skip BTC itself

            symbol_df = self.fetch_kline_data(symbol, lookback_hours=lookback_hours)
            correlation = self.calculate_correlation(btc_df, symbol_df)

            if correlation is not None:
                results.append({
                    "symbol": symbol,
                    "correlation": correlation,
                    "abs_correlation": abs(correlation),
                    "volume": volume
                })

        # 4.Filter and Sort list of assets that move most independently of Bitcoin.
        non_correlated_assets = [r for r in results if abs(r["correlation"]) < correlation_threshold]
        non_correlated_assets.sort(key=lambda x: x["abs_correlation"])

        return non_correlated_assets

    def find_divergent_assets(self, volume_threshold=30000000, lookback_hours=24, save_results=False):
        """
        Find assets that move differently from BTC, considering both weak correlation
        and negative correlation (completely opposite movement)

        Returns:
            - non_correlated: assets with very weak correlation to BTC (-0.3 to 0.3)
            - inverse_correlated: assets with strong negative correlation (< -0.7)
        """
        # 1. Get high volume symbols
        high_volume_symbols = self.get_high_volume_symbols(threshold=volume_threshold)
        print(f"Found {len(high_volume_symbols)} high volume symbols")

        # 2. Get BTC price data
        btc_df = self.fetch_kline_data("BTCUSDT", lookback_hours=lookback_hours)

        # 3. Calculate correlation for each high volume symbol
        results = []

        for symbol, volume in high_volume_symbols:
            if symbol == "BTCUSDT":
                continue  # Skip BTC itself

            try:
                symbol_df = self.fetch_kline_data(symbol, lookback_hours=lookback_hours)
                correlation = self.calculate_correlation(btc_df, symbol_df)

                if correlation is not None:
                    results.append({
                        "symbol": symbol,
                        "correlation": correlation,
                        "volume": volume
                    })
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")

        # 4. Sort results by correlation
        results.sort(key=lambda x: x["correlation"])

        # 5. Categorize results
        non_correlated = [r for r in results if abs(r["correlation"]) < 0.3]
        inverse_correlated = [r for r in results if r["correlation"] < -0.7]
        output = {
            "non_correlated": non_correlated,
            "inverse_correlated": inverse_correlated,
            "all_results": results
        }
        if save_results:
            self.save_results_to_csv(output)
        return output

    def save_results_to_csv(self, results, filename="artifacts/filtered_futures/divergent_assets.csv"):
        """
        Save divergent assets results to CSV file
        Args:
            results: Should be a dictionary with "non_correlated" and "inverse_correlated" keys
                     as returned by find_divergent_assets()
            filename: Output filename (default: divergent_assets.csv)
        """
        # Ensure results is in the expected format
        if not isinstance(results, dict) or "non_correlated" not in results:
            raise ValueError("Results must be a dictionary with 'non_correlated' key")

        # Prepare full file path
        full_path = os.path.join(PROJECT_DIR, filename)

        with open(full_path, "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["symbol", "correlation", "volume"])

            # Write non-correlated assets (-0.3 < corr < 0.3)
            if "non_correlated" in results:
                for asset in results["non_correlated"]:
                    writer.writerow([
                        asset["symbol"],
                        f"{asset['correlation']:.4f}",
                        asset["volume"]
                    ])

            # Write inverse-correlated assets (corr < -0.7)
            if "inverse_correlated" in results:
                for asset in results["inverse_correlated"]:
                    writer.writerow([
                        asset["symbol"],
                        f"{asset['correlation']:.4f}",
                        "inverse_correlated"
                    ])

        logger.info(f"Results saved to {full_path}")
        return full_path


if __name__ == "__main__":
    logger.info("Searching for assets that move most independently of Bitcoin.")
    finder = BitUnixDivergentAssetsFinder()

    results = finder.find_divergent_assets(
        volume_threshold=10000000,
        lookback_hours=2
    )
    # Print results
    logger.info("\nNon-correlated assets (correlation between -0.3 and 0.3):")
    logger.info(f"Found {len(results['non_correlated'])} asset(s)")
    for asset in results["non_correlated"]:
        logger.info(f"{asset['symbol']}: Correlation = {asset['correlation']:.3f} | Volume: {asset['volume']}")

    logger.info("\nInverse-correlated assets (correlation below -0.7):")
    for asset in results["inverse_correlated"]:
        logger.info(f"{asset['symbol']}: Correlation = {asset['correlation']:.3f}")
