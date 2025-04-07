from datetime import datetime, timedelta, timezone

import pandas as pd

from settings import logger, PROJECT_DIR
from src.brokers.BitUnixClient import BitUnixClient


class FuturesFilter:
    def __init__(self):
        self.cl = BitUnixClient()

    def get_symbols(self):
        data = self.cl.get_trading_pairs()
        pairs = [asset["symbol"] for asset in data]
        return pairs

    def get_high_volume_symbols(self, pairs: [str], threshold=90000):
        high_volume_symbols = []
        tickers = self.cl.get_tickers(pairs)
        for ticker in tickers:
            if float(ticker["quoteVol"]) >= threshold:
                high_volume_symbols.append(ticker["symbol"])
        return high_volume_symbols

    def filter_futures(self, pairs):
        filtered_assets = []

        logger.info("\nAnalyzing breakouts for top assets...")
        for symbol in pairs:
            # Get 2 daily candles - this should give us today's (incomplete) and yesterday's (complete) candles
            daily_data = self.cl.get_klines(symbol=symbol, interval="1d", limit=2)

            if not daily_data or len(daily_data) < 2:
                logger.info(f"{symbol} Does Not Have enough Day Data - New Coin??")
                continue

            # The candle at index 0 should be yesterday's completed candle
            prev_day_data = daily_data[-1]
            prev_day_high, prev_day_low = float(prev_day_data["high"]), float(prev_day_data["low"])
            prev_day_timestamp = int(prev_day_data["time"])

            # Verify that the data is actually from the previous day
            now = datetime.now(timezone.utc)
            today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_midnight = today_midnight - timedelta(days=1)

            # Convert to millisecond timestamps
            yesterday_timestamp = int(yesterday_midnight.timestamp() * 1000)
            today_timestamp = int(today_midnight.timestamp() * 1000)

            # Check if the timestamp is from yesterday
            if not (yesterday_timestamp <= prev_day_timestamp < today_timestamp):
                yesterday_date = datetime.fromtimestamp(yesterday_timestamp / 1000, tz=timezone.utc).strftime(
                    '%Y-%m-%d')
                data_date = datetime.fromtimestamp(prev_day_timestamp / 1000, tz=timezone.utc).strftime('%Y-%m-%d')
                error_msg = f"Error: {symbol} data is from {data_date}, expected previous day ({yesterday_date})"

                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.debug(f"{symbol} prev day data: High={prev_day_high}, Low={prev_day_low}, Time={prev_day_timestamp}")

            start = today_midnight.timestamp() * 1000
            first_five_min_data = self.cl.get_klines(symbol=symbol, interval="5m", limit=1, end_time=int(start))[0]
            first_five_min_high = float(first_five_min_data["high"])
            first_five_min_low = float(first_five_min_data["low"])

            last_min_data = self.cl.get_klines(symbol=symbol, interval="1m", limit=1)[0]
            last_min_close = float(last_min_data["close"])

            # Check if five-minute candle is within the range of last day
            if prev_day_low < first_five_min_low and first_five_min_high < prev_day_high:
                asset_data = {
                    "asset": symbol,
                    "prev_day_high": prev_day_high,
                    "prev_day_low": prev_day_low,
                    "first_five_min_low": first_five_min_low,
                    "first_five_min_high": first_five_min_high,
                    "in_range": True,
                    "candidate": False
                }

            else:
                # The five_min levels are not in range
                asset_data = {
                    "asset": symbol,
                    "prev_day_high": prev_day_high,
                    "prev_day_low": prev_day_low,
                    "first_five_min_low": first_five_min_low,
                    "first_five_min_high": first_five_min_high,
                    "in_range": False,
                    "candidate": True
                }
                logger.info(f"{symbol} Not In Range.")

            # Determine the distance between last 1m close price and day/5min levels
            last_min_close_distance_pct_low_five = abs(last_min_close - first_five_min_low) / first_five_min_low * 100
            last_min_close_distance_pct_high_five = abs(
                last_min_close - first_five_min_high) / first_five_min_high * 100

            if last_min_close_distance_pct_high_five < last_min_close_distance_pct_low_five:
                asset_data["curr_price_distance_pct"] = f"{last_min_close_distance_pct_high_five}"

            else:
                asset_data["curr_price_distance_pct"] = f"{last_min_close_distance_pct_low_five}"

            # Determine the distance between five_min and prev_day levels
            low_distance_pct = abs(first_five_min_low - prev_day_low) / prev_day_low * 100
            high_distance_pct = abs(first_five_min_high - prev_day_high) / prev_day_high * 100

            asset_data["five_min_vs_prev_day_low_distance_pct"] = low_distance_pct
            asset_data["five_min_vs_prev_day_high_distance_pct"] = high_distance_pct

            # Check other criteria
            last_min_close_distance_pct_low_day = abs(last_min_close - prev_day_low) / prev_day_low * 100
            last_min_close_distance_pct_high_day = abs(last_min_close - prev_day_high) / prev_day_high * 100
            if last_min_close > prev_day_low and last_min_close_distance_pct_low_day < 0.5:
                asset_data["status"] = "✅near-prev-day-low"
                asset_data["candidate"] = True
                logger.info(f"✅near-prev-day-low - {symbol} ")
            elif last_min_close < prev_day_high and last_min_close_distance_pct_high_day < 0.5:
                asset_data["status"] = "✅near-prev-day-high"
                asset_data["candidate"] = True
                logger.info(f"✅near-prev-day-high - {symbol} ")

            elif last_min_close < prev_day_low:
                asset_data["status"] = " ❌below-prev-day-low"
                asset_data["candidate"] = False

            elif last_min_close > prev_day_high:
                asset_data["status"] = " ❌above-prev-day-high"
                asset_data["candidate"] = False

            elif low_distance_pct >= 4 and last_min_close_distance_pct_low_five <= 0.5:
                asset_data["status"] = "✅near-five-min-low"
                asset_data["candidate"] = True
                logger.info(f"✅near-five-min-low - {symbol} ")

            elif high_distance_pct >= 4 and last_min_close_distance_pct_high_five <= 0.5:
                asset_data["status"] = "✅near-five-min-high"
                asset_data["candidate"] = True
                logger.info(f"✅near-five-min-high - {symbol} ")

            else:
                asset_data["status"] = "❌"
                asset_data["candidate"] = False

            filtered_assets.append(asset_data)

        # Convert to DataFrame for better visualization
        if filtered_assets:
            df = pd.DataFrame(filtered_assets)

            # Sort the DataFrame to have in_range: False at the top
            df = df.sort_values(by="candidate", ascending=False)

            # Add a column with numeration
            df.insert(0, 'num', range(1, len(df) + 1))

            # Define the exact column order, with status after candidate
            column_order = ['num', 'asset', 'in_range', 'candidate', 'status', 'curr_price_distance_pct',
                            'prev_day_high', 'prev_day_low', 'first_five_min_high','first_five_min_low',
                            'five_min_vs_prev_day_low_distance_pct',
                            'five_min_vs_prev_day_high_distance_pct']

            # Reorder the DataFrame columns
            df = df[column_order]

            logger.info("\nFiltered Assets Summary:")

            # Save to CSV with the specific column format
            filename = PROJECT_DIR + f"/artifacts/filtered_futures/filtered_futures.csv"
            df.to_csv(filename, index=False)
            logger.info(f"\nResults saved to {filename}")

            return df
        else:
            logger.info("No assets met the criteria")
            return None


if __name__ == "__main__":
    logger.info("Starting Crypto Futures Filter...")
    cl = FuturesFilter()
    symbols = cl.get_symbols()
    high_volume_symbols = cl.get_high_volume_symbols(symbols, threshold=30000000)
    cl.filter_futures(high_volume_symbols)
