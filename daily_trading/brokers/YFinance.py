from datetime import datetime, timedelta

import yfinance as yf


class YFinanceClient:
    """
    Client for fetching futures data from Yahoo Finance
    Mirrors BitUnix API structure for easy drop-in replacement

    FREE - No API token required
    Supports 15m, 1h, 4h intervals for ES, NQ, GC futures
    """

    # Symbol mapping: your symbol -> Yahoo Finance ticker
    SYMBOL_MAP = {
        "ES": "ES=F",  # S&P 500 E-mini futures
        "NQ": "NQ=F",  # Nasdaq-100 E-mini futures
        "GC": "GC=F",  # Gold futures
        "ESUSDT": "ES=F",
        "NQUSDT": "NQ=F",
        "GCUSDT": "GC=F",
    }

    # Interval mapping: BitUnix format -> yfinance format
    INTERVAL_MAP = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d",
        "1w": "1wk",
    }

    def __init__(self):
        """Initialize Yahoo Finance client - no API token needed"""
        pass

    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=100):
        """
        Fetch kline (candlestick) data from Yahoo Finance

        Args:
            symbol (str): Futures symbol, e.g. "ES", "NQ", "GC"
            interval (str): Kline interval such as "5m", "15m", "1h", "4h", "1d" etc.
            start_time (int, optional): Start time in milliseconds (Unix timestamp)
            end_time (int, optional): End time in milliseconds (Unix timestamp)
            limit (int, optional): Number of klines to fetch, default 100
            kline_type (str, optional): Kline type (ignored for Yahoo Finance compatibility)

        Returns:
            dict: Response in BitUnix format containing kline data
        """
        try:
            # Map symbol to Yahoo Finance ticker
            yf_symbol = self.SYMBOL_MAP.get(symbol.upper(), f"{symbol.upper()}=F")

            # Map interval to yfinance format
            yf_interval = self.INTERVAL_MAP.get(interval, interval)

            # Convert timestamps to datetime if provided
            if start_time:
                start_dt = datetime.fromtimestamp(start_time / 1000)
            else:
                # Default: fetch last 3 months for daily, less for intraday
                if interval in ["1m", "5m", "15m", "30m"]:
                    start_dt = datetime.now() - timedelta(days=7)  # Last 7 days for minute data
                elif interval in ["1h", "4h"]:
                    start_dt = datetime.now() - timedelta(days=60)  # Last 60 days for hourly
                else:
                    start_dt = datetime.now() - timedelta(days=90)  # Last 90 days for daily+

            if end_time:
                end_dt = datetime.fromtimestamp(end_time / 1000)
            else:
                end_dt = datetime.now()

            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_dt, end=end_dt, interval=yf_interval)

            # Check if data is empty
            if df.empty:
                return []

            # Limit results (take the most recent ones)
            if len(df) > limit:
                df = df.tail(limit)

            # Convert to BitUnix format
            data = []
            for index, row in df.iterrows():
                # Handle timezone-aware datetime
                if hasattr(index, 'timestamp'):
                    timestamp_ms = int(index.timestamp() * 1000)
                else:
                    timestamp_ms = int(index.value // 10 ** 6)  # Convert numpy datetime64 to ms

                kline = {
                    "open": f"{row['Open']:.2f}",
                    "high": f"{row['High']:.2f}",
                    "low": f"{row['Low']:.2f}",
                    "close": f"{row['Close']:.2f}",
                    "quoteVol": f"{row['Volume']:.3f}",
                    "baseVol": f"{row['Volume'] * row['Close']:.5f}",
                    "time": str(timestamp_ms)
                }
                data.append(kline)

            return data

        except Exception as e:
            return {
                "code": 1,
                "data": [],
                "msg": f"Error: {str(e)}"
            }


# Usage example
if __name__ == "__main__":
    client = YFinanceClient()

    # Example 1: Fetch last 10 15-minute candles for S&P 500
    print("Fetching ES (S&P 500) 15-minute data...")
    result = client.get_klines("ES", "15m", limit=10)
    print(f"Code: {result['code']}, Message: {result['msg']}")
    print(f"Number of candles: {len(result['data'])}")
    if result['data']:
        print(f"Latest candle: {result['data'][-1]}")

    print("\n" + "=" * 50 + "\n")
