from datetime import datetime, timedelta

from daily_trading.settings import logger
from daily_trading.brokers.BitUnixClient import BitUnixClient


def date_to_unix_timestamp(date_time: datetime):
    """
    Convert a date string in format 'YYYY/MM/DD' to Unix timestamp in milliseconds.
    """
    try:
        # Parse the date string
        dt = date_time.replace(hour=0, minute=0, second=0, microsecond=0)
        timestamp_seconds = int(dt.timestamp())
        timestamp_ms = timestamp_seconds * 1000
        return timestamp_ms
    except ValueError as e:
        raise ValueError(f"Invalid date format. Please use 'YYYY/MM/DD' format. Error: {e}")


if __name__ == "__main__":
    logger.info("Fetching trade history...")
    cl = BitUnixClient()
    start_date = datetime(2025, 8, 28)
    end_date = datetime(2025, 8, 30)

    all_trades = []
    # Loop through March in 3-day increments
    current_start = start_date
    while current_start < end_date:
        current_end = current_start + timedelta(days=3)
        if current_end > end_date:
            current_end = end_date

        logger.info(f"Fetching trades from {current_start.strftime('%Y/%m/%d')} to {current_end.strftime('%Y/%m/%d')}")
        params = {"startTime": date_to_unix_timestamp(current_start), "endTime": date_to_unix_timestamp(current_end)}
        trades = cl.get_history_orders(query_params=params)

        if trades:
            logger.info(f"Found {len(trades)} trades in this range.")
            all_trades.extend(trades)
        else:
            logger.warning("No trades found in this range.")

        current_start = current_end

    if all_trades:
        print(f"Total trades found.")
        cl.save_trades_history_to_csv(all_trades)
    else:
        logger.warning("No trades found for March.")
