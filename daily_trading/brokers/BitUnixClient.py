import csv
import hashlib
import json
import os
import string
import time
from datetime import datetime
from enum import Enum
from random import choices

import dotenv
import requests

from daily_trading.settings import logger, PROJECT_DIR

dotenv.load_dotenv()

BITUNIX_CONFIG = {
    "API_KEY": os.environ.get("BITUNIX_API_KEY"),
    "API_SECRET": os.environ.get("BITUNIX_SECRET_KEY"),
    "PASSWORD": os.environ.get("BITUNIX_PSWRD")
}


class Endpoints(Enum):
    GET_HISTORY_TRADES = "/api/v1/futures/trade/get_history_trades"
    GET_HISTORY_ORDERS = "/api/v1/futures/trade/get_history_orders"
    GET_ORDER_DETAIL = "/api/v1/futures/trade/get_order_detail"
    GET_TICKERS = "/api/v1/futures/market/tickers"
    GET_TRADING_PAIRS = "/api/v1/futures/market/trading_pairs"
    PLACE_ORDER = "/api/v1/futures/trade/place_order"


class BitUnixClient:
    BASE_URL = "https://fapi.bitunix.com"

    def __init__(self):
        self.api_key = BITUNIX_CONFIG["API_KEY"]
        self.secret_key = BITUNIX_CONFIG["API_SECRET"]
        self.nonce = None
        self.headers = {}

    @staticmethod
    def sha256_hex(input_string):
        return hashlib.sha256(input_string.encode('utf-8')).hexdigest()

    @staticmethod
    def generate_random_nonce(length=32):
        """Generate a random nonce of specified length"""
        return ''.join(choices(string.ascii_letters + string.digits, k=length))

    def generate_signature(self, query_params=None, body=None):
        nonce = self.generate_random_nonce()
        timestamp = str(int(time.time() * 1000))
        digest_input = nonce + timestamp + self.api_key
        if query_params:
            sorted_params = sorted(query_params.items(), key=lambda x: x[0])
            query_params = "".join([f"{k}{v}" for k, v in sorted_params])
            digest_input += query_params
        if body:
            digest_input += json.dumps(body)

        digest = self.sha256_hex(digest_input)
        sign_input = digest + self.secret_key

        sign = self.sha256_hex(sign_input)
        return sign, nonce, timestamp

    def get_headers(self, query_params=None, body=None):
        sign, nonce, timestamp = self.generate_signature(query_params, body)
        headers = {
            "api-key": self.api_key,
            "nonce": nonce,
            "timestamp": timestamp,
            "sign": sign,
        }
        return headers

    def get_tickers(self, symbols: [str]):
        symbols = ",".join(symbols)
        url = f"{self.BASE_URL}{Endpoints.GET_TICKERS.value}"
        params = {"symbols": symbols}
        headers = self.get_headers(query_params=params)
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        if data["code"] != 0:
            raise Exception(f"API Error: {data['msg']}")

        return data["data"]

    def get_trading_pairs(self):
        url = f"{self.BASE_URL}{Endpoints.GET_TRADING_PAIRS.value}"
        headers = self.get_headers()
        response = requests.get(url, headers=headers)
        data = response.json()
        if data["code"] != 0:
            raise Exception(f"API Error: {data['msg']}")

        return data["data"]

    def get_klines(self, symbol, interval, start_time=None, end_time=None, limit=100, kline_type="LAST_PRICE"):
        """
        Fetch kline (candlestick) data from BitUnix API

        Args:
            symbol (str): Trading pair, e.g. "BTCUSDT"
            interval (str): Kline interval such as "5m", "1h", "1d" etc."
            start_time (int, optional): Start time in milliseconds (Unix timestamp)
            end_time (int, optional): End time in milliseconds (Unix timestamp)
            limit (int, optional): Number of klines to fetch, default 100, max 200
            kline_type (str, optional): Kline type, values: "LAST_PRICE", "MARK_PRICE"; default: "LAST_PRICE"

        Returns:
            dict: Response from the API containing kline data
        """
        url = f"{self.BASE_URL}/api/v1/futures/market/kline"
        s = symbol.upper()
        query_params = {
            "symbol": s,
            "interval": interval,
            "limit": limit,
            "type": kline_type
        }

        if start_time:
            query_params["startTime"] = start_time
        if end_time:
            query_params["endTime"] = end_time

        headers = self.get_headers(query_params=query_params)
        response = requests.get(url, headers=headers, params=query_params)

        data = response.json()
        if data["code"] != 0:
            raise Exception(f"API Error: {data['msg']}")

        return data["data"]

    def get_order_detail(self, order_id):
        url = f"{self.BASE_URL}{Endpoints.GET_ORDER_DETAIL.value}"
        query_params = {"orderId": order_id}
        headers = self.get_headers(query_params)
        response = requests.get(url, headers=headers, params=query_params)
        return response

    def get_history_orders(self, query_params=None):
        """Fetch history trades from BitUnix API"""
        url = f"{self.BASE_URL}{Endpoints.GET_HISTORY_ORDERS.value}"
        headers = self.get_headers(query_params)
        response = requests.get(url, headers=headers, params=query_params)
        if response.status_code != 200:
            raise Exception(response.text)
        data = response.json()
        if data["code"] != 0:
            raise Exception(f"API Error: {data['msg']}")
        return data["data"]["orderList"]

    def place_order(self, symbol, side, price, qty, tp_trigger_price, sl_trigger_price):
        symbol = symbol.replace("/", "").replace("USD", "USDT")
        body = {"symbol": symbol, "side": side,
                "price": price,
                "qty": qty,
                "orderType": "LIMIT",
                "reduceOnly": False,
                "clientId": "1110000aaa",
                "effect": "GTC",
                "tpPrice": tp_trigger_price,
                "tpStopType": "LAST_PRICE",
                "tpOrderType": "MARKET",
                "slPrice": sl_trigger_price,
                "slStopType": "LAST_PRICE",
                "slOrderType": "MARKET"}

        url = f"{self.BASE_URL}{Endpoints.PLACE_ORDER.value}"
        headers = self.get_headers(query_params=None, body=body)
        response = requests.post(url, headers=headers,
                                 data=json.dumps(body))
        if response.status_code != 200:
            raise Exception(response.text)
        return response.json()

    def get_pending_positions(self):
        # TODO Implement
        pass

    def get_position(self, asset):
        # TODO Implement
        pass

    def close_all_positions(self):
        # TODO Implement

        pass

    @staticmethod
    def save_trades_history_to_csv(trades_history):
        """Convert trade data to CSV format and include totals for Fee and realizedPNL"""
        filename = PROJECT_DIR + "/artifacts/history/trade_history.csv"

        # Initialize totals
        total_fee = 0
        total_realized_pnl = 0

        # Calculate totals first
        for trade in trades_history:
            fee = float(trade.get("fee", 0) or 0)  # Handle None or empty values
            realized_pnl = float(trade.get("realizedPNL", 0) or 0)  # Handle None or empty values

            total_fee += fee
            total_realized_pnl += realized_pnl

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header
            writer.writerow([
                "Date", "Time", "Symbol", "Quantity", "Price", "Side",
                "Leverage", "Fee", "realizedPNL"
            ])

            # Write trade data
            for trade in trades_history:
                # Convert timestamp to date and time
                trade_time = datetime.fromtimestamp(int(trade["ctime"]) / 1000)
                date_str = trade_time.strftime("%m/%d/%Y")
                time_str = trade_time.strftime("%H:%M:%S")

                # Determine side
                side = "Buy" if trade.get("side", "") == "BUY" else "Sell"
                fee = trade.get("fee", 0)
                realized_pnl = trade.get("realizedPNL", 0)

                # Write row with correct number of columns
                writer.writerow([
                    date_str,
                    time_str,
                    trade["symbol"],
                    trade["qty"],
                    trade["avgPrice"],
                    side,
                    trade["leverage"],
                    fee,
                    realized_pnl
                ])

            # Add a blank row to separate totals
            writer.writerow([])

            # Add totals row
            writer.writerow([
                "", "", "", "", "", "", "TOTALS:",
                f"{total_fee:.8f}", f"{total_realized_pnl:.8f}"
            ])

        logger.info(f"CSV file has been created: {filename}")
