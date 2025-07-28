import asyncio
import json
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import pytz
from google import genai
import plotly.graph_objects as go
from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.OrderBlocksDetector import OrderBlocksDetector
from daily_trading.bots.smc_bot.detectors.VolumeProfiler import VolumeProfileCollector
from daily_trading.settings import GEMINI_API_KEY, logger
from daily_trading.technicals.CryptoSentimentAnalyzer import CryptoSentimentAnalyzer

GEMINI_FLASH = "gemini-2.0-flash"
GEMINI_PRO = "gemini-2.5-pro"
TEST_PROMPT = """
        Tell me about SMC trading OB, very short 1 sentence (5 words). Your response must be in the format:{"message": "your message here",
            "entry": null,
            "stop_loss": null,
            "take_profit": null
        }
        """


@dataclass
class AssetData:
    symbol: str
    sentiment: dict
    fvg_signals: list
    bos_choch_signals: list
    pivot_high_points: list
    pivot_low_points: list
    ob_signals: list
    vol_profile: dict
    ohlc_1d: dict
    ohlc_1h: dict
    ohlc_4h: dict
    ohlc_15m: dict


@dataclass
class Ohlc:
    ohlc_1d: pd.DataFrame
    ohlc_4h: pd.DataFrame
    ohlc_1h: pd.DataFrame
    ohlc_15m: pd.DataFrame


class GeminiTradeAssistant:
    def __init__(self, symbol: str, model: str, ohlc: Ohlc,
                 main_tf="1h",
                 debug=False):
        self.api_key = GEMINI_API_KEY
        self.model = model
        self.symbol = symbol.upper()
        self.ohlc_1h = ohlc.ohlc_1h
        self.ohlc_15m = ohlc.ohlc_15m
        self.ohlc_4h = ohlc.ohlc_4h
        self.ohlc_1d = ohlc.ohlc_1d
        self.main_tf = main_tf
        self.client = genai.Client()
        self.debug = debug

    async def fetch_data(self) -> AssetData:
        """Fetch OHLC data and calculate ICT signals."""

        symbol = self.ohlc_1d["symbol"].iloc[0]

        # Select the last 24 candles for 1H and 96 candles for 15m
        ohlc_1d_subset = self.ohlc_1h.tail(10).to_dict(orient="records")
        ohlc_1h_subset = self.ohlc_1h.tail(24).to_dict(orient="records")
        ohlc_15m_subset = self.ohlc_15m.tail(96).to_dict(orient="records")
        ohlc_4h_subset = self.ohlc_4h.tail(18).to_dict(orient="records")

        detector_fvg = FvgDetector(main_tf="15min", pivot_period=5, left_bars=5)
        detector_bos = BosChochDetector(self.main_tf, show_bull=True, show_bear=True)
        high_low_detector = HighLowDetector(self.main_tf)
        ob_detector = OrderBlocksDetector(self.main_tf)
        vol_profiler = VolumeProfileCollector(main_tf="1h")
        analyzer = CryptoSentimentAnalyzer()
        sentiment_summary = analyzer.format_for_trading_prompt(symbol)
        logger.info(f"Sentiment: {sentiment_summary}")

        bos_results = detector_bos.calculate_bos_choch(self.ohlc_1h, plot=False)
        fvg_results = detector_fvg.calculate_fvg(self.ohlc_15m, plot=False)
        ob_results = ob_detector.calculate_order_blocks(self.ohlc_1h, plot=False)
        vol_profiler_results = vol_profiler.calculate_volume_profile(self.ohlc_4h, plot=False)
        del vol_profiler_results["fig"]

        return AssetData(
            symbol=symbol,
            sentiment=sentiment_summary,
            fvg_signals=fvg_results["signals"],
            bos_choch_signals=bos_results["signals"],
            pivot_high_points=high_low_detector.calculate_pivots(self.ohlc_1h)[0],
            pivot_low_points=high_low_detector.calculate_pivots(self.ohlc_1h)[1],
            ob_signals=ob_results["order_blocks"],
            ohlc_1d=ohlc_1d_subset,
            ohlc_1h=ohlc_1h_subset,
            ohlc_15m=ohlc_15m_subset,
            ohlc_4h=ohlc_4h_subset,
            vol_profile=vol_profiler_results
        )

    async def build_prompt(self) -> str:
        """Build the prompt for Gemini API based on ICT Smart Money Concepts."""
        data = await self.fetch_data()
        prompt = f"""
You are an elite ICT Smart Money Concepts trading assistant with an 80% win rate, specializing in institutional order flow, market structure, volume profile, and sentiment integration for high-probability crypto trades.

## Trading Methodology
- Market Structure Analysis (BOS/ChoCH) prioritizing 4H/1D
- Fair Value Gap (FVG), order blocks, liquidity sweeps, premium/discount zones
- Volume Profile for institutional activity
- Sentiment for contrarian edge or sizing
- 15M strictly for entry/SL/TP execution

## Input Data
- Crypto Coin: {data.symbol}
- OHLC 1D/4H/1H/15M: {data.ohlc_1d}, {data.ohlc_4h}, {data.ohlc_1h}, {data.ohlc_15m}
- 15M FVG Signals: {data.fvg_signals}
- 1H BOS/ChoCH: {data.bos_choch_signals}
- 1H Pivot Highs/Lows: {data.pivot_high_points}, {data.pivot_low_points}
- 1H Order Blocks: {data.ob_signals}
- 4H Volume Profile: {data.vol_profile}
- Sentiment: {data.sentiment}""" + """

## Analysis Framework
1. **Higher Timeframe Structure (4H/1D)**
   - Daily/4H trend, institutional levels, BOS/ChoCH
   - 1H confirms structure, swing points
   - Bias: Bullish/Bearish/Neutral (4H/1D priority; 1H if neutral)
   - If data limited, prioritize 4H/1D signals and sentiment for watch levels

2. **Volume Profile**
   - POC, VAH/VAL, high/low volume nodes
   - Identify institutional accumulation/distribution

3. **Sentiment Integration**
   - Fear & Greed, news sentiment, composite score
   - Secondary to 4H/1D structure; adjust sizing or confirm contrarian setups

4. **ICT Concepts**
   - FVGs, order blocks, liquidity zones, premium/discount
   - Institutional candle patterns at key levels

5. **Confluence**
   - 4H/1D sets bias; 1H confirms; 15M executes
   - Minimum three factors (4H/1D structure mandatory, plus two of FVG, order block, BOS/ChoCH, volume profile)

## Trading Decision Framework
### Entry Criteria (ALL required)
- Clear 4H/1D bias (or 1H if neutral)
- Three+ confluence factors
- Clean structure, 1:2 R:R minimum
- No conflicting signals
- Sentiment supports or provides contrarian edge

### Trade Setup
- Entry: Precise 15M price (FVG/order block)
- SL: Beyond 1H/4H structure invalidation
- TP: FVG, order block, or volume profile targets
- Sizing: 1-2% risk, sentiment-adjusted
- Trigger: 15M institutional signal

### Risk Management
- SL logic: Invalidates institutional thesis
- TP reasoning: Institutional levels
- Partial profits at key zones
- Optimal session timing
- Alternative scenarios for monitoring

## Response Format
### High-Probability Setup
```json
{{
    "message": "• **Setup Summary**: [Trade direction, institutional reasoning]
• **Key Confluence**: [Three main factors, 4H/1D focus]
• **Entry Logic**: [15M FVG/order block aligning with 4H/1D]
• **Volume Profile**: [POC/VAH/VAL support]
• **Sentiment Edge**: [Psychology impact]
• **Risk Management**: [SL reasoning, R:R]
• **Expected Move**: [TP based on institutional levels]
• **Trade Quality**: [High/Medium, institutional strength]",
    "entry": [exact_price],
    "stop_loss": [exact_price],
    "take_profit": [exact_price]
}}"""
        return prompt

    async def call_gemini_api(self, prompt: str):
        try:
            logger.warning("$$$$$ Making Gemini API Request $$$$")
            response = self.client.models.generate_content(
                model=self.model, contents=prompt
            )
            return response
        except Exception as e:
            logger.error(f"Gemini API failed: {e}")
            return {"error": f"API call failed: {e}"}

    async def analyze_asset_trade_data(self):
        my_prompt = await self.build_prompt()
        if self.debug:
            my_prompt = TEST_PROMPT
        response = await self.call_gemini_api(my_prompt)

        if "error" not in response:
            try:
                logger.info(f"AI Usage: {response.usage_metadata.total_token_count} Total Tokens.")
                content = response.candidates[0].content.parts[0].text
                raw_message = json.loads(content.replace("```json\n", "").replace("\n```", "").strip())
                logger.info(f"Gemini AI analysis complete - {json.dumps(raw_message, indent=1)}")
                formatted_ai_message = f"\n*** 🤖📊 AI Analysis | {self.symbol} ***\n" \
                                       f"\n{raw_message['message']}\n" \
                                       f"\nEntry: {raw_message['entry']}" \
                                       f"\nSL: {raw_message['stop_loss']}" \
                                       f"\nTP: {raw_message['take_profit']}"
                return formatted_ai_message, raw_message
            except (KeyError, json.JSONDecodeError) as e:
                logger.error(f"Error parsing Gemini response: {e}")
                return {"message": "Error processing Gemini response", "entry": None, "stop_loss": None,
                        "take_profit": None}
        return {"message": "Failed to retrieve Gemini analysis", "entry": None, "stop_loss": None, "take_profit": None}

    @staticmethod
    def update_plot_with_entry_data(fig: go.Figure, raw_message):
        entry = raw_message['entry']
        stop_loss = raw_message['stop_loss']
        take_profit = raw_message['take_profit']
        # Get x-axis range from candlestick data
        x_range = fig.data[0]['x']  # Assumes candlestick is first trace
        x_start, x_end = x_range[0], x_range[-1]

        # Calculate midpoint for text placement
        x_mid = x_range[len(x_range) // 2]

        # Add Entry line (Dashed Orange with centered text)
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[entry, entry],
            mode='lines',
            name='Entry',
            line=dict(color='orange', width=2, dash='dash')
        ))
        fig.add_annotation(
            x=x_mid,
            y=entry,
            text='Entry',
            showarrow=False,
            font=dict(size=18),
            xanchor='center',
            yanchor='middle'
        )

        # Add Stop Loss line (Dashed Red with centered text)
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[stop_loss, stop_loss],
            mode='lines',
            name='Stop Loss',
            line=dict(color='red', width=2, dash='dash')
        ))
        fig.add_annotation(
            x=x_mid,
            y=stop_loss,
            text='Stop Loss',
            showarrow=False,
            font=dict(size=18),
            xanchor='center',
            yanchor='middle'
        )

        # Add Take Profit line (Dashed Green with centered text)
        fig.add_trace(go.Scatter(
            x=[x_start, x_end],
            y=[take_profit, take_profit],
            mode='lines',
            name='Take Profit',
            line=dict(color='green', width=2, dash='dash')
        ))
        fig.add_annotation(
            x=x_mid,
            y=take_profit,
            text='Take Profit',
            showarrow=False,
            font=dict(size=18),
            xanchor='center',
            yanchor='middle'
        )
        return fig

    @staticmethod
    def within_ai_request_hours(start_hour=8, end_hour=22, end_minute=0):
        pacific = pytz.timezone('US/Pacific')
        now = datetime.now(pacific)
        current_time = now.hour * 60 + now.minute
        start_time = start_hour * 60
        end_time = end_hour * 60 + end_minute
        return start_time <= current_time <= end_time


async def main():
    from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
    logger.debug("Searching for assets that move most independently of Bitcoin.")
    symbol = "ERAUSDT"
    want_to_pay_for_analyses = True
    bot = SMCBotBase(symbols=[symbol], main_tf="1h")

    # Fetch OHLC data within async context

    ohlc_1d = await bot.fetch_ohlc_data(symbol, "1d", num_candles=10)
    ohlc_1h = await bot.fetch_ohlc_data(symbol, "1h", num_candles=120)
    ohlc_15m = await bot.fetch_ohlc_data(symbol, "15m", num_candles=100)
    ohlc_4h = await bot.fetch_ohlc_data(symbol, "4h", num_candles=18)
    ohlc = Ohlc(ohlc_1d, ohlc_4h, ohlc_1h, ohlc_15m)

    ai_agent = GeminiTradeAssistant(symbol=symbol, model=GEMINI_PRO, ohlc=ohlc, debug=False)

    if want_to_pay_for_analyses:
        formatted_ai_message, raw_message = await ai_agent.analyze_asset_trade_data()
        logger.info(formatted_ai_message)
    else:
        prompt = await ai_agent.build_prompt()
        logger.info(prompt)


if __name__ == "__main__":
    asyncio.run(main())
