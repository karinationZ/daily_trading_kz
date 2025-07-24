import asyncio
import json
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import pytz
from google import genai

from daily_trading.bots.smc_bot.detectors.BosChochDetector import BosChochDetector
from daily_trading.bots.smc_bot.detectors.FvgDetector import FvgDetector
from daily_trading.bots.smc_bot.detectors.HighLowDetector import HighLowDetector
from daily_trading.bots.smc_bot.detectors.OrderBlocksDetector import OrderBlocksDetector
from daily_trading.settings import GEMINI_API_KEY, logger

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
    fvg_signals: list
    bos_choch_signals: list
    pivot_high_points: list
    pivot_low_points: list
    ob_signals: list
    ohlc_1h: list
    ohlc_4h: list
    ohlc_15m: list


class GeminiTradeAssistant:
    def __init__(self, symbol: str, model: str, ohlc_4h: pd.DataFrame, ohlc_1h: pd.DataFrame, ohlc_15m: pd.DataFrame,
                 main_tf="1h",
                 debug=False):
        self.api_key = GEMINI_API_KEY
        self.model = model
        self.symbol = symbol.upper()
        self.ohlc_1h = ohlc_1h
        self.ohlc_15m = ohlc_15m
        self.ohlc_4h = ohlc_4h
        self.main_tf = main_tf
        self.client = genai.Client()
        self.debug = debug

    async def fetch_data(self) -> AssetData:
        """Fetch OHLC data and calculate ICT signals."""

        # Select the last 24 candles for 1H and 96 candles for 15m
        ohlc_1h_subset = self.ohlc_1h.tail(24).to_dict(orient="records")
        ohlc_15m_subset = self.ohlc_15m.tail(96).to_dict(orient="records")
        ohlc_4h_subset = self.ohlc_4h.tail(18).to_dict(orient="records")

        detector_fvg = FvgDetector(main_tf="15min", pivot_period=5, left_bars=5)
        detector_bos = BosChochDetector(self.main_tf, show_bull=True, show_bear=True)
        high_low_detector = HighLowDetector(self.main_tf)
        ob_detector = OrderBlocksDetector(self.main_tf)

        bos_results = detector_bos.calculate_bos_choch(self.ohlc_1h, plot=False)
        fvg_results = detector_fvg.calculate_fvg(self.ohlc_15m, plot=False)
        ob_results = ob_detector.calculate_order_blocks(self.ohlc_1h, plot=False)

        return AssetData(
            fvg_signals=fvg_results["signals"],
            bos_choch_signals=bos_results["signals"],
            pivot_high_points=high_low_detector.calculate_pivots(self.ohlc_1h)[0],
            pivot_low_points=high_low_detector.calculate_pivots(self.ohlc_1h)[1],
            ob_signals=ob_results["order_blocks"],
            ohlc_1h=ohlc_1h_subset,
            ohlc_15m=ohlc_15m_subset,
            ohlc_4h=ohlc_4h_subset
        )

    async def build_prompt(self) -> str:
        """Build the prompt for Gemini API based on ICT Smart Money Concepts."""
        data = await self.fetch_data()
        prompt = f"""
You are an elite ICT (Inner Circle Trader) Smart Money Concepts trading assistant with a proven 80% win rate. 
Your expertise lies in analyzing institutional order flow and market structure to identify high-probability trading opportunities.

## Your Trading Methodology
You follow strict ICT Smart Money Concepts including:
- Market Structure Analysis (BOS/ChoCH)
- Fair Value Gap (FVG) identification and trading
- Liquidity sweeps and stop hunts
- Order block analysis
- Premium/Discount array mapping
- Institutional candle patterns

## Input Data:
1. **OHLC 1H Data**: {data.ohlc_1h}
2. **OHLC 4H Data**: {data.ohlc_4h}
3. **OHLC 15MIN Data** - Entry timeframe for precise execution: {data.ohlc_15m}
4. **FVG Signals** - Fair Value Gap locations and classifications (bullish/bearish/mitigation): {data.fvg_signals}
5. **BOS (Break of Structure) Signals**: {data.bos_choch_signals}
6. **Pivot Highs** - Recent swing highs and potential resistance levels: {data.pivot_high_points}
7. **Pivot Lows** - Recent swing lows and potential support levels: {data.pivot_low_points}
8. **Order Blocks** - Institutional order zones: {data.ob_signals}""" + """

### Market Analysis
- **Higher Timeframe Bias**: [Bullish/Bearish/Neutral] based on 1H structure
- **Market Structure**: Current state of BOS/ChoCH and what it indicates
- **Key Levels**: Most relevant pivot highs/lows acting as liquidity zones

### Trade Setup
- **Entry Point**: Exact price level with reasoning
- **Stop Loss (SL)**: Precise placement with invalidation logic
- **Take Profit (TP)**: Target based on structure/FVG levels
- **Risk:Reward Ratio**: Calculated R:R for the setup

### Trade Rationale
Explain your analysis using ICT concepts:
- Why this setup aligns with smart money flow
- How FVGs and market structure support the trade
- What institutional behavior you're capitalizing on
- Confluence factors that increase probability

### Response Output Format
REQUIRED: You must always return your analysis as a JSON object in this exact format:
For HIGH-PROBABILITY SETUPS ONLY:
{
"message": "Clear, concise trading analysis. Keep it simple and informative - explain the setup, why you're taking it, key levels in plain language. Use bullet points. 
Avoid jargon overload. Be precise and short.",
    "entry": int,
    "stop_loss": int,
    "take_profit": int
}
For NO CLEAR SETUP:
{
"message": "No clear entry opportunity. [Explain why - conflicting signals, choppy structure, lack of confluence, waiting for better setup, etc.]",
    "entry": null,
    "stop_loss": null,
    "take_profit": null
}

Entry Criteria - ALL must be present:
- Clear higher timeframe bias (1H structure)
- Multiple confluence (FVG + Order Block + BOS/ChoCH alignment)
- Clean market structure with defined levels
- Risk:Reward minimum 1:2
- No conflicting signals

Message Content Requirements:
Write in clear, simple language covering:
- Setup Summary: What you see and trade direction (or why no trade). Use bullet points.
- Key Reasons: 2-3 main confluence factors OR reasons to wait
- Entry Logic: Why this exact price level OR what you're waiting for
- Risk Management: SL reasoning and R:R ratio (if trading)
- Expected Move: TP target and why (if trading)

Keep it concise, informative, and easy to understand. Better to wait than force low-quality trades.
Trading Rules You Follow:
- Only trade with clear market structure bias
- Require multiple confluences for entry
- Never risk more than 1-2% per trade
- Always define invalidation levels before entry
- Focus on reaction at key liquidity levels
- Prioritize quality setups over quantity
- NO TRADE ZONE: If there's no clear, high-probability setup, explicitly state "No clear entry opportunity" and explain why

Example Response Style:
"Based on the 1H bullish market structure and recent ChoCH, I'm looking for long opportunities. The 15min chart shows a bullish FVG at 1.2650 that aligns with a validated bullish order block from the previous session and the pivot low, creating a high-probability buy zone. Smart money likely accumulated in this order block, and I expect institutional buying to defend this level when price returns..."

Remember: Your 80% win rate comes from patience, discipline, and only taking the highest probability setups that align with institutional order flow. Quality over quantity is your mantra. When in doubt, DON'T TRADE - wait for the perfect setup with clear confluence and structure.
"""
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
    def can_send_request(start_hour=6, end_hour=23, end_minute=0):
        pacific = pytz.timezone('US/Pacific')
        now = datetime.now(pacific)
        current_time = now.hour * 60 + now.minute
        start_time = start_hour * 60
        end_time = end_hour * 60 + end_minute
        return start_time <= current_time <= end_time


async def main():
    from daily_trading.bots.smc_bot.bots.BaseBot import SMCBotBase
    logger.debug("Searching for assets that move most independently of Bitcoin.")
    symbol = "HUMAUSDT"
    want_to_pay_for_analyses = True
    bot = SMCBotBase(symbols=[symbol], main_tf="1h")

    # Fetch OHLC data within async context
    ohlc_1h = await bot.fetch_ohlc_data(symbol, "1h", num_candles=120)
    ohlc_15m = await bot.fetch_ohlc_data(symbol, "15m", num_candles=100)
    ohlc_4h = await bot.fetch_ohlc_data(symbol, "4h", num_candles=18)

    ai_agent = GeminiTradeAssistant(symbol=symbol, model=GEMINI_PRO, ohlc_4h=ohlc_4h, ohlc_1h=ohlc_1h,
                                    ohlc_15m=ohlc_15m, debug=False)

    if want_to_pay_for_analyses:
        formatted_ai_message, raw_message = await ai_agent.analyze_asset_trade_data()
        logger.info(formatted_ai_message)
    else:
        prompt = await ai_agent.build_prompt()  # Assuming this is synchronous
        logger.info(prompt)


if __name__ == "__main__":
    asyncio.run(main())
